import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

print("right now using device num:", os.environ['CUDA_VISIBLE_DEVICES'])
import pandas as pd
from eval_utils import compute_mu_scores, compute_fq_scores, read_file
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config_eval
from peft import PeftModel
from template import LLAMA3_CHAT_TEMPLATE, qwen_chat_template
import warnings
from transformers import logging as hf_logging
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from scipy.stats import hmean
import numpy as np
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

hf_logging.set_verbosity_error()

warnings.filterwarnings("ignore")

cfg = Config_eval()
print('loading forget and test set')


results_json_path = '/home/praveen/eacl/results/scores/all_results2.json'
if Path(results_json_path).exists():
    with open(results_json_path, 'r') as f:
        all_results = json.load(f)
else:
    all_results = {}

if cfg.dataset == 'muse':
    cfg.forget_path = '/home/praveen/eacl/data/muse_forget.parquet'
if cfg.dataset == 'bio':
    cfg.forget_path = '/home/praveen/nnomp/data/wmdp_bio.parquet'

if cfg.model_name == 'llama':
    cfg.model_id = 'meta-llama/Llama-3.1-8B-Instruct'
    cfg.test_path = '/home/praveen/eacl/data/llama_muse_test.parquet'
    
if cfg.model_name == 'qwen':
    cfg.model_id = 'Qwen/Qwen2.5-3B-Instruct'
    cfg.test_path = '/home/praveen/eacl/data/qwen_muse_test.parquet'
    
if cfg.selection == 'high':
    cfg.retain_path = f'/home/praveen/eacl/data/retain_sets/{cfg.model_name}_{cfg.loss_type}_{cfg.dataset}_{cfg.selection}.parquet'


metrics = [
     ("Selection type", f'{cfg.selection}'),
     ("Dataset", f'{cfg.dataset}'),
    ("Unlearning Loss", f'{cfg.loss_type}'),
    ("Forget Path",   f'{cfg.forget_path}'),
    ("Test Path",   f'{cfg.test_path}'),
    ("Audit Path",   f'{cfg.audit_path}'),
    ("Retain Path",   f'{cfg.retain_path}'),
    ('Model id', f'{cfg.model_id}'),
]

print("\n\n============ Check List ============\n")
print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="github"))


forget = read_file(cfg.forget_path)
test_data = read_file(cfg.test_path)
audit_data = read_file(cfg.audit_path)
forget = forget.loc[forget['type'] == 'forget']


device = 'cuda'

print('forget shape:', forget.shape)
print('\ntest shape:', test_data.shape)

# ---- Loading Tokenizer -----------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token 

print(f'\n\nConducting evaluation on: {cfg.model_name}_{cfg.loss_type}_{cfg.dataset}_{cfg.selection}')
print(f'\nAdaptor path:', cfg.adaptor_path)

# ---- Loading model -----------

if cfg.loss_type == 'pre_unlearning':
     base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = hf_token , device_map = "auto", torch_dtype=torch.bfloat16)
     model = PeftModel.from_pretrained(base_model, cfg.adaptor_path, device_map="auto", torch_dtype=torch.bfloat16)
else:
     print('loading peft model from ', cfg.adaptor_path)
     base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = hf_token, device_map = "auto", torch_dtype=torch.bfloat16) 
     model = PeftModel.from_pretrained(base_model, cfg.adaptor_path, device_map="auto", torch_dtype=torch.bfloat16) 


# ------- creating template format for tokenization --------
def make_template_format(df, model_name):
     if model_name == 'llama':
        df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
     elif model_name == 'qwen':
        df['question'] = df['question'].apply(lambda x : qwen_chat_template.format(question = x))
     return df


forget_2 = make_template_format(forget, cfg.model_name)
test_data = make_template_format(test_data, cfg.model_name)
audit_data = make_template_format(audit_data, cfg.model_name)
forget_2 = forget_2[['id','question', 'answer', 'num_tokens', 'type']]
print(forget_2.columns)


model_name = cfg.retriever_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer(model_name, device=device)

print('\ncalculating forget efficacy')

forget,all_scores_fe, fq, f_ppl  = compute_fq_scores(df=forget_2, model=model, tokenizer=tokenizer, device=device)
print('\nforget quality', fq)

print('\ncalculating model utility on test data')
test,all_scores_mu, mu, rt_ppl  = compute_mu_scores(df=test_data, model=model, tokenizer=tokenizer, embedding_model=embedding_model, device=device)

print('\nCalculating utility on audit data')
audit, all_scores_au, au, a_ppl  = compute_mu_scores(df=audit_data, model=model, tokenizer=tokenizer, embedding_model=embedding_model, device=device)

retain_data = read_file(cfg.retain_path)
retain_ids = retain_data['id'].tolist()
audit_ids = audit_data['id'].tolist()

def calculate_audit_score(audit, count, non_overlap_count):
    audit_scores = []
    for i, row in audit.iterrows():
        rouge_l = row['rouge_l']
        probs = row['probs']
        cos_sims = row['cos_sims']
        hmean = hmean([rouge_l, probs, cos_sims])
        audit_scores.append(hmean)
    ind_utility = np.mean(audit_scores)
    full_len = count + non_overlap_count
    clean_coverage = non_overlap_count / full_len
    audit_utility = ind_utility * clean_coverage
    return audit_utility, ind_utility, clean_coverage


count = len(set(retain_ids).intersection(audit_ids))
print(count)
non_overlap_count = len(audit_ids) - count

audit_data = audit.loc[~audit['id'].isin(retain_ids)]
audit_utility, ind_utility, clean_coverage = calculate_audit_score(audit_data, count, non_overlap_count)



if cfg.loss_type == 'pre_unlearning':
    forget.to_parquet(f'/home/praveen/eacl/results/datasets/audit_res/{cfg.model_name}_forget_{cfg.loss_type}_{cfg.dataset}.parquet', index = False)
    test.to_parquet(f'/home/praveen/eacl/results/datasets/audit_res/{cfg.model_name}_test_{cfg.loss_type}_{cfg.dataset}.parquet', index = False)
    audit.to_parquet(f'/home/praveen/eacl/results/datasets/audit_res/{cfg.model_name}_audit_{cfg.loss_type}_{cfg.dataset}.parquet', index = False)
else:
    forget.to_parquet(f'/home/praveen/eacl/results/datasets/audit_res/{cfg.model_name}_forget_{cfg.loss_type}_{cfg.dataset}_{cfg.selection}.parquet', index = False)
    test.to_parquet(f'/home/praveen/eacl/results/datasets/audit_res/{cfg.model_name}_test_{cfg.loss_type}_{cfg.dataset}_{cfg.selection}.parquet', index = False)
    audit.to_parquet(f'/home/praveen/eacl/results/datasets/audit_res/{cfg.model_name}_audit_{cfg.loss_type}_{cfg.dataset}_{cfg.selection}.parquet', index = False)

metrics = [
     ("experiment_type", f'{cfg.loss_type}_{cfg.dataset}_{cfg.selection}'),
    ("FQ",      fq.item()),
    ("Test",   mu.item()),
    ("AU",   audit_utility),
    ("Ind",   ind_utility),
    ("Clean",   clean_coverage),
    ("PPL-F",   f_ppl.item()),
    ("PPL-T",rt_ppl.item()),
]

try:
    from tabulate import tabulate
    print("\n\n============ ALL RESULTS ============\n")
    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="github"))
    
except ImportError:
    col1_w = max(len(name) for name, _ in metrics)
    col2_w = max(len(f"{val:.4f}") for _, val in metrics)

    print("\n\n============ ALL RESULTS ============\n")
    print(f"| {'Metric'.ljust(col1_w)} | {'Value'.rjust(col2_w)} |")
    print(f"|{'-'*(col1_w+2)}|{'-'*(col2_w+2)}|")
    for name, val in metrics:
        print(f"| {name.ljust(col1_w)} | {val:>{col2_w}.4f} |")


if cfg.loss_type == 'pre_unlearning':
    results = {f'{cfg.model_name}_{cfg.loss_type}_{cfg.dataset}': 
            {"FQ":      fq.item(),
            "Test Utility":   mu.item(),
            "Audit Utility":   audit_utility,
            "Ind Utility":   ind_utility,
            "Clean Coverage":   clean_coverage,
            'forget_scores' : all_scores_fe.tolist(),
            'test_scores': all_scores_mu.tolist(),
            'audit_scores': all_scores_au.tolist(),
            "PPL-F" : f_ppl.item(),
            "PPL-T":  rt_ppl.item(),
            "PPL-A":  a_ppl.item(),
            }}
else:
    results = {f'{cfg.model_name}_{cfg.loss_type}_{cfg.dataset}_{cfg.selection}': 
                {"FQ":      fq.item(),
                "Test Utility":   mu.item(),
                "Audit Utility":   audit_utility,
                "Ind Utility":   ind_utility,
                "Clean Coverage":   clean_coverage,
                'forget_scores' : all_scores_fe.tolist(),
                'test_scores': all_scores_mu.tolist(),
                "PPL-F" : f_ppl.item(),
                "PPL-T":  rt_ppl.item(),
                "PPL-A":  a_ppl.item(),
                }}

all_results.update(results)

with open(results_json_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\nAppended results to {results_json_path}')
print(f'Total experiments logged so far: {len(all_results)}')