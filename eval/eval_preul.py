import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

print("right now using device num:", os.environ['CUDA_VISIBLE_DEVICES'])
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from peft import PeftModel
from scipy.stats import hmean as hmean_fn
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

from config import Config_eval
from eval_utils import compute_fq_scores, compute_mu_scores
from template import LLAMA3_CHAT_TEMPLATE, qwen_chat_template, olmo_chat_template

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
hf_logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

results_json_path = '/home/praveen/eacl/results/scores/all_results_olmo_muse_pre.json'
if Path(results_json_path).exists():
    with open(results_json_path, 'r') as f:
        all_results = json.load(f)
else:
    all_results = {}


def read_file(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.json'):
        df = pd.read_json(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    return df


def make_template_format(df, model_name):
    if model_name == 'llama':
        df['question'] = df['question'].apply(lambda x: LLAMA3_CHAT_TEMPLATE.format(question=x))
    elif model_name == 'qwen':
        df['question'] = df['question'].apply(lambda x: qwen_chat_template.format(question=x))
    elif model_name == 'olmo':
        df['question'] = df['question'].apply(lambda x: olmo_chat_template.format(question=x))
    return df


def calculate_audit_score(audit_clean_df, count, non_overlap_count):
    row_scores = []
    for _, row in audit_clean_df.iterrows():
        h = hmean_fn([row['rouge_l'], row['probs'], row['cos_sims']])
        row_scores.append(h)

    ind_utility = float(np.mean(row_scores)) if row_scores else 0.0
    total = count + non_overlap_count
    clean_coverage = non_overlap_count / total if total > 0 else 0.0
    audit_utility = ind_utility * clean_coverage
    return audit_utility, ind_utility, clean_coverage


def run_eval(model_name):
    """Runs the full pre_unlearning eval for a single model and returns a metrics dict."""
    cfg = Config_eval()
    cfg.model_name = model_name
    cfg.loss_type = 'pre_unlearning'
    cfg.dataset = 'muse'

    if cfg.dataset == 'muse':
        cfg.forget_path = '/home/praveen/eacl/data/muse_forget.parquet'
    if cfg.dataset == 'bio':
        cfg.forget_path = '/home/praveen/eacl/data/bio_forget.parquet'


    if cfg.model_name == 'llama':
        cfg.model_id = 'meta-llama/Llama-3.1-8B-Instruct'
        cfg.test_path = f'/home/praveen/eacl/data/{cfg.model_name}_{cfg.dataset}_test.parquet'
        cfg.adaptor_path = f'/home/praveen/eacl/outputs/{cfg.model_name}_{cfg.dataset}'
    if cfg.model_name == 'qwen':
        cfg.model_id = 'Qwen/Qwen2.5-3B-Instruct'
        cfg.test_path = f'/home/praveen/eacl/data/{cfg.model_name}_{cfg.dataset}_test.parquet'
        cfg.adaptor_path = f'/home/praveen/eacl/outputs/{cfg.model_name}_{cfg.dataset}'
    if cfg.model_name == 'olmo':
        cfg.model_id = 'allenai/OLMo-2-0425-1B-Instruct'
        cfg.test_path = f'/home/praveen/eacl/data/{cfg.model_name}_{cfg.dataset}_test.parquet'
        cfg.adaptor_path = f'/home/praveen/eacl/outputs/{cfg.model_name}_{cfg.dataset}'

    # NOTE: adjust count / non_overlap_count if these need to be computed per-run
    # rather than fixed constants.
    cfg.audit_path = f'/home/praveen/eacl/data/{cfg.model_name}_{cfg.dataset}_audit.parquet'
    audit_count = 0
    audit_non_overlap_count = 200
    

    checklist = [
        ("Dataset", f'{cfg.dataset}'),
        ("Unlearning Loss", f'{cfg.loss_type}'),
        ("Forget Path", f'{cfg.forget_path}'),
        ("Test Path", f'{cfg.test_path}'),
        ("Audit Path", f'{cfg.audit_path}'),
        ('Model id', f'{cfg.model_id}'),
        ('Adaptor Path', f'{cfg.adaptor_path}'),
    ]
    print(f"\n\n============ Check List ({model_name}) ============\n")
    print(tabulate(checklist, headers=["Metric", "Value"], tablefmt="github"))

    forget = read_file(cfg.forget_path)
    test_data = read_file(cfg.test_path)
    audit_data = read_file(cfg.audit_path)
    forget = forget.loc[forget['type'] == 'forget']

    device = 'cuda'
    print('forget shape:', forget.shape)
    print('\ntest shape:', test_data.shape)
    print('\naudit shape:', audit_data.shape)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print(f'\n\nConducting evaluation on: {cfg.loss_type}_{cfg.dataset}_{cfg.model_name}')

    print('loading peft model from ', cfg.adaptor_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, token=hf_token, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(
        base_model, cfg.adaptor_path, device_map="auto", torch_dtype=torch.bfloat16
    )

    forget_2 = make_template_format(forget, cfg.model_name)
    test_data = make_template_format(test_data, cfg.model_name)
    audit_data = make_template_format(audit_data, cfg.model_name)
    forget_2 = forget_2[['id', 'question', 'answer', 'num_tokens', 'type']]
    print(forget_2.columns)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(cfg.retriever_model, device=device)

    print('\ncalculating forget efficacy')
    forget, all_scores_fe, fq, f_ppl = compute_fq_scores(
        df=forget_2, model=model, tokenizer=tokenizer, device=device
    )
    print('\nforget quality', fq)

    print('\ncalculating model utility')
    retain, all_scores_mu, mu, rt_ppl = compute_mu_scores(
        df=test_data, model=model, tokenizer=tokenizer, embedding_model=embedding_model, device=device
    )

    print('\ncalculating audit utility')
    audit, all_scores_au, au, a_ppl = compute_mu_scores(
        df=audit_data, model=model, tokenizer=tokenizer, embedding_model=embedding_model, device=device
    )
    audit_utility, ind_utility, clean_coverage = calculate_audit_score(
        audit, audit_count, audit_non_overlap_count
    )
    print('\naudit utility', audit_utility, '| ind utility', ind_utility, '| clean coverage', clean_coverage)

    forget.to_parquet(
        f'/home/praveen/eacl/results/datasets/forget_{cfg.loss_type}_{cfg.dataset}_{cfg.model_name}.parquet',
        index=False,
    )
    retain.to_parquet(
        f'/home/praveen/eacl/results/datasets/test_{cfg.loss_type}_{cfg.dataset}_{cfg.model_name}.parquet',
        index=False,
    )
    audit.to_parquet(
        f'/home/praveen/eacl/results/datasets/audit_{cfg.loss_type}_{cfg.dataset}_{cfg.model_name}.parquet',
        index=False,
    )

    result_key = f'{cfg.model_name}_{cfg.loss_type}_{cfg.dataset}'
    result_entry = {
        "FQ": fq.item(),
        "Test Utility": mu.item(),
        "Audit Utility": audit_utility,
        "Audit Ind Utility": ind_utility,
        "Audit Clean Coverage": clean_coverage,
        'forget_scores': all_scores_fe.tolist(),
        'test_scores': all_scores_mu.tolist(),
        'audit_scores': all_scores_au.tolist(),
        "PPL-F": f_ppl.item(),
        "PPL-T": rt_ppl.item(),
        "PPL-A": a_ppl.item(),
    }
    all_results[result_key] = result_entry

    metrics_row = (
        "experiment_type", result_key, fq.item(), mu.item(), audit_utility,
        f_ppl.item(), rt_ppl.item(), a_ppl.item(),
    )

    # ---- free GPU memory before the next model loads ----
    del model, base_model, embedding_model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics_row


if __name__ == "__main__":
    all_rows = []
    for model_name in ['olmo']:
        row = run_eval(model_name)
        all_rows.append(row)

    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n\n============ ALL RESULTS (qwen + llama) ============\n")
    table_rows = [(r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in all_rows]
    print(tabulate(
        table_rows,
        headers=["Experiment", "FQ", "Test Utility", "Audit Utility", "PPL-F", "PPL-T", "PPL-A"],
        tablefmt="github",
    ))

    print(f'\nAppended results to {results_json_path}')
    print(f'Total experiments logged so far: {len(all_results)}')