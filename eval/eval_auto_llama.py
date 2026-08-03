import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from scipy.stats import hmean as hmean_fn
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
from sentence_transformers import SentenceTransformer
from peft import PeftModel

from eval_utils import compute_fq_scores, compute_mu_scores, read_file
from template import LLAMA3_CHAT_TEMPLATE, qwen_chat_template, olmo_chat_template

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

out_dir = '/home/praveen/eacl/results/audit_res'

LOSS_TYPES = ['ga', 'npo', 'snpo']
SELECTIONS = ['randri']
#SELECTIONS = ['raslik', 'emb', 'ri']#, 'emb']
#SELECTIONS = ['ri']

# ---------------- Config ----------------
class Config_eval:
    def __init__(self, selection='high', loss_type='ga', model_name='llama', dataset='muse'):
        self.loss_type = loss_type
        self.model_name = model_name
        self.dataset = dataset
        self.selection = selection
        self.max_length = 512
        self.retriever_model = 'thenlper/gte-small'
        self._set_model_id()
        self._set_paths()

    def _set_model_id(self):
        if self.model_name == 'llama':
            self.model_id = 'meta-llama/Llama-3.1-8B-Instruct'
        elif self.model_name == 'qwen':
            self.model_id = 'Qwen/Qwen2.5-3B-Instruct'
        elif self.model_name == 'olmo':
            self.model_id = 'allenai/OLMo-2-0425-1B-Instruct'

    def _set_paths(self):
        self.adaptor_path = (
            f'/home/praveen/eacl/outputs/repair/'
            f'{self.model_name}_{self.loss_type}_{self.dataset}_{self.selection}'
        )
        if self.dataset == 'muse':
            self.forget_path = '/home/praveen/eacl/data/muse_forget.parquet'
            self.test_path = f'/home/praveen/eacl/data/{self.model_name}_muse_test.parquet'
        elif self.dataset == 'bio':
            self.forget_path = '/home/praveen/eacl/data/bio_forget.parquet'
            self.test_path = f'/home/praveen/eacl/data/{self.model_name}_bio_test.parquet'

        
        self.audit_path = f'/home/praveen/eacl/data/{self.model_name}_{self.dataset}_audit.parquet'

        if self.selection == 'high':
            self.retain_path = (
                f'/home/praveen/eacl/data/retain_sets/'
                f'{self.model_name}_{self.loss_type}_{self.dataset}_{self.selection}.parquet'
            )
        elif self.selection == 'emb':
            self.retain_path = (
                f'/home/praveen/eacl/data/retain_sets/'
                f'{self.model_name}_{self.dataset}_{self.selection}.parquet'
            )
        elif self.selection == 'ri':
            self.retain_path = (
                f'/home/praveen/eacl/data/retain_sets/'
                f'{self.model_name}_{self.loss_type}_{self.dataset}_{self.selection}.parquet'
            )
        elif self.selection == 'raslik':
            self.retain_path = (
                f'/home/praveen/eacl/data/retain_sets/'
                f'{self.model_name}_{self.dataset}_{self.selection}.parquet'
            )
        elif self.selection == 'rand':
            self.retain_path = (
                f'/home/praveen/eacl/data/retain_sets/'
                f'{self.model_name}_{self.dataset}_{self.selection}.parquet'
            )
        elif self.selection == 'randri':
            self.retain_path = (
                f'/home/praveen/eacl/data/retain_sets/'
                f'{self.model_name}_{self.loss_type}_{self.dataset}_{self.selection}.parquet'
            )


# ---------------- Audit score (fixed) ----------------
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


def make_template_format(df, model_name):
    df = df.copy()
    if model_name == 'llama':
        df['question'] = df['question'].apply(lambda x: LLAMA3_CHAT_TEMPLATE.format(question=x))
    elif model_name == 'qwen':
        df['question'] = df['question'].apply(lambda x: qwen_chat_template.format(question=x))
    elif model_name == 'olmo':
        df['question'] = df['question'].apply(lambda x: olmo_chat_template.format(question=x))
    return df


def run_one(loss_type, selection, model, tokenizer, embedding_model,
            forget_2, test_data, audit_data, model_name, dataset, all_results):
    cfg = Config_eval(selection=selection, loss_type=loss_type,
                       model_name=model_name, dataset=dataset)

    print(f"\n\n============ Running eval: loss_type={loss_type}, selection={selection} ============\n")
    print(tabulate([
        ("Loss type", cfg.loss_type),
        ("Selection", cfg.selection),
        ("Adaptor", cfg.adaptor_path),
        ("Retain Path", cfg.retain_path),
    ], headers=["Field", "Value"], tablefmt="github"))

    adapter_name = f"adapter_{loss_type}_{selection}"
    if adapter_name not in model.peft_config:
        model.load_adapter(cfg.adaptor_path, adapter_name=adapter_name)
    model.set_adapter(adapter_name)

    print('\ncalculating forget efficacy')
    forget, all_scores_fe, fq, f_ppl = compute_fq_scores(
        df=forget_2, model=model, tokenizer=tokenizer, device=device)

    print('\ncalculating model utility on test data')
    test, all_scores_mu, mu, rt_ppl = compute_mu_scores(
        df=test_data, model=model, tokenizer=tokenizer,
        embedding_model=embedding_model, device=device)

    print('\ncalculating utility on audit data')
    audit, all_scores_au, au, a_ppl = compute_mu_scores(
        df=audit_data, model=model, tokenizer=tokenizer,
        embedding_model=embedding_model, device=device)

    retain_data = read_file(cfg.retain_path)
    retain_ids = set(retain_data['id'].tolist())
    audit_ids = audit['id'].tolist()

    count = len(retain_ids.intersection(audit_ids))
    print(f'Number of overlapping ids: {count}')
    non_overlap_count = len(audit_ids) - count
    audit_clean = audit.loc[~audit['id'].isin(retain_ids)]

    audit_utility, ind_utility, clean_coverage = calculate_audit_score(
        audit_clean, count, non_overlap_count)

    suffix = f'{cfg.loss_type}_{cfg.dataset}_{cfg.selection}'
    forget.to_parquet(f'{out_dir}/{cfg.model_name}_forget_{suffix}.parquet', index=False)
    test.to_parquet(f'{out_dir}/{cfg.model_name}_test_{suffix}.parquet', index=False)
    audit.to_parquet(f'{out_dir}/{cfg.model_name}_audit_{suffix}.parquet', index=False)

    metrics = [
        ("experiment_type", suffix),
        ("FQ", fq.item()),
        ("Test", mu.item()),
        ("AU", audit_utility),
        ("Ind", ind_utility),
        ("Clean", clean_coverage),
        ("PPL-F", f_ppl.item()),
        ("PPL-T", rt_ppl.item()),
        ('overlap', count),
    ]
    print(f"\n\n============ RESULTS: {loss_type} / {selection} ============\n")
    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="github"))

    key = f'{cfg.model_name}_{cfg.loss_type}_{cfg.dataset}_{cfg.selection}'
    all_results[key] = {
        "FQ": fq.item(),
        "Test Utility": mu.item(),
        "Audit Utility": audit_utility,
        "Ind Utility": ind_utility,
        "Clean Coverage": clean_coverage,
        "forget_scores": all_scores_fe.tolist(),
        "test_scores": all_scores_mu.tolist(),
        "audit_scores": all_scores_au.tolist(),
        "PPL-F": f_ppl.item(),
        "PPL-T": rt_ppl.item(),
        "PPL-A": a_ppl.item(),
        "overlap": count
    }

    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\nAppended results to {results_json_path}')
    print(f'Total experiments logged so far: {len(all_results)}')
    return all_results


# ---------------- Main ----------------
MODEL_NAME = 'llama'
DATASET = 'muse'
results_json_path = f'/home/praveen/eacl/results/scores/llama_muse_randri.json'


cfg0 = Config_eval(selection='high', loss_type='ga', model_name=MODEL_NAME, dataset=DATASET)

if Path(results_json_path).exists():
    with open(results_json_path, 'r') as f:
        all_results = json.load(f)
else:
    all_results = {}

tokenizer = AutoTokenizer.from_pretrained(cfg0.model_id)
tokenizer.pad_token = tokenizer.eos_token

forget_raw = read_file(cfg0.forget_path)
forget_raw = forget_raw.loc[forget_raw['type'] == 'forget']
test_raw = read_file(cfg0.test_path)
audit_raw = read_file(cfg0.audit_path)

forget_2 = make_template_format(forget_raw, MODEL_NAME)[
    ['id', 'question', 'answer', 'num_tokens', 'type']]
test_data = make_template_format(test_raw, MODEL_NAME)
audit_data = make_template_format(audit_raw, MODEL_NAME)

print(forget_2.columns)

embedding_model = SentenceTransformer(cfg0.retriever_model, device=device)

print('loading base model')
base_model = AutoModelForCausalLM.from_pretrained(
    cfg0.model_id, token=hf_token, device_map="auto", torch_dtype=torch.bfloat16)

# bootstrap PeftModel with the first (loss_type, selection) adapter
first_cfg = Config_eval(selection=SELECTIONS[0], loss_type=LOSS_TYPES[0],
                         model_name=MODEL_NAME, dataset=DATASET)
model = PeftModel.from_pretrained(
    base_model, first_cfg.adaptor_path,
    adapter_name=f'adapter_{LOSS_TYPES[0]}_{SELECTIONS[0]}',
    device_map="auto", torch_dtype=torch.bfloat16)

for loss_type in LOSS_TYPES:
    for selection in SELECTIONS:
        all_results = run_one(
            loss_type, selection, model, tokenizer, embedding_model,
            forget_2, test_data, audit_data, MODEL_NAME, DATASET, all_results)