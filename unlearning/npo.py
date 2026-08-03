# 1. export CUDA_VISIBLE_DEVICES=3,5
# 2. accelerate launch --multi_gpu --num_processes 2 preference.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from npo_utils import *
from data_module import *
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from accelerate import  Accelerator
import torch
from peft import PeftModel
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE, qwen_chat_template, olmo_chat_template
from tabulate import tabulate
from collators import custom_gd_collator_forget, custom_data_collator

from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

class Config_npo:
    def __init__(self):
        super(Config_npo, self).__init__()
        self.loss_type      = 'npo' # change this with the experiment types provided above
        self.model_id       = 'check'#'Qwen/Qwen2.5-3B-Instruct'#'meta-llama/Llama-3.1-8B-Instruct' 
        self.lr             = 1e-04
        self.batch_size     = 1
        self.gradient_accumulation_steps = 8 #always batch size of 8
        self.num_epochs     = 1
        self.overwrite_dir  = True
        self.weight_decay   = 0.01 # 0.01 for unlearning normally but rep shift we need only 0.0
        self.max_length     = 512
        self.selection       ='coreset'
        self.dataset       = 'bio'
        self.model_name     = 'llama'
        self.adaptor_path    = f'./outputs/{self.model_name}_{self.dataset}' # this will be set based on the dataset in the code below
        self.save_dir       = f'./outputs/unlearning/{self.model_name}_{self.loss_type}_{self.dataset}_{self.selection}_model' 
        self.forget_path    = f'./data/{self.dataset}_forget.parquet'
        self.retain_path    = './data/bio_retain.parquet'
        self.max_steps      = 100
        self.save_steps     = 20
        self.npo_beta       = 0.1
        self.npo_retain_alpha = 1.0
        self.npo_forget_gamma = 1.0


cfg = Config_npo()

accelerator = Accelerator()

if cfg.model_name == 'qwen':
    cfg.model_id = 'Qwen/Qwen2.5-3B-Instruct'
if cfg.model_name == 'llama':
    cfg.model_id = 'meta-llama/Llama-3.1-8B-Instruct'
if cfg.model_name == 'olmo':
    cfg.model_id = 'allenai/OLMo-2-0425-1B-Instruct'

if cfg.dataset == 'bio':
    cfg.forget_path = './data/bio_forget.parquet'


if cfg.selection == 'coreset':
    cfg.forget_path = f'/home/praveen/eacl/data/random_{cfg.dataset}_forget.parquet'

metrics = [
    ("Model", f'{cfg.model_id}'),
     ("Selection type", f'{cfg.selection}'),
     ("Dataset", f'{cfg.dataset}'),
    ("Unlearning Loss", f'{cfg.loss_type}'),
    ("Forget Path",   f'{cfg.forget_path}'),
    ("Retain Path",   f'{cfg.retain_path}'),
    ("Number of Steps",   f'{cfg.max_steps}'),
]

print("\n\n============ Check List ============\n")
print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="github"))

def read_file(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.json'):
        df = pd.read_json(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.jsonl'):
        df = pd.read_json(path, lines=True)    
    return df


print('loading the forget set')
forget = read_file(cfg.forget_path)
print(f'\nThere are {len(forget)} samples in the forget set')

if cfg.loss_type != 'npo':
    print('loading the retain set')
    retain = read_file(cfg.retain_path)


# --- Load tokenizer ---
print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = hf_token)
tokenizer.pad_token = tokenizer.eos_token

# --- Load policy model ---
# we load it on cpu, let accelerate move it to GPU with accelerate.prepare_model
base_model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token
    )
print("Base model loaded.")

# --- Apply LoRA on policy model ---

print(f"\nLoading the LoRA adapter from {cfg.adaptor_path}")
policy_model = PeftModel.from_pretrained(
    base_model,
    cfg.adaptor_path,
    torch_dtype=torch.bfloat16,
    is_trainable = True
)

policy_model.config.use_cache = False
policy_model.print_trainable_parameters()

# --- Load reference model ---
ref_model_base = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token
)
ref_model = PeftModel.from_pretrained(
    ref_model_base,
    cfg.adaptor_path,
    torch_dtype=torch.bfloat16,
)


# ------- creating template format for tokenization --------
def make_template_format(df, model_name):
     if model_name == 'llama':
        df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
     elif model_name == 'qwen':
        df['question'] = df['question'].apply(lambda x : qwen_chat_template.format(question = x))
     elif model_name == 'olmo':
        df['question'] = df['question'].apply(lambda x : olmo_chat_template.format(question = x))
     return df

def normalize_columns(df):
    df = df.rename(columns={'prompt': 'question', 'generation': 'answer'})
    return df

forget = make_template_format(forget, cfg.model_name)
print('forget question and answer\n', forget['question'][0], forget['answer'][0])

if cfg.loss_type != 'npo':
    retain = make_template_format(retain, cfg.model_name)
    print('\n\nretain question and answer\n', retain['question'][0], retain['answer'][0])



# ---- Training args ----
training_args = TrainingArguments(
    output_dir = cfg.save_dir,
    learning_rate = cfg.lr,
    per_device_train_batch_size= cfg.batch_size, 
    num_train_epochs = cfg.num_epochs,
    max_steps = cfg.max_steps,
    save_strategy= 'steps',
    save_steps= cfg.save_steps,
    weight_decay = cfg.weight_decay,
    logging_dir = f'{cfg.save_dir}/logs',
    eval_strategy= 'no',
    label_names = ['labels'],
    bf16 = True,
    gradient_accumulation_steps= cfg.gradient_accumulation_steps,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
)


# ------- dataset and trainer setup, branched by loss_type -----

if cfg.loss_type == 'npo':
    print('\nNow running NPO without Retain')
    dataset = SingleDataset(
        forget_data = forget,
        tokenizer = tokenizer,
        max_length = cfg.max_length,
    )
    print('\nlength of the dataset', len(dataset))

    trainer = NPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        beta=0.1,
        data_collator = custom_data_collator,
        gamma = 1.0,
        alpha = 1.0,
)
else:
    print('\nNow running NPO with Retain')
    dataset = DualDatasetRandom(forget_data = forget,
                                            retain_data = retain,
                                            tokenizer = tokenizer,
                                            max_length = 512,
                                            question_key='question',
                                            answer_key='answer',
                                                    )
    print('\n\nlength of the dataset', len(dataset))
    trainer = RetainNPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            beta=0.1,
            data_collator = custom_gd_collator_forget,
            gamma = 1.0,
            alpha = 1.0,
    )


trainer.train()

accelerator.wait_for_everyone()

policy_model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')