# 1. export CUDA_VISIBLE_DEVICES=4,5
# 2. accelerate launch --multi_gpu --num_processes 2 simnpo.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel
from data_module import DualDatasetRandom, SingleDataset
from collators import custom_gd_collator_forget, custom_data_collator
from snpo_utils import SNPOR, SNPO
from accelerate import Accelerator
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE, qwen_chat_template, olmo_chat_template
from tabulate import tabulate
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")


accelerator = Accelerator()


class Config_snpo:
    def __init__(self):
        super(Config_snpo, self).__init__()
        self.loss_type      = 'snpo' 
        self.model_id       = 'check'#'Qwen/Qwen2.5-3B-Instruct'#'meta-llama/Llama-3.1-8B-Instruct' 
        self.lr             = 1e-04
        self.batch_size     = 2
        self.gradient_accumulation_steps = 4 #always batch size of 8
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
        self.delta          = 0.0
        self.beta           = 3.5

cfg = Config_snpo() 

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
     ("Selection type", f'{cfg.selection}'),
     ("Dataset", f'{cfg.dataset}'),
    ("Unlearning Loss", f'{cfg.loss_type}'),
    ("Forget Path",   f'{cfg.forget_path}'),
    ("Retain Path",   f'{cfg.retain_path}' if cfg.loss_type != 'ga' else 'N/A (ga)'),
    ("Number of Steps",   f'{cfg.max_steps}'),
    ("SimNPO Delta",   f'{cfg.delta}'),
    ("SimNPO Beta",   f'{cfg.beta}'),
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


print('loading the forget, retain')
forget = read_file(cfg.forget_path)
print(f'\nThere are {len(forget)} samples in the forget set')
if cfg.loss_type != 'snpo':
    print('loading the retain set')
    retain = read_file(cfg.retain_path)
    print(retain.shape)


# ------- Load the tokenizer ----------------
print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token



# ------- Load the model ----------------
print(f"\nLoading the Model {cfg.model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    device_map = 'auto',
    token = hf_token
)

print(f"\nLoading the LoRA adapter from {cfg.adaptor_path}")
model = PeftModel.from_pretrained(
    base_model,
    cfg.adaptor_path,
    is_trainable=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False
model.print_trainable_parameters()

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

if cfg.loss_type != 'snpo':
    retain = make_template_format(retain, cfg.model_name)
    print('\n\nretain question and answer\n', retain['question'][0], retain['answer'][0])


# ------- Training Arguments ---------
training_args = TrainingArguments(
    output_dir = cfg.save_dir,
    learning_rate = cfg.lr,
    per_device_train_batch_size= cfg.batch_size, 
    max_steps = cfg.max_steps,
    save_strategy= 'steps',
    save_steps= cfg.save_steps,
    weight_decay = cfg.weight_decay,
    logging_dir = f'{cfg.save_dir}/logs',
    eval_strategy= 'no',
    label_names = ['labels'],
    num_train_epochs = cfg.num_epochs,
    bf16 = True,
    gradient_accumulation_steps= cfg.gradient_accumulation_steps,
    ddp_find_unused_parameters=False,
)

if cfg.loss_type == 'snpo':
    dataset = SingleDataset(
        forget_data = forget,
        tokenizer = tokenizer,
        max_length = cfg.max_length,
    )
    print('\nlength of the dataset', len(dataset))

    trainer = SNPO(
        model = model,
        args = training_args,
        train_dataset = dataset,
        processing_class = tokenizer,
        data_collator = custom_data_collator,
    )
else:
    dataset = DualDatasetRandom(
        forget_data = forget,
        retain_data = retain,
        tokenizer = tokenizer,
        max_length = cfg.max_length,
    )
    print('\nlength of the dataset', len(dataset))

    trainer = SNPOR(
        model = model,
        args = training_args,
        train_dataset = dataset,
        processing_class = tokenizer,
        data_collator = custom_gd_collator_forget,
    )

trainer.train()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')