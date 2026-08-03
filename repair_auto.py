import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import pandas as pd
from config import Config_repair_auto
from peft import PeftModel
from accelerate import Accelerator
from unlearning.template import LLAMA3_CHAT_TEMPLATE, olmo_chat_template
from unlearning.data_module import SingleDataset
from unlearning.collators import custom_data_collator
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

qwen_chat_template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""



def read_file(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.json'):
        return pd.read_json(path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.jsonl'):
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file type: {path}")


def run_repair(model_name, loss_type):
    cfg = Config_repair_auto(model_name=model_name, loss_type=loss_type)
    
    metrics = [
        ("Model", f'{cfg.model_name}'),
        ("Selection type", f'{cfg.selection}'),
        ("Dataset", f'{cfg.dataset}'),
        ("Unlearning Loss", f'{cfg.loss_type}'),
        ('Model id', f'{cfg.model_id}'),
        ("Adaptor Path", f'{cfg.adaptor_path}'),
        ("Data Path", f'{cfg.data_path}'),
        ("Number of Steps", f'{cfg.max_steps}'),
    ]
    print(f"\n\n============ Check List: {model_name} | {loss_type} ============\n")
    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="github"))

    data = read_file(cfg.data_path)
    print(data.shape)

    if cfg.model_name == 'llama':
        data['question'] = data['question'].apply(lambda x: LLAMA3_CHAT_TEMPLATE.format(question=x))
    elif cfg.model_name == 'qwen':
        data['question'] = data['question'].apply(lambda x: qwen_chat_template.format(question=x))
    elif cfg.model_name == 'olmo':
        data['question'] = data['question'].apply(lambda x: olmo_chat_template.format(question=x))

    print('\n\n', data['question'][0])

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map='auto',
        dtype=torch.bfloat16,
        token=hf_token,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False

    model = PeftModel.from_pretrained(
        model,
        cfg.adaptor_path,
        is_trainable=True,
    )
    model.print_trainable_parameters()

    dataset = SingleDataset(data, tokenizer, max_length=cfg.max_length)

    args = TrainingArguments(
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        bf16=True,
        max_steps=cfg.max_steps,
        logging_dir=f'{cfg.save_dir}/logs',
        eval_strategy='no',
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        save_strategy='steps',
        save_steps=cfg.save_steps,
        save_total_limit=10,
        output_dir=cfg.save_dir,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=custom_data_collator,
    )

    trainer.train()

    model = model.cpu()
    print(f"Model and tokenizer saved to {cfg.save_dir}")
    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

    # --- cleanup so the next run starts on a clean GPU ---
    del trainer
    del model
    del tokenizer
    del dataset
    del data
    del accelerator
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == '__main__':
    model_names = ['llama']
    loss_types = ['ga', 'npo', 'snpo']
    for model_name in model_names:
        for lt in loss_types:
            run_repair(model_name, lt)