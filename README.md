# OARS: Objective-Aware Retain-set selection for LLM Unlearning

## Abstract

LLM unlearning benchmarks typically provide both forget and retain sets,
while practical removal requests often specify only the data to forget.
This creates the problem of selecting a compact retain set from a much
larger candidate retain pool. We introduce **O**bjective-**A**ware
**R**etain-Set **S**election (OARS), which measures each
candidate's relative hidden-state shift after a short forget-only scoring
warm-up. OARS clusters candidates in the pre-unlearning representation
space and selects the highest-shift examples within each cluster, combining
objective sensitivity with distributional coverage. Across three model
families, two datasets, and three unlearning objectives, OARS achieves the
strongest average post-repair utility and recovers 65.9\% of
impairment-induced utility loss. These results show that objective-aware
retain-set selection improves utility while maintaining high Forget
Quality.


## Setup

We use `uv` for environment management and dependency installation.

Create and activate a virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

Create a `.env` file containing your Hugging Face token:

```bash
HF_TOKEN=your_huggingface_token
```

The experiments support the following models and datasets:

* Models: `llama`, `olmo`, `qwen`
* Datasets: `muse`, `bio`
* Unlearning methods: `ga`, `npo`, `snpo`
* Repair selection methods: `emb`, `random`, `raslik`, `oars`

The standard experiment sequence is:

```text
Fine-tuning → Impair → Repair → Evaluation
```

---

## 1. Fine-tuning

Fine-tune a model on a dataset using:

```bash
python finetune.py --model_name llama --dataset bio
```

For example:

```bash
python finetune.py --model_name olmo --dataset muse
python finetune.py --model_name qwen --dataset bio
```

The fine-tuned LoRA adapter is saved under:

```text
outputs/{model_name}_{dataset}
```

---

## 2. Impair

The impairment stage supports GA, NPO, and SimNPO. Use `--stage impair`.

### GA

```bash
python unlearning/ga.py --model_name llama --dataset bio --stage impair
```

### NPO

```bash
python unlearning/npo.py --model_name llama --dataset bio --stage impair
```

### SimNPO

```bash
python unlearning/simnpo.py --model_name llama --dataset bio --stage impair
```

The same commands can be used for `muse`, `bio`, and any of the three supported models.

Impairment runs use 100 optimization steps and saves intermediate checkpoints.

---

## 3. Repair

Repair is performed using a selected retain set.

Available selection methods are:

```text
emb
random
raslik
oars
```

For example:

```bash
python repair.py --dataset bio --selection emb
```

or:

```bash
python repair.py --dataset muse --selection raslik
```

The repair script checks which model/unlearning combinations are available and runs only those experiments. Missing combinations are automatically skipped.

Repair outputs are saved under:

```text
outputs/repair/{model_name}_{loss_type}_{dataset}_{selection}
```

---

## 4. Evaluation

After repair, evaluate the available models using:

```bash
python eval/evaluate.py --model_name llama --dataset bio
```

For example:

```bash
python evaluate.py --model_name olmo --dataset muse
```

The evaluation script automatically evaluates the available:

```text
Pre-unlearning model
Impaired models
Repaired models
```

Missing experiments are skipped automatically.

The main reported metrics are:

* `FQ` — Forget Quality
* `Test` — Test Utility
* `AU` — Audit Utility
* Perplexity scores

Results are stored under:

```text
results/
```

---

# OARS

OARS requires representation extraction after fine-tuning and after warm-up, so it should be run after the warm-up models have been created.

For the WARM-UP, RUN:

```bash
python unlearning/ga.py --model_name llama --dataset bio --stage warmup
python unlearning/npo.py --model_name llama --dataset bio --stage warmup
python unlearning/snpo.py --model_name llama --dataset bio --stage warmup
```
This will automatically run for 1 epoch with the same hyperparameters.

For the OARS,
Run:

```bash
python oars/retain.py --model_name llama --dataset bio
```

For example:

```bash
python oars/retain.py --model_name olmo --dataset muse
```

The script:

```text
Pre-unlearning representations
        ↓
Post-impair representations
        ↓
Representation shift
        ↓
Clustered OARS selection
```

It automatically checks the available `ga`, `npo`, and `snpo` impairment checkpoints and skips missing ones.

The selected retain sets are saved as:

```text
data/retain_sets/{model_name}_{loss_type}_{dataset}_oars.parquet
```

Once the OARS retain sets have been generated, run repair with:

```bash
python repair.py --dataset bio --selection oars
```

and then evaluate normally:

```bash
python evaluate.py --model_name llama --dataset bio
```
