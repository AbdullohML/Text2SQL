# Text2SQL

Text-to-SQL experiments on WikiSQL using T5 models.

The project converts a natural language question and table schema into a SQL query.

```text
question + table columns -> SQL query
```

## Experiments

| Experiment | Model | Fine-tuning type | Train data | Eval data | Main metric | Result |
|---|---|---:|---:|---:|---|---:|
| Baseline | `google-t5/t5-small` | full fine-tuning | 56,355 WikiSQL train examples | 1,000 sampled WikiSQL test examples | normalized exact match | 0.594 |
| Experiment 2 | `google-t5/t5-base` | LoRA | 56,355 WikiSQL train examples | 1,000 sampled WikiSQL test examples | normalized exact match | 0.605 |

The trained models are not stored in this repository. The repository contains code, configs, notebooks, and experiment notes so the results can be reproduced.

## Main metric

The main comparison metric is **normalized exact match**.

Before comparison, SQL strings are normalized by:

- lowercasing
- normalizing whitespace
- normalizing punctuation spacing around `()`, `,`, `=`, `<`, `>`
- normalizing quotes

Example:

```sql
SELECT Written by FROM table WHERE Original airdate = 'June 6, 1999'
```

and

```sql
select Written by from table where Original airdate='june 6, 1999'
```

are treated as the same after normalization.

## Setup

Recommended environment: Google Colab with Tesla T4 GPU.

```bash
pip install -r requirements.txt
```

For Colab, if `peft` fails because of an old `torchao`, run:

```bash
pip uninstall -y torchao
pip install -U transformers accelerate peft bitsandbytes
```

## Reproduce T5-small baseline

```bash
python scripts/train.py --config configs/t5_small_full.yaml
```

## Reproduce T5-base + LoRA

```bash
python scripts/train.py --config configs/t5_base_lora.yaml
```

## Run inference

After training, point `model_path` to your saved model or adapter path:

```bash
python scripts/predict.py \
  --model_path runs/t5-small-full/best \
  --question "How many players are from France?" \
  --columns "player | country | age | team"
```

## Project structure

```text
Text2SQL/
├── configs/
│   ├── t5_small_full.yaml
│   └── t5_base_lora.yaml
├── docs/
│   └── results.md
├── notebooks/
│   └── text2sql_t5_small_baseline.ipynb
├── scripts/
│   ├── train.py
│   └── predict.py
├── src/text2sql/
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   ├── training.py
│   └── inference.py
├── requirements.txt
└── README.md
```

## Key conclusion

T5-base + LoRA slightly improved normalized exact match from `0.594` to `0.605`, while training only about `0.4%` of the model parameters. However, it required much longer training time on Colab T4, so the T5-small full fine-tuning baseline remained the more efficient setup.
