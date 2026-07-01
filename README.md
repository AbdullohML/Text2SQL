# Text2SQL

Text-to-SQL experiments on WikiSQL using encoder-decoder and decoder-only Transformer models.

The project converts a natural language question and table schema into a SQL query.

```text
question + table columns -> SQL query
```

## Experiments

| Experiment | Model | Setup | Train data | Eval data | Main metric | Result |
|---|---|---|---:|---:|---|---:|
| Baseline | `google-t5/t5-small` | full fine-tuning | 56,355 WikiSQL train examples | 1,000 sampled test examples | normalized exact match | 0.594 |
| Experiment 2 | `google-t5/t5-base` | LoRA | 56,355 WikiSQL train examples | 1,000 sampled test examples | normalized exact match | 0.605 |
| Experiment 3 | `gpt2` | decoder-only full fine-tuning | 56,355 WikiSQL train examples | 1,000 sampled test examples | normalized exact match | 0.679 |

The trained models are not stored in this repository. The repository contains code, configs, notebooks, and experiment notes so the results can be reproduced.

## Reproduce experiments

T5-small full fine-tuning:

```bash
python scripts/train.py --config configs/t5_small_full.yaml
```

T5-base + LoRA:

```bash
python scripts/train.py --config configs/t5_base_lora.yaml
```

GPT-2 decoder-only full fine-tuning:

```bash
python scripts/train.py --config configs/gpt2_full.yaml
```

## Key conclusion

The best result in these experiments came from the decoder-only GPT-2 setup:

```text
GPT-2 full fine-tune: 0.679 normalized exact match
```

This suggests that a decoder-only model can work well for Text-to-SQL when the task is formatted as prompt completion and the prompt tokens are masked during training.
