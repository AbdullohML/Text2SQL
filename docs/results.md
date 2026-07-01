# Results

## Evaluation setup

Dataset: WikiSQL  
Task: Text-to-SQL generation  
Main metric: **normalized exact match** on 1,000 sampled WikiSQL test examples.

## Results table

| Experiment | Model | Setup | Train time on T4 | Normalized EM |
|---|---|---|---:|---:|
| T5-small full fine-tune | `google-t5/t5-small` | encoder-decoder, full fine-tuning | ~17 min | 0.594 |
| T5-base + LoRA | `google-t5/t5-base` | encoder-decoder, LoRA on `q`, `v` | ~2h35m | 0.605 |
| GPT-2 full fine-tune | `gpt2` | decoder-only, full fine-tuning with prompt masking | ~30 min | 0.679 |

## GPT-2 decoder-only full fine-tuning

Configuration:

```text
model: gpt2
train examples: 56,355
test examples: 1,000
batch size: 4
gradient accumulation steps: 8
effective batch size: 32
epochs: 2
learning rate: 5e-5
max length: 384
GPU: Tesla T4
```

Training format:

```text
Task: translate question to SQL.
Question: <question>
Columns: <col1> | <col2> | ...
SQL: <target SQL>
```

Important detail: prompt tokens are masked with `-100`, so the loss is computed only on the SQL answer part.

Metrics:

```text
normalized exact match: 0.679
valid SQL-like rate: 0.689
select match: 0.918
where match: 0.808
aggregation match: 1.000
```

## Conclusion

The GPT-2 decoder-only model achieved the best normalized exact match in these experiments.

Current ranking:

```text
1. GPT-2 full fine-tune:    0.679
2. T5-base + LoRA:         0.605
3. T5-small full fine-tune: 0.594
```
