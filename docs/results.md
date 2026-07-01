# Results

## Evaluation setup

Dataset: WikiSQL  
Task: Text-to-SQL generation  
Input format:

```text
translate to SQL: question: <question> columns: <col1> | <col2> | ...
```

Target format:

```sql
SELECT <column> FROM table WHERE <column> = '<value>'
```

Main metric: **normalized exact match** on 1,000 sampled WikiSQL test examples.

## Results table

| Experiment | Model | Setup | Trainable parameters | Train time on T4 | Normalized EM |
|---|---|---|---:|---:|---:|
| T5-small full fine-tune | `google-t5/t5-small` | full fine-tuning | all model params | ~17 min | 0.594 |
| T5-base + LoRA | `google-t5/t5-base` | LoRA on `q`, `v` | 884,736 / 223,788,288 = 0.395% | ~2h35m | 0.605 |

## T5-small full fine-tuning

Configuration:

```text
model: google-t5/t5-small
train examples: 56,355
test examples: 1,000
batch size: 8
gradient accumulation steps: 4
effective batch size: 32
epochs: 2
learning rate: 3e-4
GPU: Tesla T4
```

Result:

```text
normalized exact match: 0.594
```

## T5-base + LoRA

Configuration:

```text
model: google-t5/t5-base
train examples: 56,355
test examples: 1,000
batch size: 2
gradient accumulation steps: 8
effective batch size: 16
epochs: 2
learning rate: 1e-4
LoRA rank: 8
LoRA alpha: 16
LoRA target modules: q, v
GPU: Tesla T4
```

Output:

```text
trainable params: 884,736
all params: 223,788,288
trainable%: 0.3953
```

Metrics:

```text
normalized exact match: 0.605
valid SQL-like rate: 0.714
select match: 0.912
where match: 0.756
aggregation match: 1.000
```

## Conclusion

T5-base + LoRA slightly improved normalized exact match from `0.594` to `0.605`, but required much longer training time. This suggests that for this setup, the T5-small full fine-tuning baseline is more efficient, while T5-base + LoRA is useful as a parameter-efficient experiment.
