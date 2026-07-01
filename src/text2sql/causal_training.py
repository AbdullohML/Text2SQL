import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from text2sql.data import clean_text, get_columns, serialize_sql
from text2sql.metrics import compute_metrics, make_results_table
from text2sql.utils import set_seed


def make_prompt(example) -> str:
    question = clean_text(example["question"])
    columns = " | ".join(clean_text(col) for col in get_columns(example))
    return (
        "Task: translate question to SQL.\n"
        f"Question: {question}\n"
        f"Columns: {columns}\n"
        "SQL:"
    )


def extract_sql(generated_text: str) -> str:
    if "SQL:" in generated_text:
        sql = generated_text.split("SQL:", 1)[1]
    else:
        sql = generated_text

    sql = sql.strip()

    for marker in ["\nTask:", "\nQuestion:", "\nColumns:"]:
        if marker in sql:
            sql = sql.split(marker, 1)[0].strip()

    return sql


class CausalText2SQLCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)

        input_ids = []
        attention_mask = []
        labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_raw_splits(config: dict):
    dataset = load_dataset(config["dataset_name"], trust_remote_code=True)
    seed = config["seed"]

    if config["use_full_train"]:
        train_raw = dataset["train"].shuffle(seed=seed)
    else:
        train_raw = dataset["train"].shuffle(seed=seed).select(range(config["train_size"]))

    valid_raw = dataset["validation"].shuffle(seed=seed).select(range(config["valid_size"]))
    test_raw = dataset["test"].shuffle(seed=seed).select(range(config["test_size"]))

    return train_raw, valid_raw, test_raw


def tokenize_causal_splits(train_raw, valid_raw, test_raw, tokenizer, config: dict):
    def tokenize_example(example):
        prompt = make_prompt(example)
        target = " " + serialize_sql(example) + tokenizer.eos_token
        full_text = prompt + target

        full = tokenizer(full_text, max_length=config["max_length"], truncation=True)
        prompt_ids = tokenizer(prompt, max_length=config["max_length"], truncation=True)["input_ids"]

        labels = full["input_ids"].copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        full["labels"] = labels
        return full

    train_ds = train_raw.map(tokenize_example, remove_columns=train_raw.column_names)
    valid_ds = valid_raw.map(tokenize_example, remove_columns=valid_raw.column_names)
    test_ds = test_raw.map(tokenize_example, remove_columns=test_raw.column_names)

    return train_ds, valid_ds, test_ds


def generate_sql_causal(model, tokenizer, example, max_length: int, max_new_tokens: int = 96) -> str:
    prompt = make_prompt(example)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_sql(generated)


def train_causal_from_config(config: dict):
    set_seed(config["seed"])

    train_raw, valid_raw, test_raw = load_raw_splits(config)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    model.config.pad_token_id = tokenizer.pad_token_id

    if torch.cuda.is_available():
        model.to("cuda")

    train_ds, valid_ds, _ = tokenize_causal_splits(train_raw, valid_raw, test_raw, tokenizer, config)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=config["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=CausalText2SQLCollator(tokenizer),
    )

    trainer.train()

    best_dir = f'{config["output_dir"]}/best'
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    test_predictions = [
        generate_sql_causal(model, tokenizer, example, config["max_length"])
        for example in test_raw
    ]
    test_targets = [serialize_sql(example) for example in test_raw]
    test_prompts = [make_prompt(example) for example in test_raw]

    metrics = compute_metrics(test_predictions, test_targets)
    results = make_results_table(test_prompts, test_targets, test_predictions)

    print("\nEvaluation metrics")
    print("------------------")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nFirst errors")
    print("------------")
    print(results[results["correct"] == False][["input", "target", "prediction"]].head(10))

    return metrics, results
