import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from text2sql.causal_training import train_causal_from_config
from text2sql.data import load_wikisql_splits, tokenize_splits
from text2sql.inference import generate_many
from text2sql.metrics import compute_metrics, make_results_table
from text2sql.model import load_tokenizer_and_model
from text2sql.utils import set_seed


def train_seq2seq_from_config(config: dict):
    set_seed(config["seed"])

    train_text, valid_text, test_text = load_wikisql_splits(config)
    tokenizer, model = load_tokenizer_and_model(config)

    train_ds, valid_ds, _ = tokenize_splits(train_text, valid_text, test_text, tokenizer, config)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        predict_with_generate=True,
        generation_max_length=config["max_target_length"],
        fp16=torch.cuda.is_available() and not config.get("use_8bit", False),
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=config["seed"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
    )

    trainer.train()

    best_dir = f'{config["output_dir"]}/best'
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    test_inputs = test_text["input_text"]
    test_targets = test_text["target_text"]
    test_predictions = generate_many(
        model=model,
        tokenizer=tokenizer,
        input_texts=test_inputs,
        max_input_length=config["max_input_length"],
        max_target_length=config["max_target_length"],
        batch_size=16,
    )

    metrics = compute_metrics(test_predictions, test_targets)
    results = make_results_table(test_inputs, test_targets, test_predictions)

    print("\nEvaluation metrics")
    print("------------------")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nFirst errors")
    print("------------")
    print(results[results["correct"] == False][["input", "target", "prediction"]].head(10))

    return metrics, results


def train_from_config(config: dict):
    if config.get("model_type", "seq2seq") == "causal":
        return train_causal_from_config(config)

    return train_seq2seq_from_config(config)
