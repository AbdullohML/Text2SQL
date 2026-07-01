import re
from datasets import load_dataset


AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


def clean_text(x) -> str:
    x = str(x).replace("\n", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def get_columns(example) -> list[str]:
    table = example["table"]

    if "header" in table:
        return table["header"]
    if "headers" in table:
        return table["headers"]
    if "column_names" in table:
        return table["column_names"]

    raise ValueError(f"Unknown table format: {table.keys()}")


def serialize_input(example) -> str:
    question = clean_text(example["question"])
    columns = " | ".join(clean_text(c) for c in get_columns(example))
    return f"translate to SQL: question: {question} columns: {columns}"


def iter_conditions(conditions):
    if isinstance(conditions, dict):
        col_indices = conditions.get("column_index", [])
        op_indices = conditions.get("operator_index", [])
        values = conditions.get("condition", [])

        for col_idx, op_idx, value in zip(col_indices, op_indices, values):
            yield col_idx, op_idx, value
    else:
        for cond in conditions:
            yield cond[0], cond[1], cond[2]


def serialize_sql(example) -> str:
    sql = example["sql"]
    columns = get_columns(example)

    select_col = clean_text(columns[sql["sel"]])
    agg = AGG_OPS[sql["agg"]]

    if agg:
        query = f"SELECT {agg}({select_col}) FROM table"
    else:
        query = f"SELECT {select_col} FROM table"

    where_parts = []

    for col_idx, op_idx, value in iter_conditions(sql.get("conds", [])):
        col_name = clean_text(columns[col_idx])
        op = COND_OPS[op_idx] if op_idx < len(COND_OPS) else "="
        value = clean_text(value)
        where_parts.append(f"{col_name} {op} '{value}'")

    if where_parts:
        query += " WHERE " + " AND ".join(where_parts)

    return query


def add_text_fields(example) -> dict:
    return {
        "input_text": serialize_input(example),
        "target_text": serialize_sql(example),
    }


def load_wikisql_splits(config: dict):
    dataset = load_dataset(config["dataset_name"], trust_remote_code=True)
    seed = config["seed"]

    if config["use_full_train"]:
        train_raw = dataset["train"].shuffle(seed=seed)
    else:
        train_raw = dataset["train"].shuffle(seed=seed).select(range(config["train_size"]))

    valid_raw = dataset["validation"].shuffle(seed=seed).select(range(config["valid_size"]))
    test_raw = dataset["test"].shuffle(seed=seed).select(range(config["test_size"]))

    train_text = train_raw.map(add_text_fields)
    valid_text = valid_raw.map(add_text_fields)
    test_text = test_raw.map(add_text_fields)

    return train_text, valid_text, test_text


def tokenize_splits(train_text, valid_text, test_text, tokenizer, config: dict):
    def preprocess(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=config["max_input_length"],
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=config["max_target_length"],
            truncation=True,
        )

        inputs["labels"] = labels["input_ids"]
        return inputs

    remove_cols = train_text.column_names

    train_ds = train_text.map(preprocess, batched=True, remove_columns=remove_cols)
    valid_ds = valid_text.map(preprocess, batched=True, remove_columns=remove_cols)
    test_ds = test_text.map(preprocess, batched=True, remove_columns=remove_cols)

    return train_ds, valid_ds, test_ds
