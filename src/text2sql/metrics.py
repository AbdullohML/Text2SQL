import re
import unicodedata

import numpy as np
import pandas as pd


def normalize_sql(sql) -> str:
    sql = unicodedata.normalize("NFKC", str(sql))
    sql = sql.lower()
    sql = sql.replace('"', "'")
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*([(),=<>])\s*", r"\1", sql)
    return sql.strip()


def parse_sql_parts(sql) -> dict:
    sql = normalize_sql(sql)
    match = re.match(r"^select (.*?) from table(?: where (.*))?$", sql)

    if not match:
        return {
            "valid": False,
            "select": "",
            "where": "",
            "agg": "",
        }

    select_part = match.group(1).strip()
    where_part = match.group(2).strip() if match.group(2) else ""

    agg_match = re.match(r"^(max|min|count|sum|avg)\((.*)\)$", select_part)
    agg = agg_match.group(1) if agg_match else ""

    return {
        "valid": True,
        "select": select_part,
        "where": where_part,
        "agg": agg,
    }


def compute_metrics(predictions: list[str], targets: list[str]) -> dict:
    pred_parts = [parse_sql_parts(pred) for pred in predictions]
    target_parts = [parse_sql_parts(target) for target in targets]

    exact = [
        normalize_sql(pred) == normalize_sql(target)
        for pred, target in zip(predictions, targets)
    ]

    valid = [part["valid"] for part in pred_parts]

    select_match = [
        pred["select"] == target["select"]
        for pred, target in zip(pred_parts, target_parts)
    ]

    where_match = [
        pred["where"] == target["where"]
        for pred, target in zip(pred_parts, target_parts)
    ]

    aggregation_match = [
        pred["agg"] == target["agg"]
        for pred, target in zip(pred_parts, target_parts)
    ]

    return {
        "normalized_exact_match": float(np.mean(exact)),
        "valid_sql_like": float(np.mean(valid)),
        "select_match": float(np.mean(select_match)),
        "where_match": float(np.mean(where_match)),
        "aggregation_match": float(np.mean(aggregation_match)),
    }


def make_results_table(inputs: list[str], targets: list[str], predictions: list[str]) -> pd.DataFrame:
    results = pd.DataFrame({
        "input": inputs,
        "target": targets,
        "prediction": predictions,
    })

    results["target_norm"] = results["target"].apply(normalize_sql)
    results["prediction_norm"] = results["prediction"].apply(normalize_sql)
    results["correct"] = results["target_norm"] == results["prediction_norm"]

    return results
