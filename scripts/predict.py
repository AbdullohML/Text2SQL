import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from text2sql.inference import generate_sql


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--columns", type=str, required=True)
    parser.add_argument("--max_target_length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    input_text = f"translate to SQL: question: {args.question} columns: {args.columns}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.base_model:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    if torch.cuda.is_available():
        model.to("cuda")

    prediction = generate_sql(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        max_target_length=args.max_target_length,
    )

    print(prediction)


if __name__ == "__main__":
    main()
