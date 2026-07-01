import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from text2sql.training import train_from_config
from text2sql.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    train_from_config(config)


if __name__ == "__main__":
    main()
