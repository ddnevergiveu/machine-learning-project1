"""Repository entry point; importing this module never trains a model."""

import sys

from customer_ml.cli import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    # Retain the convenient `python main.py --demo` spelling.
    if arguments and arguments[0] not in {"train", "predict", "demo-data", "-h", "--help"}:
        arguments.insert(0, "train")
    raise SystemExit(main(arguments))
