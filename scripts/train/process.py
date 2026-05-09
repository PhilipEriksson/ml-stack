#!/usr/bin/env python3
"""Convert a raw HF dataset to Unsloth alpaca format (instruction, input, output).

Column mapping is auto-detected from common dataset patterns. Override with:
  --instruction <col>  --input <col>  --output <col>

Usage:
  python scripts/train/process.py /path/to/raw /path/to/processed [--instruction x --input y --output z]
"""

import sys
import os
import argparse

try:
    from datasets import load_dataset
except ImportError:
    print("❌ datasets not found. Run: pip install datasets")
    sys.exit(1)

# Common column name patterns for each alpaca field (lowercased for matching)
INSTRUCTION_HINTS = {"instruction", "prompt", "query", "question", "title"}
INPUT_HINTS = {"input", "context", "article", "passage", "content", "text", "story"}
OUTPUT_HINTS = {"output", "response", "answer", "completion", "highlights", "summary", "label"}


def detect_column(columns, hints, label):
    """Find the first column name that matches any hint in the hints set."""
    col_lower = {c.lower(): c for c in columns}
    for hint in hints:
        if hint in col_lower:
            return col_lower[hint]
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert raw dataset to alpaca format")
    parser.add_argument("raw_path", help="Path to raw HF dataset directory")
    parser.add_argument("processed_path", help="Path to save processed dataset")
    parser.add_argument("--instruction", default=None, help="Column to map as instruction")
    parser.add_argument("--input", default=None, help="Column to map as input")
    parser.add_argument("--output", default=None, help="Column to map as output")
    parser.add_argument("--test-split", type=float, default=0.05,
                        help="Fraction to hold out as test split (default: 0.05)")
    args = parser.parse_args()

    raw_path = args.raw_path
    processed_path = args.processed_path

    print(f"Loading dataset from {raw_path}...")
    ds = load_dataset(raw_path)

    # Pick the first split (usually "train")
    split_name = list(ds.keys())[0]
    data = ds[split_name]
    columns = list(data.column_names)
    print(f"  Found split '{split_name}' with {len(data)} examples")
    print(f"  Columns: {columns}")

    # Auto-detect column mapping
    inst_col = args.instruction or detect_column(columns, INSTRUCTION_HINTS, "instruction")
    input_col = args.input or detect_column(columns, INPUT_HINTS, "input")
    output_col = args.output or detect_column(columns, OUTPUT_HINTS, "output")

    if not inst_col or not output_col:
        print(f"❌ Could not auto-detect columns:")
        print(f"   instruction: {inst_col} (override with --instruction)")
        print(f"   input:       {input_col} (override with --input)")
        print(f"   output:      {output_col} (override with --output)")
        print(f"")
        print(f"   Available columns: {columns}")
        print(f"   Common patterns:")
        print(f"     instruction: {', '.join(INSTRUCTION_HINTS)}")
        print(f"     input:       {', '.join(INPUT_HINTS)}")
        print(f"     output:      {', '.join(OUTPUT_HINTS)}")
        sys.exit(1)

    print(f"  Mapping: instruction={inst_col}, input={input_col}, output={output_col}")

    def to_alpaca(examples):
        instructions = []
        inputs = []
        outputs = []
        for i in range(len(examples[inst_col])):
            inst = str(examples[inst_col][i]) if examples[inst_col][i] is not None else ""
            inp = str(examples[input_col][i]) if input_col and examples[input_col][i] is not None else ""
            out = str(examples[output_col][i]) if examples[output_col][i] is not None else ""
            instructions.append(inst)
            inputs.append(inp if inp else None)
            outputs.append(out)
        return {"instruction": instructions, "input": inputs, "output": outputs}

    alpaca = data.map(to_alpaca, batched=True)
    # Remove the original columns, keep only alpaca format
    alpaca = alpaca.remove_columns([c for c in columns if c not in ("instruction", "input", "output")])

    # Split into train and eval
    if args.test_split > 0 and len(alpaca) > 10:
        split = alpaca.train_test_split(test_size=args.test_split, seed=42)
        processed = {
            "train": split["train"],
            "eval": split["test"],
        }
    else:
        processed = {"train": alpaca}

    os.makedirs(processed_path, exist_ok=True)
    for split_name, split_data in processed.items():
        split_data.save_to_disk(os.path.join(processed_path, split_name))
        print(f"  Saved {split_name}: {len(split_data)} examples")

    print(f"\n✅ Processed dataset saved to {processed_path}")


if __name__ == "__main__":
    main()
