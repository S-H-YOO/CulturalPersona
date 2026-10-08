#!/usr/bin/env python3
"""Summarize semantic AB/BA consistency and logit-text agreement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(line) for line in path.open())


def summarize(path: Path, task: str) -> list[dict]:
    frame = read_jsonl(path)
    frame["condition"] = frame["version"] if task == "task2" else frame["definition_condition"]
    frame["unit"] = (
        frame["pair_id"].astype(str)
        if task == "task2"
        else frame["pair_id"].astype(str) + "|" + frame["item_id"].astype(str) + "|" + frame["person"]
    )
    frame["logit_sem"] = frame["p_reading_a"].map(
        lambda value: "V1" if value > 0.5 else ("V2" if value < 0.5 else "Tie")
    )
    frame["text_sem"] = frame["text_reading"].map({"a": "V1", "b": "V2"})
    rows = []
    for condition, group in frame.groupby("condition"):
        for readout, column in (("logit", "logit_sem"), ("text", "text_sem")):
            wide = group.pivot_table(index="unit", columns="order", values=column, aggfunc="first")
            wide = wide[wide.reindex(columns=["AB", "BA"]).notna().all(axis=1)]
            same = wide["AB"].eq(wide["BA"])
            rows.append({
                "task": task,
                "condition": condition,
                "readout": readout,
                "paired_n": len(wide),
                "v1_maintain": (same & wide["AB"].eq("V1")).mean(),
                "v2_maintain": (same & wide["AB"].eq("V2")).mean(),
                "tie_maintain": (same & wide["AB"].eq("Tie")).mean(),
                "order_consistency": same.mean(),
                "order_mismatch": (~same).mean(),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_output", type=Path, help="Directory containing task2.jsonl and task3.jsonl")
    parser.add_argument("--output", type=Path, default=Path("order_consistency.csv"))
    args = parser.parse_args()
    rows = summarize(args.model_output / "task2.jsonl", "task2")
    rows += summarize(args.model_output / "task3.jsonl", "task3")
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()

