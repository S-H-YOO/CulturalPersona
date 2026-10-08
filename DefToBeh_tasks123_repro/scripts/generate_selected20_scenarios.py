#!/usr/bin/env python3
"""Regenerate matched scenarios for the selected 20-pair value set.

The published prompt template is used verbatim. The API key is read from
OPENAI_API_KEY; runs are resumable at the pair level.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_SOURCE = ROOT / "prompts" / "scenario_generation_selected20.txt"
REQUIRED_ITEM_KEYS = (
    "item_id", "third_person", "second_person", "shared_options",
    "researcher_metadata",
)


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", str(text).strip()) if part])


def extract_scenario_prompt(path: Path, n_items: int) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if n_items != 20:
        raise ValueError("The published selected-20 template requires --n-items 20")
    return prompt


def fill_prompt(template: str, row: dict) -> str:
    values = {
        "value_pair": row["value_pair"],
        "upper_value_definition": row["upper_value_human_definition"],
        "sub_value_1_definition": row["sub_value_1_human_definition"],
        "sub_value_2_definition": row["sub_value_2_human_definition"],
    }
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    leftovers = re.findall(r"\{[a-z_0-9]+\}", prompt)
    if leftovers:
        raise ValueError(f"Unfilled prompt placeholders: {leftovers}")
    return prompt


def normalize_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept either the canonical W2D pairs file or ValueSet_selected20.csv."""
    required = {
        "pair_id", "value_pair", "upper_value_human_definition",
        "sub_value_1_human_definition", "sub_value_2_human_definition",
    }
    if required.issubset(frame.columns):
        return frame

    selected20 = {
        "rank", "upper_value", "A_label", "B_label", "upper_definition",
        "A_definition", "B_definition",
    }
    if not selected20.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"Unsupported pairs CSV schema; missing canonical columns: {missing}")

    normalized = frame.copy()
    normalized["pair_id"] = normalized["rank"].map(lambda x: f"VP{int(x):02d}")
    normalized["value_pair"] = (
        normalized["upper_value"].astype(str) + ": "
        + normalized["A_label"].astype(str) + " vs "
        + normalized["B_label"].astype(str)
    )
    normalized["upper_value_human_definition"] = normalized["upper_definition"]
    normalized["sub_value_1_human_definition"] = normalized["A_definition"]
    normalized["sub_value_2_human_definition"] = normalized["B_definition"]
    return normalized


def validate(data: dict, n_items: int) -> list[str]:
    errors = []
    if data.get("status") != "complete":
        return [f"status={data.get('status')!r}; issue={data.get('issue')!r}"]
    items = data.get("items")
    if not isinstance(items, list) or len(items) != n_items:
        return [f"items={len(items) if isinstance(items, list) else type(items).__name__}; "
                f"expected={n_items}"]
    ids = []
    for index, item in enumerate(items, 1):
        missing = [key for key in REQUIRED_ITEM_KEYS if key not in item]
        if missing:
            errors.append(f"item {index}: missing {missing}")
            continue
        ids.append(str(item.get("item_id")))
        for perspective in ("third_person", "second_person"):
            block = item.get(perspective)
            if not isinstance(block, dict) or not block.get("scenario") or not block.get("question"):
                errors.append(f"item {index}: invalid {perspective}")
            elif not 4 <= sentence_count(block["scenario"]) <= 6:
                errors.append(
                    f"item {index}: {perspective} scenario must contain 4-6 sentences"
                )
        options = item.get("shared_options")
        if isinstance(options, dict):
            option_ids = list(options)
        elif isinstance(options, list):
            option_ids = [x.get("option_id") for x in options]
        else:
            option_ids = []
        if len(option_ids) != 2:
            errors.append(f"item {index}: shared_options must contain exactly two options")
        elif option_ids != ["v1", "v2"]:
            errors.append(f"item {index}: option ids/order are not v1,v2")
    expected = [f"{i:02d}" for i in range(1, n_items + 1)]
    if ids != expected:
        errors.append("item_id sequence is not 01..%02d" % n_items)
    return errors


def usage_dict(usage) -> dict:
    if usage is None:
        return {}
    return usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)


def run_pair(row: dict, template: str, out: Path, args) -> dict:
    pair_id = str(row["pair_id"])
    pair_dir = out / pair_id
    done = pair_dir / "items.json"
    if done.exists():
        data = json.loads(done.read_text())
        errors = validate(data, args.n_items)
        if not errors:
            return {"pair_id": pair_id, "value_pair": row["value_pair"],
                    "status": "skipped", "items": len(data["items"]), "errors": []}
    pair_dir.mkdir(parents=True, exist_ok=True)
    prompt = fill_prompt(template, row)
    (pair_dir / "request.json").write_text(json.dumps({
        "pair_id": pair_id, "model": args.model, "n_items": args.n_items,
        "value_pair": row["value_pair"], "prompt": prompt,
    }, indent=2, ensure_ascii=False))
    # Promote a previously generated valid response when resuming an interrupted run.
    for previous in sorted(pair_dir.glob("response_attempt_*.json"), reverse=True):
        prior = json.loads(previous.read_text()).get("response", {})
        if not validate(prior, args.n_items):
            done.write_text(json.dumps(prior, indent=2, ensure_ascii=False))
            return {"pair_id": pair_id, "value_pair": row["value_pair"],
                    "status": "recovered", "items": len(prior["items"]), "errors": []}
    client = OpenAI(max_retries=8, timeout=args.timeout)
    last_errors = []
    for attempt in range(1, args.retries + 1):
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_completion_tokens=args.max_completion_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            raw = response.choices[0].message.content or ""
            (pair_dir / f"raw_attempt_{attempt}.txt").write_text(raw)
            data = json.loads(raw)
            last_errors = validate(data, args.n_items)
            (pair_dir / f"response_attempt_{attempt}.json").write_text(json.dumps({
                "model": response.model, "seconds": round(time.time() - t0, 2),
                "usage": usage_dict(response.usage), "validation_errors": last_errors,
                "response": data,
            }, indent=2, ensure_ascii=False))
            if not last_errors:
                (pair_dir / "items.json").write_text(
                    json.dumps(data, indent=2, ensure_ascii=False))
                return {"pair_id": pair_id, "value_pair": row["value_pair"],
                        "status": "complete", "items": len(data["items"]),
                        "attempt": attempt, "model_version": response.model,
                        "usage": usage_dict(response.usage), "errors": []}
        except Exception as exc:  # noqa: BLE001
            last_errors = [f"{type(exc).__name__}: {exc}"]
            (pair_dir / f"error_attempt_{attempt}.txt").write_text(last_errors[0])
        print(f"[{pair_id}] attempt {attempt} failed: {last_errors[:2]}", flush=True)
    return {"pair_id": pair_id, "value_pair": row["value_pair"],
            "status": "failed", "items": 0, "errors": last_errors}


def flatten(out: Path) -> int:
    rows = []
    for path in sorted(out.glob("VP*/items.json")):
        pair_id = path.parent.name
        data = json.loads(path.read_text())
        request_path = path.parent / "request.json"
        request = json.loads(request_path.read_text()) if request_path.exists() else {}
        for item in data.get("items", []):
            raw_options = item["shared_options"]
            options = (raw_options if isinstance(raw_options, dict) else
                       {x["option_id"]: x["text"] for x in raw_options})
            rows.append({
                "pair_id": pair_id,
                "value_pair": data.get("value_pair") or request.get("value_pair"),
                "item_id": item["item_id"],
                "scenario_3p": item["third_person"]["scenario"],
                "question_3p": item["third_person"]["question"],
                "scenario_2p": item["second_person"]["scenario"],
                "question_2p": item["second_person"]["question"],
                "option_v1": options.get("v1"), "option_v2": options.get("v2"),
                **{f"metadata_{k}": v for k, v in item["researcher_metadata"].items()},
            })
    if rows:
        pd.DataFrame(rows).to_csv(out / "items_flat.csv", index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", default=str(ROOT / "data" / "value_pairs_20_selected.csv"))
    parser.add_argument("--prompt-source", default=str(DEFAULT_PROMPT_SOURCE))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--n-items", type=int, default=20)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-completion-tokens", type=int, default=30000)
    parser.add_argument("--reasoning-effort", default="low")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = normalize_pairs(pd.read_csv(args.pairs_csv, encoding="utf-8-sig"))
    template = extract_scenario_prompt(Path(args.prompt_source), args.n_items)
    (out / "prompt_template_used.txt").write_text(template)
    (out / "run_config.json").write_text(json.dumps({
        **vars(args), "started": datetime.now().isoformat(timespec="seconds"),
        "n_pairs": len(frame), "requested_items": len(frame) * args.n_items,
    }, indent=2, ensure_ascii=False))
    print(f"{len(frame)} pairs x {args.n_items} items -> {out}", flush=True)

    summaries = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_pair, row.to_dict(), template, out, args)
                   for _, row in frame.iterrows()]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                result = {"pair_id": "unknown", "status": "failed", "items": 0,
                          "errors": [repr(exc)]}
            summaries.append(result)
            print(f"[{result['pair_id']}] {result['status']} items={result['items']}", flush=True)
            (out / "summary.json").write_text(
                json.dumps(sorted(summaries, key=lambda x: x["pair_id"]),
                           indent=2, ensure_ascii=False))
    total = flatten(out)
    failed = [x for x in summaries if x["status"] == "failed"]
    print(f"[DONE] pairs={len(summaries)-len(failed)}/{len(frame)} items={total} "
          f"failed={len(failed)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
