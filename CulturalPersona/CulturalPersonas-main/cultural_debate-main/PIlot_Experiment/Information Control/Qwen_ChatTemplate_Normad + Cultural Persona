import argparse
import datetime
import os

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

import Qwen_ChatTemplate as chat_template
import qwen_no_rot_cp_probe_after_interaction as abcd
import qwen_no_rot_oeg_mcs_after_interaction as oeg_mcs
import qwen_persona_interaction as exp
from qwen_persona_interaction_all_countries import DEFAULT_COUNTRIES


CP_ROOT = "/scratch/lami2026/personal/sanghoon_2026/CulturalPersonas-main/CulturalPersona/CulturalPersonas-main"


def messages_for_condition(condition_spec, user_prompt):
    if isinstance(condition_spec, dict) and condition_spec.get("kind") == "messages":
        return condition_spec["messages"] + [{"role": "user", "content": user_prompt}]
    return [
        {"role": "system", "content": condition_spec + "\n\nAlways answer in English."},
        {"role": "user", "content": user_prompt},
    ]


def apply_messages(tokenizer, messages):
    template_kwargs = {}
    if "qwen3" in getattr(tokenizer, "name_or_path", "").lower():
        template_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        **template_kwargs,
    )


def run_oeg_condition(tokenizer, model, embedder, condition, condition_spec, payload):
    rows = []
    for idx, prompt in enumerate(payload["oeg_prompts"], start=1):
        print(f"  [OEG {idx:3d}/{len(payload['oeg_prompts'])}] {condition}")
        messages = messages_for_condition(condition_spec, prompt)
        response = chat_template.call_chat(
            tokenizer,
            model,
            messages,
            max_new_tokens=50,
            temperature=0.0,
        )
        choices = payload["answer_choices"][idx - 1]
        emb_gen = embedder.encode(response, convert_to_tensor=True)
        emb_opts = embedder.encode(choices, convert_to_tensor=True)
        sims = util.cos_sim(emb_gen, emb_opts)[0]
        sim_probs = F.softmax(sims, dim=0)
        best_idx = torch.argmax(sim_probs).item()
        row = {
            "condition": condition,
            "prompt": prompt,
            "response": response,
            "matched_option": payload["options"][best_idx],
            "matched_text": choices[best_idx],
            "cosine_sim": sim_probs[best_idx].item(),
        }
        for opt, sim in zip(payload["options"], sim_probs):
            row[f"sim_{opt}"] = sim.item()
        rows.append(row)
    return pd.DataFrame(rows)


def run_mcs_condition(tokenizer, model, condition, condition_spec, payload):
    option_token_ids = [
        tokenizer(option, add_special_tokens=False).input_ids[0]
        for option in payload["options"]
    ]
    rows = []
    for idx, prompt in enumerate(payload["mcs_prompts"], start=1):
        print(f"  [MCS {idx:3d}/{len(payload['mcs_prompts'])}] {condition}")
        messages = messages_for_condition(condition_spec, prompt)
        input_ids = apply_messages(tokenizer, messages).to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
        logits = next_token_logits[0, option_token_ids]
        logprobs = torch.log_softmax(logits, dim=-1)
        row = {"condition": condition, "prompt": prompt}
        for opt, logprob in zip(payload["options"], logprobs):
            row[f"{opt}_prob"] = logprob.item()
        rows.append(row)
    return pd.DataFrame(rows)


def run_one_country(args, tokenizer, model, embedder, gt, payload, country_key, background_contexts):
    country_name, cp_country, condition_specs = chat_template.build_condition_specs(
        args,
        tokenizer,
        model,
        country_key,
        background_contexts,
    )
    country_code = abcd.country_code_for_gt(cp_country)
    output_dir = os.path.join(args.output_dir, country_key)
    os.makedirs(output_dir, exist_ok=True)

    metric_summaries = []
    for eval_mode in args.eval_modes:
        mode_dir = os.path.join(output_dir, eval_mode)
        os.makedirs(mode_dir, exist_ok=True)
        for condition, condition_spec in condition_specs.items():
            if eval_mode == "oeg":
                raw = run_oeg_condition(tokenizer, model, embedder, condition, condition_spec, payload)
                dist = oeg_mcs.get_samples_oeg(
                    raw,
                    payload["traits"],
                    args.test_type,
                    samples=len(gt[gt.country == country_code]),
                )
            else:
                raw = run_mcs_condition(tokenizer, model, condition, condition_spec, payload)
                dist = oeg_mcs.get_samples_mcs(
                    raw,
                    payload["reverse"],
                    payload["traits"],
                    args.test_type,
                    samples=len(gt[gt.country == country_code]),
                )

            raw.to_csv(os.path.join(mode_dir, f"{condition}_raw.csv"), index=False)
            dist.to_csv(os.path.join(mode_dir, f"{condition}_dist.csv"), index=False)
            metrics = oeg_mcs.generate_metrics_like_oeg_mcs(gt, dist, country_code)
            metrics.insert(0, "condition", condition)
            metrics.insert(0, "eval_mode", eval_mode)
            metrics.insert(0, "target_country", cp_country)
            metrics.insert(0, "country_key", country_key)
            metrics.to_csv(os.path.join(mode_dir, f"{condition}_metrics.csv"), index=False)
            metric_summaries.append(metrics)

    all_metrics = pd.concat(metric_summaries, ignore_index=True)
    all_metrics.to_csv(os.path.join(output_dir, "oeg_mcs_metrics_by_trait.csv"), index=False)
    summary = (
        all_metrics.groupby(["country_key", "target_country", "eval_mode", "condition"], as_index=False)
        .agg(
            mean_kl_divergence=("kl_divergence", "mean"),
            mean_ks_stat=("ks_stat", "mean"),
            mean_ks_pvalue=("ks_pvalue", "mean"),
            mean_gt=("gt_mean", "mean") if "gt_mean" in all_metrics else ("kl_divergence", "count"),
            mean_model=("model_mean", "mean") if "model_mean" in all_metrics else ("kl_divergence", "count"),
        )
    )
    summary.to_csv(os.path.join(output_dir, "oeg_mcs_metrics_summary.csv"), index=False)
    return all_metrics, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normad_input_path", default="data/normad.jsonl")
    parser.add_argument("--output_dir", default="outputs/cp_oeg_mcs_chat_template_rot_no_rot_json_dialogue_qwen25_14b")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", exp.DEFAULT_CACHE_DIR))
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument("--limit_per_country", type=int, default=None)
    parser.add_argument("--adapt_per_label", type=int, default=2)
    parser.add_argument("--test_per_label", type=int, default=None)
    parser.add_argument("--n_turns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reflection_tokens", type=int, default=700)
    parser.add_argument("--reflection_temp", type=float, default=0.3)
    parser.add_argument("--cp_root", default=CP_ROOT)
    parser.add_argument("--cp_norm_file", default=None)
    parser.add_argument("--cp_norm_limit", type=int, default=2)
    parser.add_argument("--background_context_path", default="data/country_etiquette_backgrounds_detailed.json")
    parser.add_argument("--background_max_chars", type=int, default=6000)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--ground_truth", default=None)
    parser.add_argument("--test_type", choices=["standard", "trait", "cp", "big5chat"], default="trait")
    parser.add_argument("--max_questions", type=int, default=20)
    parser.add_argument("--eval_modes", nargs="+", choices=["oeg", "mcs"], default=["oeg", "mcs"])
    args = parser.parse_args()

    if args.cp_norm_file is None:
        args.cp_norm_file = os.path.join(args.cp_root, "datasets/cp/norms/openai_cultural_norms.json")
    if args.ground_truth is None:
        args.ground_truth = os.path.join(args.cp_root, "datasets/ground-truth/big-five-ocean.csv")
    if args.questions is None:
        default_questions = {
            "standard": "datasets/baselines/ipip-120.csv",
            "trait": "datasets/baselines/trait.csv",
            "cp": "datasets/baselines/cp.csv",
            "big5chat": "datasets/baselines/big5chat.csv",
        }
        args.questions = os.path.join(args.cp_root, default_questions[args.test_type])

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    background_contexts = chat_template.json_background.load_wikipedia_contexts(args.background_context_path)
    payload = oeg_mcs.build_eval_payload(args.test_type, args.questions, args.max_questions)
    gt = pd.read_csv(args.ground_truth)

    print(f"Loading model once: {args.model_id}")
    tokenizer, model = exp.load_qwen(args.model_id, args.cache_dir, gpu_only=args.gpu_only)
    print("Loading sentence embedder: all-MiniLM-L6-v2")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    started_at = datetime.datetime.now()
    metrics = []
    summaries = []
    for country_key in args.countries:
        country_metrics, country_summary = run_one_country(
            args,
            tokenizer,
            model,
            embedder,
            gt,
            payload,
            country_key,
            background_contexts,
        )
        metrics.append(country_metrics)
        summaries.append(country_summary)

    pd.concat(metrics, ignore_index=True).to_csv(
        os.path.join(args.output_dir, "aggregate_oeg_mcs_metrics_by_trait.csv"),
        index=False,
    )
    aggregate_summary = pd.concat(summaries, ignore_index=True)
    aggregate_summary.to_csv(
        os.path.join(args.output_dir, "aggregate_oeg_mcs_metrics_summary.csv"),
        index=False,
    )
    print(f"Started:  {started_at}")
    print(f"Finished: {datetime.datetime.now()}")
    print(f"Elapsed:  {datetime.datetime.now() - started_at}")
    print(f"Outputs:  {args.output_dir}")


if __name__ == "__main__":
    main()
