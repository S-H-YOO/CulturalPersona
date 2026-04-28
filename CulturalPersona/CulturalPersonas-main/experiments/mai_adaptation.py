import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from openai import OpenAI
from scipy.stats import entropy, gaussian_kde, ks_2samp
from sentence_transformers import SentenceTransformer, util


def normalize_country(name):
    aliases = {
        "usa": "United States",
        "us": "United States",
        "united states": "United States",
        "brazil": "Brazil",
        "korea": "Korea",
        "south korea": "Korea",
        "republic of korea": "Korea",
        "saudi": "Saudi Arabia",
        "saudi arabia": "Saudi Arabia",
        "saudiarabia": "Saudi Arabia",
        "ksa": "Saudi Arabia",
    }
    return aliases.get(name.strip().lower(), name.strip())


def load_norms(norm_file, country, limit=2):
    country = normalize_country(country)

    if norm_file.endswith(".json"):
        with open(norm_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        entries = payload.get("cultural-norms", payload if isinstance(payload, list) else [])
        for entry in entries:
            if normalize_country(entry.get("country", "")) == country:
                norms = [n["text"] for n in entry["norms"]]
                return norms[:limit]
        raise ValueError(f"No norms for {country} in {norm_file}")

    df = pd.read_csv(norm_file)
    rows = df[df.country.str.strip().str.lower() == country.lower()]
    if rows.empty:
        raise ValueError(f"No norms for {country} in {norm_file}")
    return rows.norm.values[:limit].tolist()


def build_system_prompt(country, norms, memory_bank=None, role_note=None):
    norm_text = "\n".join(f"- {norm}" for norm in norms)
    prompt = f"""You are someone who was born and brought up in {country}.
You are familiar with {country}'s culture and traditions.

Important cultural norms you often practice:
{norm_text}"""

    if role_note:
        prompt += f"\n\n{role_note}"

    if memory_bank:
        memories = "\n".join(f"- {m}" for m in memory_bank[-10:])
        prompt += f"\n\n[Past cross-cultural interaction memories]\n{memories}"

    return prompt


def build_probe_prompts(test_type, questions_file):
    df = pd.read_csv(questions_file)

    if test_type == "trait":
        question_col = "question" if "question" in df.columns else "query"
        high1_col = "high1" if "high1" in df.columns else "response_high1"
        low1_col = "low1" if "low1" in df.columns else "response_low1"
        high2_col = "high2" if "high2" in df.columns else "response_high2"
        low2_col = "low2" if "low2" in df.columns else "response_low2"
        user_prompts = [
            f"Please read the following statement carefully.\n"
            f"Question: {row['situation']} {row[question_col]}\n"
            f"Please respond with how you would react in EXACTLY 1 sentence."
            for _, row in df.iterrows()
        ]
        answer_texts = [
            [row[high1_col], row[low1_col], row[high2_col], row[low2_col]]
            for _, row in df.iterrows()
        ]
        traits = [t[0].upper() for t in df.trait.values]
        return user_prompts, ["a", "b", "c", "d"], answer_texts, traits

    if test_type == "cp":
        user_prompts = [
            f"Question: {row['scenario_text']} {row['question']}\n"
            f"Please respond with how you would react in EXACTLY 1 sentence."
            for _, row in df.iterrows()
        ]
        answer_texts = [
            [
                row["moderately_high"],
                row["low"],
                row["high"],
                row["medium"],
                row["moderately_low"],
            ]
            for _, row in df.iterrows()
        ]
        traits = [t[0].upper() for t in df.trait.values]
        return user_prompts, ["a", "b", "c", "d", "e"], answer_texts, traits

    user_prompts = [
        f"Please read the following dialogue carefully.\n"
        f"Question: {row['train_input']}\n"
        f"Please respond with how you would respond in EXACTLY 1 sentence."
        for _, row in df.iterrows()
    ]
    answer_texts = [[row["high_output"], row["low_output"]] for _, row in df.iterrows()]
    traits = [t[0].upper() for t in df.trait.values]
    return user_prompts, ["a", "b"], answer_texts, traits


def select_balanced_indices(traits, max_questions):
    if not max_questions or max_questions >= len(traits):
        return list(range(len(traits)))

    trait_order = ["O", "C", "E", "A", "N"]
    indices_by_trait = {trait: [] for trait in trait_order}
    for idx, trait in enumerate(traits):
        if trait in indices_by_trait:
            indices_by_trait[trait].append(idx)

    selected = []
    cursor = 0
    while len(selected) < max_questions:
        progressed = False
        for trait in trait_order:
            trait_indices = indices_by_trait[trait]
            if cursor < len(trait_indices):
                selected.append(trait_indices[cursor])
                progressed = True
                if len(selected) == max_questions:
                    break
        if not progressed:
            break
        cursor += 1

    return sorted(selected)


def is_openai_model(model_name):
    return model_name not in {"llama", "llama3", "qwen"}


def resolve_hf_model_name(model_name, hf_model_name=None):
    if hf_model_name:
        return hf_model_name
    if model_name in {"llama", "llama3"}:
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    if model_name == "qwen":
        return "Qwen/Qwen2-7B-Instruct"
    raise ValueError(f"Unsupported local model: {model_name}")


def load_hf_engine(model_name, hf_model_name=None, load_in_4bit=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    resolved_name = resolve_hf_model_name(model_name, hf_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_name,
        trust_remote_code=(model_name == "qwen"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        resolved_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        trust_remote_code=(model_name == "qwen"),
    )
    model.eval()
    return {
        "backend": "hf",
        "model_name": resolved_name,
        "model": model,
        "tokenizer": tokenizer,
    }


def build_generation_engine(model_name, hf_model_name=None, load_in_4bit=True):
    if is_openai_model(model_name):
        return {"backend": "openai", "client": OpenAI()}
    return load_hf_engine(model_name, hf_model_name, load_in_4bit)


def generate_chat_completion(engine, messages, model_name, temperature=0.7, max_tokens=150):
    if engine["backend"] == "openai":
        response = engine["client"].chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    tokenizer = engine["tokenizer"]
    model = engine["model"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt += "\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_agent_response(engine, system_prompt, conversation_history, new_message, model):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend({"role": role, "content": content} for role, content in conversation_history)
    messages.append({"role": "user", "content": new_message})

    reply = generate_chat_completion(engine, messages, model, temperature=0.7, max_tokens=150)
    return reply, conversation_history + [("user", new_message), ("assistant", reply)]


def run_probe(engine, agent_name, system_prompt, user_prompts, options, answer_choices, embedder, model):
    rows = []
    for i, user_prompt in enumerate(user_prompts):
        reply, _ = generate_agent_response(engine, system_prompt, [], user_prompt, model)
        choices = answer_choices[i]
        emb_gen = embedder.encode(reply, convert_to_tensor=True)
        emb_opts = embedder.encode(choices, convert_to_tensor=True)
        sims = util.cos_sim(emb_gen, emb_opts)[0]
        sim_probs = F.softmax(sims, dim=0)
        best_idx = torch.argmax(sim_probs).item()

        row = {
            "agent": agent_name,
            "prompt": user_prompt,
            "response": reply,
            "matched_option": options[best_idx],
            "matched_text": choices[best_idx],
            "cosine_sim": sim_probs[best_idx].item(),
        }
        for opt, sim in zip(options, sim_probs):
            row[f"sim_{opt}"] = sim.item()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_memory(engine, observer_country, partner_country, conversation, model):
    lines = "\n".join(f"{speaker}: {text}" for speaker, text in conversation)
    prompt = f"""Summarize what the {observer_country} person learned about {partner_country} culture
from this conversation in 1-2 sentences. Focus on values, behaviors, norms, and possible adaptation cues.

Conversation:
{lines}

Summary:"""
    return generate_chat_completion(
        engine,
        [{"role": "user", "content": prompt}],
        model,
        temperature=0.3,
        max_tokens=100,
    )


def run_interaction(engine, agent_a, agent_b, opening_topic, n_turns, model):
    a_history = []
    b_history = []
    transcript = []

    current_message = opening_topic
    for turn in range(n_turns):
        a_reply, a_history = generate_agent_response(engine, agent_a["system"], a_history, current_message, model)
        transcript.append((agent_a["name"], a_reply))

        b_reply, b_history = generate_agent_response(engine, agent_b["system"], b_history, a_reply, model)
        transcript.append((agent_b["name"], b_reply))
        current_message = b_reply

        print(f"  T{turn + 1} {agent_a['name']}: {a_reply[:80]}...")
        print(f"  T{turn + 1} {agent_b['name']}: {b_reply[:80]}...")

    return transcript


def get_samples(data, traits, test_type, samples=100):
    buckets = {trait: [] for trait in ["O", "C", "E", "A", "N"]}
    if test_type == "cp":
        probs = ["sim_a", "sim_b", "sim_c", "sim_d", "sim_e"]
        population = [4, 1, 5, 3, 2]
    elif test_type == "trait":
        probs = ["sim_a", "sim_b", "sim_c", "sim_d"]
        population = [5, 1, 5, 1]
    else:
        probs = ["sim_a", "sim_b"]
        population = [5, 1]

    for i, row in data.iterrows():
        weights = row[probs].values.astype(float).tolist()
        sampled_answers = random.choices(population=population, weights=weights, k=samples)
        buckets[traits[i]].extend(sampled_answers)

    def avg_trait(values):
        if not values:
            return [np.nan] * samples
        arr = np.array(values).reshape(-1, samples)
        return np.mean(arr, axis=0)

    return pd.DataFrame({trait: avg_trait(buckets[trait]) for trait in ["O", "C", "E", "A", "N"]})


def country_code_for_gt(country):
    mapping = {
        "United States": "USA",
        "Brazil": "Brazil",
        "India": "India",
        "Japan": "Japan",
        "Saudi Arabia": "Saudi Arab",
        "South Africa": "South Afri",
        "Korea": "Korea",
    }
    return mapping.get(normalize_country(country), normalize_country(country))


def generate_model_dist(df, traits, test_type, ground_truth, country_code):
    c_gt = ground_truth[ground_truth.country == country_code]
    sample_count = len(c_gt) if len(c_gt) else 100
    return get_samples(df, traits, test_type, samples=sample_count)


def get_gt_trait_values(gt, country_code, trait):
    country_gt = gt[gt.country == country_code]
    if country_gt.empty:
        return pd.Series(dtype=float)

    if trait in country_gt.columns:
        return pd.to_numeric(country_gt[trait], errors="coerce").dropna()

    if "trait" in country_gt.columns:
        trait_gt = country_gt[country_gt["trait"] == trait]
        value_cols = [col for col in trait_gt.columns if col not in {"country", "trait"}]
        if not value_cols:
            return pd.Series(dtype=float)
        return pd.to_numeric(trait_gt[value_cols[0]], errors="coerce").dropna()

    return pd.Series(dtype=float)


def calc_kl_for_trait(gt_values, model_values):
    gt_values = pd.Series(gt_values).dropna().astype(float)
    model_values = pd.Series(model_values).dropna().astype(float)
    if len(gt_values) < 2 or len(model_values) < 2:
        return np.nan

    if gt_values.nunique() < 2 or model_values.nunique() < 2:
        return np.nan

    common_grid = np.linspace(
        min(gt_values.min(), model_values.min()),
        max(gt_values.max(), model_values.max()),
        100,
    )
    kde_gt = gaussian_kde(gt_values, bw_method=0.3)
    kde_model = gaussian_kde(model_values, bw_method=0.3)
    gt_density = kde_gt(common_grid)
    model_density = kde_model(common_grid)
    gt_density /= np.trapz(gt_density, common_grid)
    model_density /= np.trapz(model_density, common_grid)
    return entropy(gt_density + 1e-10, model_density + 1e-10)


def generate_metrics_if_available(gt, model_scores, country_code):
    if country_code not in set(gt.country.dropna().unique()):
        return None

    divergences = []
    ks_stats = []
    ks_pvals = []
    for trait in ["O", "C", "E", "A", "N"]:
        gt_values = get_gt_trait_values(gt, country_code, trait)
        model_values = pd.to_numeric(model_scores[trait], errors="coerce").dropna()

        divergences.append(calc_kl_for_trait(gt_values, model_values))
        if len(gt_values) < 2 or len(model_values) < 2:
            ks_stats.append(np.nan)
            ks_pvals.append(np.nan)
        else:
            ks_stat, ks_pval = ks_2samp(gt_values, model_values)
            ks_stats.append(ks_stat)
            ks_pvals.append(ks_pval)

    return pd.DataFrame(
        [divergences, ks_stats, ks_pvals],
        index=["KL Divergence", "KS Stat", "KS P-Value"],
        columns=["O", "C", "E", "A", "N"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--questions", default="datasets/baselines/trait.csv")
    parser.add_argument("-t", "--test_type", default="trait")
    parser.add_argument("--agent_a_country", default="Brazil")
    parser.add_argument("--agent_b_country", default="Saudi Arabia")
    parser.add_argument("--norm_file", default="datasets/cp/norms/openai_cultural_norms.json")
    parser.add_argument("--agent_b_norm_file", default=None)
    parser.add_argument("-gt", "--ground_truth", default="datasets/ground-truth/big-five-ocean.csv")
    parser.add_argument("-m", "--model", default="gpt-4o")
    parser.add_argument("--memory_model", default=None)
    parser.add_argument("--hf_model_name", default=None)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--n_rounds", type=int, default=3)
    parser.add_argument("--n_turns", type=int, default=2)
    parser.add_argument("--max_questions", type=int, default=20)
    parser.add_argument("--results_dir", default="mai_results")
    args = parser.parse_args()

    if args.memory_model is None:
        args.memory_model = args.model

    if (is_openai_model(args.model) or is_openai_model(args.memory_model)) and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found.")

    generation_engine = build_generation_engine(
        args.model,
        hf_model_name=args.hf_model_name,
        load_in_4bit=not args.no_4bit,
    )
    memory_engine = generation_engine
    if args.memory_model != args.model:
        memory_engine = build_generation_engine(
            args.memory_model,
            load_in_4bit=not args.no_4bit,
        )

    os.makedirs(args.results_dir, exist_ok=True)

    a_country = normalize_country(args.agent_a_country)
    b_country = normalize_country(args.agent_b_country)
    a_norms = load_norms(args.norm_file, a_country)
    b_norms = load_norms(args.agent_b_norm_file or args.norm_file, b_country)

    user_prompts, options, answer_choices, traits = build_probe_prompts(args.test_type, args.questions)
    selected_indices = select_balanced_indices(traits, args.max_questions)
    user_prompts = [user_prompts[i] for i in selected_indices]
    answer_choices = [answer_choices[i] for i in selected_indices]
    traits = [traits[i] for i in selected_indices]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    gt = pd.read_csv(args.ground_truth)

    a_memory = []
    b_memory = []

    print("=== PRE-MEASUREMENT ===")
    a_pre_system = build_system_prompt(a_country, a_norms)
    b_pre_system = build_system_prompt(b_country, b_norms)
    a_pre = run_probe(generation_engine, a_country, a_pre_system, user_prompts, options, answer_choices, embedder, args.model)
    b_pre = run_probe(generation_engine, b_country, b_pre_system, user_prompts, options, answer_choices, embedder, args.model)
    a_pre.to_csv(os.path.join(args.results_dir, f"pre_{a_country}.csv"), index=False)
    b_pre.to_csv(os.path.join(args.results_dir, f"pre_{b_country}.csv"), index=False)

    topics = [
        "How does your family usually make important life decisions?",
        "How do people around you handle being late to a meeting?",
        "What would you do if a manager asked you to work late without notice?",
        "How do people show respect to older relatives or senior coworkers?",
        "What makes a celebration or gathering feel meaningful in your culture?",
    ]

    print("\n=== INTERACTION ROUNDS ===")
    for round_idx in range(args.n_rounds):
        print(f"\n--- Round {round_idx + 1}/{args.n_rounds} ---")
        a_agent = {
            "name": a_country,
            "system": build_system_prompt(a_country, a_norms, a_memory),
        }
        b_agent = {
            "name": b_country,
            "system": build_system_prompt(b_country, b_norms, b_memory),
        }
        topic = topics[round_idx % len(topics)]
        transcript = run_interaction(generation_engine, a_agent, b_agent, topic, args.n_turns, args.model)

        a_summary = summarize_memory(memory_engine, a_country, b_country, transcript, args.memory_model)
        b_summary = summarize_memory(memory_engine, b_country, a_country, transcript, args.memory_model)
        a_memory.append(a_summary)
        b_memory.append(b_summary)
        print(f"  [{a_country} memory]: {a_summary}")
        print(f"  [{b_country} memory]: {b_summary}")

    with open(os.path.join(args.results_dir, "memories.json"), "w", encoding="utf-8") as f:
        json.dump({a_country: a_memory, b_country: b_memory}, f, ensure_ascii=False, indent=2)

    print("\n=== POST-MEASUREMENT ===")
    a_post_system = build_system_prompt(a_country, a_norms, a_memory)
    b_post_system = build_system_prompt(b_country, b_norms, b_memory)
    a_post = run_probe(generation_engine, a_country, a_post_system, user_prompts, options, answer_choices, embedder, args.model)
    b_post = run_probe(generation_engine, b_country, b_post_system, user_prompts, options, answer_choices, embedder, args.model)
    a_post.to_csv(os.path.join(args.results_dir, f"post_{a_country}.csv"), index=False)
    b_post.to_csv(os.path.join(args.results_dir, f"post_{b_country}.csv"), index=False)

    print("\n=== METRICS ===")
    metric_frames = {}
    for agent_country, phase, df in [
        (a_country, "pre", a_pre),
        (a_country, "post", a_post),
        (b_country, "pre", b_pre),
        (b_country, "post", b_post),
    ]:
        dist = generate_model_dist(df, traits, args.test_type, gt, country_code_for_gt(agent_country))
        dist.to_csv(os.path.join(args.results_dir, f"{phase}_{agent_country}_dist.csv"), index=False)

        for reference_country in [a_country, b_country]:
            reference_code = country_code_for_gt(reference_country)
            metrics = generate_metrics_if_available(gt, dist, reference_code)
            if metrics is None:
                print(f"Skipping {agent_country} {phase} vs {reference_country}: no GT for {reference_code}")
                continue
            key = f"{phase}_{agent_country}_vs_{reference_country}"
            metric_frames[key] = metrics
            metrics.to_csv(os.path.join(args.results_dir, f"{key}_metrics.csv"))
            print(f"\n[{key}]\n{metrics}")

    summary_rows = []
    for agent_country, target_country in [(a_country, b_country), (b_country, a_country)]:
        pre_key = f"pre_{agent_country}_vs_{target_country}"
        post_key = f"post_{agent_country}_vs_{target_country}"
        if pre_key in metric_frames and post_key in metric_frames:
            delta = (
                metric_frames[post_key].loc["KL Divergence"]
                - metric_frames[pre_key].loc["KL Divergence"]
            )
            summary_rows.append({
                "agent": agent_country,
                "target": target_country,
                "mean_target_kl_delta": delta.mean(),
                **{f"{trait}_target_kl_delta": delta[trait] for trait in ["O", "C", "E", "A", "N"]},
            })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(os.path.join(args.results_dir, "adaptation_summary.csv"), index=False)
        print("\n[Adaptation summary]\n", summary)


if __name__ == "__main__":
    main()
