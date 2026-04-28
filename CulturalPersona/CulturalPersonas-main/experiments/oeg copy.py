import os
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import openai
import numpy as np
from sentence_transformers import SentenceTransformer, util
import random
from metrics import calc_kl, calc_ks

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def build_system_prompt(country, norm1, norm2, memory_bank=None):
    base = f"""You are someone who was born and brought up in {country}.
You are very familiar with {country}'s culture and traditions and practice many of them,
two of which include: {norm1} and {norm2}."""

    if memory_bank:
        memories = "\n".join(f"- {m}" for m in memory_bank[-10:])
        base += f"\n\n[Your past experiences living in India]\n{memories}"

    return base


def build_probe_prompts(test_type, questions_file):
    df = pd.read_csv(questions_file)

    if test_type == "trait":
        user_prompts = [
            f"Please read the following statement carefully.\n"
            f"Question: {row['situation']} {row['question']}\n"
            f"Please respond with how you would react in EXACTLY 1 sentence."
            for _, row in df.iterrows()
        ]
        answer_texts = [
            [row['high1'], row['low1'], row['high2'], row['low2']]
            for _, row in df.iterrows()
        ]
        return user_prompts, ["a", "b", "c", "d"], answer_texts

    elif test_type == "cp":
        user_prompts = [
            f"Question: {row['scenario_text']} {row['question']}\n"
            f"Please respond with how you would react in EXACTLY 1 sentence."
            for _, row in df.iterrows()
        ]
        answer_texts = [
            [row['moderately_high'], row['low'], row['high'],
             row['medium'], row['moderately_low']]
            for _, row in df.iterrows()
        ]
        return user_prompts, ["a", "b", "c", "d", "e"], answer_texts

    else:  # ✅ 원본에 있던 dialogue 타입 - 누락됐던 브랜치
        user_prompts = [
            f"Please read the following dialogue carefully.\n"
            f"Question: {row['train_input']}\n"
            f"Please respond with how you would respond in EXACTLY 1 sentence."
            for _, row in df.iterrows()
        ]
        answer_texts = [
            [row['high_output'], row['low_output']]
            for _, row in df.iterrows()
        ]
        return user_prompts, ["a", "b"], answer_texts


def generate_agent_response(system_prompt, conversation_history,
                             new_message, model="gpt-4o"):
    messages = [{"role": "system", "content": system_prompt}]
    for role, content in conversation_history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": new_message})

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    reply = response["choices"][0]["message"]["content"].strip()

    updated_history = conversation_history + [
        ("user", new_message),
        ("assistant", reply)
    ]
    return reply, updated_history


def get_samples(data, reverse, traits, test_type, samples=100):
    o_, c_, e_, a_, n_ = [], [], [], [], []
    if test_type == "cp":
        probs = ['sim_a', 'sim_b', 'sim_c', 'sim_d', 'sim_e']
        population = [4, 1, 5, 3, 2]
    elif test_type == "trait":
        probs = ['sim_a', 'sim_b', 'sim_c', 'sim_d']
        population = [5, 1, 5, 1]
    else:
        probs = ['sim_a', 'sim_b']
        population = [5, 1]

    for i, row in data.iterrows():
        weights = row[probs].values.astype(float).tolist()
        sampled_answers = random.choices(population=population, weights=weights, k=samples)
        trait = traits[i]
        if trait == 'O':
            o_.extend(sampled_answers)
        elif trait == 'C':
            c_.extend(sampled_answers)
        elif trait == 'E':
            e_.extend(sampled_answers)
        elif trait == 'A':
            a_.extend(sampled_answers)
        else:
            n_.extend(sampled_answers)

    def avg_trait(trait_list):
        arr = np.array(trait_list).reshape(-1, samples)
        return np.mean(arr, axis=0)

    return (
        avg_trait(o_).tolist(), avg_trait(c_).tolist(), avg_trait(e_).tolist(),
        avg_trait(a_).tolist(), avg_trait(n_).tolist()
    )


def generate_model_dist(out_file, reverse_values, traits, test_type,
                         ground_truth_file, country_code):
    data = pd.read_csv(out_file)
    ground_truth = pd.read_csv(ground_truth_file)
    c_gt = ground_truth[ground_truth.country == country_code]
    o, c, e, a, n = get_samples(data, reverse_values, traits, test_type,
                                  samples=len(c_gt))
    return pd.DataFrame({"O": o, "C": c, "E": e, "A": a, "N": n})


def generate_metrics(gt, model_scores, country_code):
    divergences = calc_kl(gt, model_scores, country_code)
    ks_stats, ks_pvals = calc_ks(gt, model_scores, country_code)
    df = pd.DataFrame(
        [divergences, ks_stats, ks_pvals],
        index=['KL Divergence', 'KS Stat', 'KS P-Value'],
        columns=['O', 'C', 'E', 'A', 'N']
    )
    return df


def summarize_to_memory(conversation_history, model="gpt-4o-mini"):
    conv_text = "\n".join(
        f"{'Brazilian' if r == 'assistant' else 'Indian'}: {c}"  # ✅ Korean → Brazilian
        for r, c in conversation_history
    )
    summary_prompt = f"""
Summarize what the Brazilian person experienced or learned about Indian culture
from this conversation in 1-2 sentences. Focus on cultural values, behaviors,
or social norms encountered.

Conversation:
{conv_text}

Summary:"""

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3,
        max_tokens=100
    )
    return resp["choices"][0]["message"]["content"].strip()


def run_interaction(brazil_system, indian_system, opening_topic,
                    n_turns=2, model="gpt-4o"):
    brazil_history = []
    indian_history = []

    indian_reply, indian_history = generate_agent_response(
        indian_system, indian_history, opening_topic, model
    )
    current_message = indian_reply
    print(f"  [Indian]:    {indian_reply[:80]}...")

    for turn in range(n_turns):
        brazil_reply, brazil_history = generate_agent_response(
            brazil_system, brazil_history, current_message, model
        )
        print(f"  [Brazilian T{turn+1}]: {brazil_reply[:80]}...")

        indian_reply, indian_history = generate_agent_response(
            indian_system, indian_history, brazil_reply, model
        )
        print(f"  [Indian    T{turn+1}]: {indian_reply[:80]}...")

        current_message = indian_reply

    return brazil_history


def run_probe(system_prompt, user_prompts, options,
              all_answer_choices, embedder, model="gpt-4o"):
    rows = []
    for i, user_prompt in enumerate(user_prompts):
        reply, _ = generate_agent_response(system_prompt, [], user_prompt, model)

        choices   = all_answer_choices[i]
        emb_gen   = embedder.encode(reply,    convert_to_tensor=True)
        emb_opts  = embedder.encode(choices,  convert_to_tensor=True)
        sims      = util.cos_sim(emb_gen, emb_opts)[0]
        sim_probs = F.softmax(sims, dim=0)
        best_idx  = torch.argmax(sim_probs).item()

        row = {"prompt": user_prompt, "response": reply,
               "matched_option": options[best_idx]}
        for opt, sim in zip(options, sim_probs):
            row[f"sim_{opt}"] = sim.item()
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")  # ✅ HuggingFace 코드 제거
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY not found.")

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--questions',     default='ipip-120.csv')
    parser.add_argument('-t', '--test_type',     default="trait")
    parser.add_argument('-c', '--country',       default="brazil",
                        choices=["usa", "brazil", "southafrica",
                                 "saudiarabia", "india", "japan"])
    parser.add_argument('-n', '--norm_file',     default="country_norms.csv")
    parser.add_argument('-gt', '--ground_truth', default="datasets/ground-truth/big-five-ocean.csv")
    parser.add_argument('-m',  '--model',        default="gpt-4o")
    parser.add_argument('--n_rounds', type=int,  default=3)   # 파일럿
    parser.add_argument('--n_turns',  type=int,  default=2)   # 파일럿
    parser.add_argument('--results_file',        default="results.csv")
    args = parser.parse_args()

    # ── 노름 로드 ─────────────────────────────────────────────────────
    norms_df = pd.read_csv(args.norm_file)
    brazil_norms = norms_df[
        norms_df.country.str.strip().str.lower() == "brazil"
    ].norm.values
    indian_norms = norms_df[
        norms_df.country.str.strip().str.lower() == "india"
    ].norm.values

    if len(brazil_norms) < 2 or len(indian_norms) < 2:
        raise ValueError("Not enough norms for Brazil or India")

    # ── Probe 질문 로드 ───────────────────────────────────────────────
    user_prompts, options, answer_choices = build_probe_prompts(
        args.test_type, args.questions
    )
    reverse_df     = pd.read_csv(args.questions)
    traits         = [t[0].upper() for t in reverse_df.trait.values]
    reverse_values = []

    # 파일럿: 앞 20개만
    user_prompts   = user_prompts[:20]
    answer_choices = answer_choices[:20]
    traits         = traits[:20]

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    country_map = {
        "usa": "USA", "brazil": "Brazil", "india": "India",
        "japan": "Japan", "saudiarabia": "Saudi Arab",
        "southafrica": "South Afri"
    }
    country_code = country_map[args.country]

    # ── STEP 1: 사전 측정 ────────────────────────────────────────────
    print("=== PRE-MEASUREMENT ===")
    brazil_system_pre = build_system_prompt(
        "Brazil", brazil_norms[0], brazil_norms[1]
    )
    pre_df = run_probe(
        brazil_system_pre, user_prompts, options,
        answer_choices, embedder, args.model
    )
    pre_df.to_csv("pre_" + args.results_file, index=False)
    print(f"Pre-probe done. {len(pre_df)} rows saved.\n")

    # ── STEP 2: 상호작용 라운드 ──────────────────────────────────────
    indian_sub_personas = [
        "\nYou are a software engineer living in Mumbai.",
        "\nYou are a middle-class homemaker from Delhi.",
        "\nYou are a graduate student from Chennai.",
    ]
    indian_systems = [
        build_system_prompt("India", indian_norms[0], indian_norms[1]) + sub
        for sub in indian_sub_personas
    ]
    interaction_topics = [
        "I'm planning a big family wedding next month with 500 guests. It's overwhelming!",
        "My manager asked me to stay late again without notice. Does this happen to you?",
        "I feel like time moves so differently here than back home. Do you think punctuality matters?",
        "In my family, big decisions are always made together. How is it in your culture?",
        "I believe things happen for a reason — what do you think about fate and hard work?",
    ]
    memory_bank = []

    print("=== INTERACTION ROUNDS ===")
    for round_idx in range(args.n_rounds):
        print(f"\n--- Round {round_idx + 1}/{args.n_rounds} ---")

        brazil_system = build_system_prompt(
            "Brazil", brazil_norms[0], brazil_norms[1],
            memory_bank=memory_bank
        )
        indian_system = indian_systems[round_idx % len(indian_systems)]
        opening_topic = interaction_topics[round_idx % len(interaction_topics)]

        brazil_history = run_interaction(
            brazil_system, indian_system, opening_topic,
            n_turns=args.n_turns, model=args.model
        )
        memory = summarize_to_memory(brazil_history, model="gpt-4o-mini")
        memory_bank.append(memory)
        print(f"  [Memory]: {memory}")

    # ── STEP 3: 사후 측정 ────────────────────────────────────────────
    print("\n=== POST-MEASUREMENT ===")
    brazil_system_post = build_system_prompt(
        "Brazil", brazil_norms[0], brazil_norms[1],
        memory_bank=memory_bank
    )
    post_df = run_probe(
        brazil_system_post, user_prompts, options,
        answer_choices, embedder, args.model
    )
    post_df.to_csv("post_" + args.results_file, index=False)
    print(f"Post-probe done. {len(post_df)} rows saved.\n")

    # ── STEP 4: 메트릭 비교 ──────────────────────────────────────────
    gt = pd.read_csv(args.ground_truth)

    pre_dist  = generate_model_dist(
        "pre_"  + args.results_file, reverse_values, traits,
        args.test_type, args.ground_truth, country_code
    )
    post_dist = generate_model_dist(
        "post_" + args.results_file, reverse_values, traits,
        args.test_type, args.ground_truth, country_code
    )

    pre_metrics  = generate_metrics(gt, pre_dist,  country_code)
    post_metrics = generate_metrics(gt, post_dist, country_code)
    pre_india    = generate_metrics(gt, pre_dist,  "India")
    post_india   = generate_metrics(gt, post_dist, "India")

    print("=== RESULTS ===")
    print("[PRE  vs Brazil GT]\n", pre_metrics)
    print("[POST vs Brazil GT]\n", post_metrics)
    print("[DELTA Brazil KL (post-pre)]\n",
          post_metrics.loc['KL Divergence'] - pre_metrics.loc['KL Divergence'])
    print("[DELTA India alignment KL (post-pre)]\n",
          post_india.loc['KL Divergence'] - pre_india.loc['KL Divergence'])

    summary = pd.concat(
        [pre_metrics.add_prefix("pre_"), post_metrics.add_prefix("post_")],
        axis=1
    )
    summary.to_csv(args.results_file)
    print(f"\nResults saved to {args.results_file}")


if __name__ == '__main__':
    main()