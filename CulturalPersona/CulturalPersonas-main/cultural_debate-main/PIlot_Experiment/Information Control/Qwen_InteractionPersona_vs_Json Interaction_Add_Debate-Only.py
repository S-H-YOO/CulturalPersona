import argparse
import datetime
import json
import os

import Qwen_Normad_Debate as debate
import qwen_no_rot_oeg_mcs_after_interaction as base
import qwen_no_rot_oeg_mcs_after_interaction_norm_control as norm_control


def normalize_key(value):
    return base.exp.normalize_country_key(str(value))


def load_json_agents(path):
    if not path:
        raise ValueError("--json_agent_path is required.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON agent file not found: {path}")

    with open(path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    agents = {}
    for key, record in data.items():
        if not isinstance(record, dict):
            record = {"content": str(record)}
        country_name = record.get("country_name") or key.replace("_", " ").title()
        content = str(record.get("content", "")).strip()
        system_prompt = str(record.get("agent_system_prompt", "")).strip()
        if not system_prompt:
            system_prompt = build_json_agent_system_prompt(country_name, content)
        agents[normalize_key(key)] = {
            "country_name": country_name,
            "content": content,
            "system_prompt": system_prompt,
            "source": record.get("source", ""),
        }
    return agents


def get_json_agent(agents, country_key, country_name, cp_country):
    lookup_keys = [
        country_key,
        country_name,
        cp_country,
        country_key.replace("_", " "),
        cp_country.replace("_", " "),
    ]
    for key in lookup_keys:
        agent = agents.get(normalize_key(key))
        if agent:
            return agent
    available = ", ".join(sorted(agents))
    raise KeyError(f"No JSON agent found for {country_key}. Available keys: {available}")


def build_json_agent_system_prompt(country_name, background_context):
    return f"""You are a JSON-grounded cultural background agent for {country_name}.
You are not role-playing as a private individual. You answer only from the cultural
etiquette background provided below.
When discussing social acceptability, use this background to judge what would
likely be respectful, rude, acceptable, or ambiguous in the target culture.
During interaction, you may also use the Rule-of-Thumb shown in the current
story prompt. Do not use gold labels, hidden explanations, or external memories.

[JSON cultural etiquette background for {country_name}]
{background_context}"""


def build_reflection(args, tokenizer, model, country_name, debate_transcript):
    reflection_prompt = base.exp.fill_prompt(
        base.exp.REFLECTION_PROMPT,
        country_name,
        "",
        "",
        target_country=country_name,
        debate_transcript=debate_transcript,
    )
    return base.exp.call_qwen(
        tokenizer,
        model,
        base.exp.PERSONAS["korea"],
        reflection_prompt,
        max_new_tokens=args.reflection_tokens,
        temperature=args.reflection_temp,
    )


def build_debate_only_korea_system_prompt(target_country, target_norms, debate_transcript):
    norm_text = "\n".join(f"- {norm}" for norm in target_norms)
    return f"""You are someone who was born and brought up in Korea.
You currently live in Korea.
You recently completed a cross-cultural interaction with someone from {target_country}.

Important {target_country} cultural norms to consider:
{norm_text}

[Transcript from the NormAd cross-cultural interaction with a {target_country} agent]
{debate_transcript}

Use the transcript directly when judging later {target_country} social
acceptability questions. Do not assume there was a separate self-reflection
summary or additional memory beyond this transcript."""


def build_rot_debate_transcript_without_gold(history, agent_a_name, agent_b_name):
    blocks = []
    for idx, item in enumerate(history, start=1):
        debate_item = item["debate"]
        blocks.append(
            f"[Adapt story {idx}] {item.get('subaxis', '')}\n"
            f"Story: {item['story']}\n"
            f"Rule: {item.get('rule', '')}\n"
            f"{agent_a_name} initial: {debate_item.get(f'{agent_a_name}_initial', '')}\n"
            f"{agent_b_name} initial: {debate_item.get(f'{agent_b_name}_initial', '')}\n"
            f"{agent_a_name} feedback: {debate_item.get(f'{agent_a_name}_feedback', '')}\n"
            f"{agent_b_name} feedback: {debate_item.get(f'{agent_b_name}_feedback', '')}\n"
            f"{agent_a_name} final: {debate_item.get(f'{agent_a_name}_final', '')}\n"
            f"{agent_b_name} final: {debate_item.get(f'{agent_b_name}_final', '')}"
        )
    return "\n\n".join(blocks)


def build_interaction_memory(
    args,
    tokenizer,
    model,
    country_name,
    adapt_rows,
    partner_system_prompt,
    partner_name,
):
    print(f"\n[Adaptation: RoT] {partner_name} <-> korea | n_turns={args.n_turns}")
    adaptation_history = base.exp.run_adaptation_dialogue(
        tokenizer,
        model,
        partner_system_prompt,
        base.exp.PERSONAS["korea"],
        country_name,
        adapt_rows,
        partner_name,
        "korea",
        args.n_turns,
    )
    debate_transcript = build_rot_debate_transcript_without_gold(
        adaptation_history,
        partner_name,
        "korea",
    )
    reflection = build_reflection(args, tokenizer, model, country_name, debate_transcript)
    return adaptation_history, debate_transcript, reflection


def save_condition_interaction_artifacts(
    args,
    country_key,
    country_name,
    condition,
    adaptation_history,
    debate_transcript,
    reflection,
):
    output_dir = os.path.join(args.output_dir, country_key, "interaction", condition)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "interaction_memory.json"), "w", encoding="utf-8") as outfile:
        json.dump(
            {
                "country_key": country_key,
                "country": country_name,
                "condition": condition,
                "interaction_uses_rule_of_thumb": True,
                "adaptation_history": adaptation_history,
                "reflection": reflection,
            },
            outfile,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(output_dir, "debate_transcript.txt"), "w", encoding="utf-8") as outfile:
        outfile.write(debate_transcript)
    with open(os.path.join(output_dir, "reflection.txt"), "w", encoding="utf-8") as outfile:
        outfile.write(reflection)


def build_condition_specs(args, tokenizer, model, country_key, json_agents):
    cp_country = base.abcd.normalize_cp_country(country_key)
    filtered_rows = base.abcd.load_country_rows(
        args.normad_input_path,
        country_key,
        args.limit_per_country,
    )
    if not filtered_rows:
        raise ValueError(f"No NormAd rows found for country={country_key}")

    country_name = base.exp.country_capitalized_mapping.get(
        filtered_rows[0]["Country"],
        filtered_rows[0]["Country"],
    )
    adapt_rows, _ = base.exp.select_adapt_test_rows(
        filtered_rows,
        args.adapt_per_label,
        args.test_per_label,
        args.seed,
    )
    target_norms = base.abcd.load_cp_norms(args.cp_norm_file, cp_country, args.cp_norm_limit)
    json_agent = get_json_agent(json_agents, country_key, country_name, cp_country)

    print("\n" + "=" * 60)
    print(f"NormAd interaction-persona vs JSON-agent: {country_name} ({country_key})")
    print("=" * 60)

    a_history, a_transcript, a_reflection = build_interaction_memory(
        args,
        tokenizer,
        model,
        country_name,
        adapt_rows,
        base.exp.PERSONAS[country_key],
        country_key,
    )
    save_condition_interaction_artifacts(
        args,
        country_key,
        country_name,
        "A_interaction_korea_target_persona",
        a_history,
        a_transcript,
        a_reflection,
    )

    b_history, b_transcript, b_reflection = build_interaction_memory(
        args,
        tokenizer,
        model,
        country_name,
        adapt_rows,
        json_agent["system_prompt"],
        "json_agent",
    )
    save_condition_interaction_artifacts(
        args,
        country_key,
        country_name,
        "B_interaction_korea_json_agent",
        b_history,
        b_transcript,
        b_reflection,
    )

    condition_specs = {
        "A_interaction_korea_target_persona": base.abcd.build_interaction_korea_system_prompt(
            cp_country,
            target_norms,
            a_reflection,
        ),
        "A_debate_only_korea_target_persona": build_debate_only_korea_system_prompt(
            cp_country,
            target_norms,
            a_transcript,
        ),
        "B_interaction_korea_json_agent": base.abcd.build_interaction_korea_system_prompt(
            cp_country,
            target_norms,
            b_reflection,
        ),
        "B_debate_only_korea_json_agent": build_debate_only_korea_system_prompt(
            cp_country,
            target_norms,
            b_transcript,
        ),
    }
    return country_name, cp_country, condition_specs


def run_one_country(args, tokenizer, model, country_key, json_agents):
    country_name, _, condition_specs = build_condition_specs(
        args,
        tokenizer,
        model,
        country_key,
        json_agents,
    )
    return debate.run_normad_condition_evaluation(
        args,
        tokenizer,
        model,
        country_key,
        country_name,
        condition_specs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normad_input_path", default="data/normad.jsonl")
    parser.add_argument("--output_dir", default="outputs/normad_rot_interaction_persona_vs_json_agent_qwen25_14b")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", base.exp.DEFAULT_CACHE_DIR))
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--countries", nargs="+", default=["india", "brazil", "saudi_arabia", "south_africa", "united_states_of_america", "japan"])
    parser.add_argument("--limit_per_country", type=int, default=None)
    parser.add_argument("--adapt_per_label", type=int, default=2)
    parser.add_argument("--test_per_label", type=int, default=None)
    parser.add_argument("--n_turns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reflection_tokens", type=int, default=700)
    parser.add_argument("--reflection_temp", type=float, default=0.3)
    parser.add_argument("--cp_root", default=base.CP_ROOT)
    parser.add_argument("--cp_norm_file", default=None)
    parser.add_argument("--cp_norm_limit", type=int, default=2)
    parser.add_argument("--json_agent_path", default="data/country_etiquette_interaction_agents.json")
    args = parser.parse_args()

    if args.cp_norm_file is None:
        args.cp_norm_file = os.path.join(args.cp_root, "datasets/cp/norms/openai_cultural_norms.json")

    json_agents = load_json_agents(args.json_agent_path)

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    print(f"Loading model once: {args.model_id}")
    tokenizer, model = base.exp.load_qwen(args.model_id, args.cache_dir, gpu_only=args.gpu_only)

    started_at = datetime.datetime.now()
    for country_key in args.countries:
        run_one_country(args, tokenizer, model, country_key, json_agents)

    finished_at = datetime.datetime.now()
    print(f"Started:  {started_at}")
    print(f"Finished: {finished_at}")
    print(f"Elapsed:  {finished_at - started_at}")
    print(f"Outputs:  {args.output_dir}")


if __name__ == "__main__":
    main()
