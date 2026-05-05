'''
A : RoT X + [Debate + Self Reflection] + No Json
B : RoT X + [Debate] + No Json
C : RoT X + Json
A : RoT O + [Debate + Self Reflection] + No Json
A : RoT O + [Debate] + No Json
'''
import argparse
import datetime
import os

import Qwen_Normad_Debate as debate
import Qwen_Normad_Debate_Wikipedia as json_background
import qwen_no_rot_oeg_mcs_after_interaction as base
import qwen_no_rot_oeg_mcs_after_interaction_norm_control as norm_control


def build_interaction_no_persona_system_prompt(target_country, target_norms, reflection):
    norm_text = "\n".join(f"- {norm}" for norm in target_norms)
    return f"""You are a helpful assistant.

Important {target_country} cultural norms to consider:
{norm_text}

[Memory from a NormAd cross-cultural interaction with a {target_country} agent]
{reflection}

Use the provided norms and interaction memory when judging {target_country}
social acceptability questions.
Do not adopt a specific personal, national, or study-abroad identity."""


def build_debate_only_korea_system_prompt(target_country, target_norms, debate_transcript, interaction_type):
    norm_text = "\n".join(f"- {norm}" for norm in target_norms)
    return f"""You are someone who was born and brought up in Korea.
You currently live in Korea.
You previously completed a {interaction_type} cross-cultural interaction with a
{target_country} agent.

Important {target_country} cultural norms to consider:
{norm_text}

[Raw transcript from the NormAd cross-cultural interaction with a {target_country} agent]
{debate_transcript}

Use the raw interaction transcript directly when judging {target_country}
social acceptability questions.
Do not assume there was a separate self-reflection summary or additional memory."""


def build_no_interaction_no_persona_system_prompt(target_country, target_norms):
    norm_text = "\n".join(f"- {norm}" for norm in target_norms)
    return f"""You are a helpful assistant.

Important {target_country} cultural norms to consider:
{norm_text}

Use the provided norms when judging {target_country} social acceptability
questions.
You did not complete a cross-cultural interaction with a {target_country} agent.
Do not adopt a specific personal, national, or study-abroad identity."""


def build_json_no_interaction_no_persona_system_prompt(target_country, target_norms, background_context):
    norm_text = "\n".join(f"- {norm}" for norm in target_norms)
    return f"""You are a helpful assistant.

Important {target_country} cultural norms to consider:
{norm_text}

[General etiquette and cultural background about {target_country}]
{background_context}

Use the provided norms and background when judging {target_country} social
acceptability questions.
You did not complete a cross-cultural interaction with a {target_country} agent.
Do not adopt a specific personal, national, or study-abroad identity."""


def build_json_no_interaction_korea_persona_system_prompt(target_country, target_norms, background_context):
    norm_text = "\n".join(f"- {norm}" for norm in target_norms)
    return f"""You are someone who was born and brought up in Korea.
You currently live in Korea.

Important {target_country} cultural norms to consider:
{norm_text}

[General etiquette and cultural background about {target_country}]
{background_context}

Use the provided norms and background when judging {target_country} social
acceptability questions.
You did not complete a cross-cultural interaction with a {target_country} agent.
Do not assume you remember any debate transcript, another agent's arguments,
Rules-of-Thumb from adapt stories, or any reflection summary."""


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


def build_rot_memory_without_gold(args, tokenizer, model, country_key):
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

    print("\n" + "=" * 60)
    print(f"NormAd RoT interaction memory: {country_name} ({country_key})")
    print("=" * 60)
    adaptation_history = base.exp.run_adaptation_dialogue(
        tokenizer,
        model,
        base.exp.PERSONAS[country_key],
        base.exp.PERSONAS["korea"],
        country_name,
        adapt_rows,
        country_key,
        "korea",
        args.n_turns,
    )
    debate_transcript = build_rot_debate_transcript_without_gold(
        adaptation_history,
        country_key,
        "korea",
    )
    reflection_prompt = base.exp.fill_prompt(
        base.exp.REFLECTION_PROMPT,
        country_name,
        "",
        "",
        target_country=country_name,
        debate_transcript=debate_transcript,
    )
    reflection = base.exp.call_qwen(
        tokenizer,
        model,
        base.exp.PERSONAS["korea"],
        reflection_prompt,
        max_new_tokens=args.reflection_tokens,
        temperature=args.reflection_temp,
    )
    return country_name, adaptation_history, debate_transcript, reflection


def build_condition_specs(args, tokenizer, model, country_key, background_contexts):
    cp_country = base.abcd.normalize_cp_country(country_key)
    (
        country_name,
        no_rot_adaptation_history,
        no_rot_debate_transcript,
        no_rot_reflection,
    ) = base.abcd.build_no_rot_memory(args, tokenizer, model, country_key)

    norm_control.save_interaction_artifacts(
        args,
        country_key,
        country_name,
        no_rot_adaptation_history,
        no_rot_debate_transcript,
        no_rot_reflection,
    )

    (
        _rot_country_name,
        _rot_adaptation_history,
        rot_debate_transcript,
        rot_reflection,
    ) = build_rot_memory_without_gold(args, tokenizer, model, country_key)

    target_norms = base.abcd.load_cp_norms(args.cp_norm_file, cp_country, args.cp_norm_limit)
    background_context = json_background.get_wikipedia_context(
        background_contexts,
        country_key,
        country_name,
        cp_country,
        args.background_max_chars,
    )

    condition_specs = {
        "A_no_rot_interaction_persona_o": base.abcd.build_interaction_korea_system_prompt(
            cp_country,
            target_norms,
            no_rot_reflection,
        ),
        "B_no_rot_interaction_debate_only_persona_o": build_debate_only_korea_system_prompt(
            cp_country,
            target_norms,
            no_rot_debate_transcript,
            "no-Rule-of-Thumb",
        ),
        "C_persona_o_json": build_json_no_interaction_korea_persona_system_prompt(
            cp_country,
            target_norms,
            background_context,
        ),
        "D_rot_interaction_persona_o": base.abcd.build_interaction_korea_system_prompt(
            cp_country,
            target_norms,
            rot_reflection,
        ),
        "E_rot_interaction_debate_only_persona_o": build_debate_only_korea_system_prompt(
            cp_country,
            target_norms,
            rot_debate_transcript,
            "Rule-of-Thumb",
        ),
    }
    return country_name, cp_country, condition_specs


def run_one_country(args, tokenizer, model, country_key, background_contexts):
    country_name, _, condition_specs = build_condition_specs(
        args,
        tokenizer,
        model,
        country_key,
        background_contexts,
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
    parser.add_argument("--output_dir", default="outputs/normad_selected_interaction_rot_no_rot_json_5cond_qwen25_14b")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", base.exp.DEFAULT_CACHE_DIR))
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--countries", nargs="+", default=["india", "brazil", "saudi_arabia", "south_africa", "united_states_of_america"])
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
    parser.add_argument("--background_context_path", default="data/country_wikipedia_culture_selected.json")
    parser.add_argument("--background_max_chars", type=int, default=6000)
    args = parser.parse_args()

    if args.cp_norm_file is None:
        args.cp_norm_file = os.path.join(args.cp_root, "datasets/cp/norms/openai_cultural_norms.json")

    background_contexts = json_background.load_wikipedia_contexts(args.background_context_path)

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    print(f"Loading model once: {args.model_id}")
    tokenizer, model = base.exp.load_qwen(args.model_id, args.cache_dir, gpu_only=args.gpu_only)

    started_at = datetime.datetime.now()
    for country_key in args.countries:
        run_one_country(args, tokenizer, model, country_key, background_contexts)

    finished_at = datetime.datetime.now()
    print(f"Started:  {started_at}")
    print(f"Finished: {finished_at}")
    print(f"Elapsed:  {finished_at - started_at}")
    print(f"Outputs:  {args.output_dir}")


if __name__ == "__main__":
    main()
