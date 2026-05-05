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


def build_condition_specs(args, tokenizer, model, country_key, background_contexts):
    cp_country = base.abcd.normalize_cp_country(country_key)
    (
        country_name,
        adaptation_history,
        debate_transcript,
        reflection,
    ) = base.abcd.build_no_rot_memory(args, tokenizer, model, country_key)

    norm_control.save_interaction_artifacts(
        args,
        country_key,
        country_name,
        adaptation_history,
        debate_transcript,
        reflection,
    )

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
            reflection,
        ),
        "B_no_rot_interaction_persona_x": build_interaction_no_persona_system_prompt(
            cp_country,
            target_norms,
            reflection,
        ),
        "C_no_interaction_persona_x": build_no_interaction_no_persona_system_prompt(
            cp_country,
            target_norms,
        ),
        "D_no_interaction_persona_x_json": build_json_no_interaction_no_persona_system_prompt(
            cp_country,
            target_norms,
            background_context,
        ),
        "E_no_interaction_persona_o_json": build_json_no_interaction_korea_persona_system_prompt(
            cp_country,
            target_norms,
            background_context,
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
    parser.add_argument("--output_dir", default="outputs/normad_selected_no_rot_interaction_wikipedia_json_agent_qwen25_14b")
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
