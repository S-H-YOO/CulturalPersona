import argparse
import datetime
import json
import os

import Qwen_Normad_Debate as debate
import Qwen_Normad_Debate_Wikipedia as json_background
import qwen_no_rot_oeg_mcs_after_interaction as base
import qwen_no_rot_oeg_mcs_after_interaction_norm_control as norm_control
import qwen_persona_interaction_no_rot_all_countries as no_rot


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


def build_json_dialogue_korea_messages(target_country, target_norms, background_context):
    norm_text = "\n".join(f"- {norm}" for norm in target_norms)
    system_prompt = f"""You are someone who was born and brought up in Korea.
You currently live in Korea.

Important {target_country} cultural norms to consider:
{norm_text}

You will later judge {target_country} social acceptability questions as this Korean persona."""
    return {
        "kind": "messages",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Before the test, I will give you JSON-style cultural background about {target_country}. "
                    "Read it as contextual background for later judgments. This is not an interaction transcript, "
                    "not a self-reflection memory, and not a Rule-of-Thumb list from adapt stories.\n\n"
                    f"[JSON cultural etiquette background for {target_country}]\n{background_context}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"Understood. I will use the provided JSON cultural background about {target_country} "
                    "as context for later social acceptability judgments, without assuming any prior debate, "
                    "reflection, or adapt-story Rules-of-Thumb."
                ),
            },
        ],
    }


def call_chat(tokenizer, model, messages, max_new_tokens=256, temperature=0.0):
    return debate.call_qwen_messages(
        tokenizer,
        model,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def chat_turn(tokenizer, model, messages, user_prompt, parse_marker, max_new_tokens=256):
    messages.append({"role": "user", "content": user_prompt})
    raw = call_chat(tokenizer, model, messages, max_new_tokens=max_new_tokens)
    response = base.exp.parse_response(raw, parse_marker) if parse_marker else raw
    messages.append({"role": "assistant", "content": response})
    return response


def run_chat_template_debate(
    tokenizer,
    model,
    persona_a,
    persona_b,
    country,
    story,
    rule,
    agent_a_name,
    agent_b_name,
    use_rot,
):
    messages_a = [{"role": "system", "content": persona_a}]
    messages_b = [{"role": "system", "content": persona_b}]

    if use_rot:
        prompt_a1 = base.exp.fill_prompt(base.exp.prompts["prompt_1"], country, story, rule)
        prompt_b1 = base.exp.fill_prompt(base.exp.LEARNER_INITIAL_PROMPT, country, story, rule)
    else:
        prompt_a1 = base.exp.fill_prompt(no_rot.NO_ROT_INITIAL_PROMPT, country, story, "")
        prompt_b1 = base.exp.fill_prompt(no_rot.NO_ROT_INITIAL_PROMPT, country, story, "")

    a1 = chat_turn(tokenizer, model, messages_a, prompt_a1, "Answer:", 256)
    b1 = chat_turn(tokenizer, model, messages_b, prompt_b1, "Answer:", 256)

    if use_rot:
        prompt_a2 = base.exp.fill_prompt(
            base.exp.prompts["prompt_2"],
            country,
            story,
            rule,
            your_response=a1,
            other_response=b1,
        )
        prompt_b2 = base.exp.fill_prompt(
            base.exp.LEARNER_FEEDBACK_PROMPT,
            country,
            story,
            rule,
            your_response=b1,
            other_response=a1,
        )
    else:
        prompt_a2 = base.exp.fill_prompt(
            no_rot.NO_ROT_FEEDBACK_PROMPT,
            country,
            story,
            "",
            your_response=a1,
            other_response=b1,
        )
        prompt_b2 = base.exp.fill_prompt(
            no_rot.NO_ROT_FEEDBACK_PROMPT,
            country,
            story,
            "",
            your_response=b1,
            other_response=a1,
        )

    a2 = chat_turn(tokenizer, model, messages_a, prompt_a2, "Response:", 256)
    b2 = chat_turn(tokenizer, model, messages_b, prompt_b2, "Response:", 256)

    if use_rot:
        prompt_a3 = base.exp.fill_prompt(
            base.exp.prompts["prompt_3"],
            country,
            story,
            rule,
            your_response=a1,
            other_response=b1,
            your_feedback=a2,
            other_feedback=b2,
            feedback=b2,
        )
        prompt_b3 = base.exp.fill_prompt(
            base.exp.LEARNER_FINAL_PROMPT,
            country,
            story,
            rule,
            your_response=b1,
            other_response=a1,
            your_feedback=b2,
            other_feedback=a2,
            feedback=a2,
        )
    else:
        prompt_a3 = base.exp.fill_prompt(
            no_rot.NO_ROT_FINAL_PROMPT,
            country,
            story,
            "",
            your_response=a1,
            other_response=b1,
            your_feedback=a2,
            other_feedback=b2,
        )
        prompt_b3 = base.exp.fill_prompt(
            no_rot.NO_ROT_FINAL_PROMPT,
            country,
            story,
            "",
            your_response=b1,
            other_response=a1,
            your_feedback=b2,
            other_feedback=a2,
        )

    a3 = chat_turn(tokenizer, model, messages_a, prompt_a3, None, 64)
    b3 = chat_turn(tokenizer, model, messages_b, prompt_b3, None, 64)

    transcript = (
        f"{agent_a_name} initial: {a1}\n"
        f"{agent_b_name} initial: {b1}\n"
        f"{agent_a_name} feedback: {a2}\n"
        f"{agent_b_name} feedback: {b2}\n"
        f"{agent_a_name} final: {a3}\n"
        f"{agent_b_name} final: {b3}"
    )
    return {
        f"{agent_a_name}_initial": a1,
        f"{agent_b_name}_initial": b1,
        f"{agent_a_name}_feedback": a2,
        f"{agent_b_name}_feedback": b2,
        f"{agent_a_name}_final": a3,
        f"{agent_b_name}_final": b3,
        f"{agent_a_name}_final_label": base.exp.parse_label(a3),
        f"{agent_b_name}_final_label": base.exp.parse_label(b3),
        "transcript": transcript,
        "chat_messages": {
            agent_a_name: messages_a,
            agent_b_name: messages_b,
        },
    }


def run_chat_template_adaptation_dialogue(
    tokenizer,
    model,
    persona_a,
    persona_b,
    country,
    adapt_rows,
    agent_a_name,
    agent_b_name,
    n_turns,
    use_rot,
):
    history = []
    for idx, row in enumerate(adapt_rows, start=1):
        story = row["Story"]
        rule = row.get("Rule-of-Thumb", "") if use_rot else ""
        subaxis = row.get("Subaxis", "")
        print(f"  [{idx:2d}/{len(adapt_rows)}] [{subaxis}] {story[:70]}...")
        debate_item = run_chat_template_debate(
            tokenizer,
            model,
            persona_a,
            persona_b,
            country,
            story,
            rule,
            agent_a_name,
            agent_b_name,
            use_rot,
        )
        print(
            f"    {agent_a_name}={debate_item.get(f'{agent_a_name}_final_label', ''):<8} "
            f"{agent_b_name}={debate_item.get(f'{agent_b_name}_final_label', ''):<8} "
            f"Gold={row.get('Gold Label', '')}"
        )
        item = {
            "id": row.get("ID"),
            "country": row.get("Country"),
            "subaxis": subaxis,
            "gold": row.get("Gold Label"),
            "story": story,
            "debate": debate_item,
        }
        if use_rot:
            item["rule"] = rule
        history.append(item)
        if len(history) >= n_turns:
            break
    return history


def build_debate_transcript(history, agent_a_name, agent_b_name, include_rule):
    blocks = []
    for idx, item in enumerate(history, start=1):
        debate_item = item["debate"]
        lines = [
            f"[Adapt story {idx}] {item.get('subaxis', '')}",
            f"Story: {item['story']}",
        ]
        if include_rule:
            lines.append(f"Rule: {item.get('rule', '')}")
        lines.extend(
            [
                f"{agent_a_name} initial: {debate_item.get(f'{agent_a_name}_initial', '')}",
                f"{agent_b_name} initial: {debate_item.get(f'{agent_b_name}_initial', '')}",
                f"{agent_a_name} feedback: {debate_item.get(f'{agent_a_name}_feedback', '')}",
                f"{agent_b_name} feedback: {debate_item.get(f'{agent_b_name}_feedback', '')}",
                f"{agent_a_name} final: {debate_item.get(f'{agent_a_name}_final', '')}",
                f"{agent_b_name} final: {debate_item.get(f'{agent_b_name}_final', '')}",
            ]
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


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


def build_chat_template_memory(args, tokenizer, model, country_key, use_rot):
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

    mode = "RoT" if use_rot else "No-RoT"
    print("\n" + "=" * 60)
    print(f"NormAd ChatTemplate {mode} interaction memory: {country_name} ({country_key})")
    print("=" * 60)
    adaptation_history = run_chat_template_adaptation_dialogue(
        tokenizer,
        model,
        base.exp.PERSONAS[country_key],
        base.exp.PERSONAS["korea"],
        country_name,
        adapt_rows,
        country_key,
        "korea",
        args.n_turns,
        use_rot,
    )
    debate_transcript = build_debate_transcript(
        adaptation_history,
        country_key,
        "korea",
        include_rule=use_rot,
    )
    reflection = build_reflection(args, tokenizer, model, country_name, debate_transcript)
    return country_name, adaptation_history, debate_transcript, reflection


def save_chat_template_interaction_artifacts(
    args,
    country_key,
    country_name,
    condition_dir,
    adaptation_history,
    debate_transcript,
    reflection,
    use_rot,
):
    output_dir = os.path.join(args.output_dir, country_key, condition_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "interaction_memory.json"), "w", encoding="utf-8") as outfile:
        json.dump(
            {
                "country_key": country_key,
                "country": country_name,
                "interaction_uses_chat_template": True,
                "interaction_uses_rule_of_thumb": use_rot,
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


def build_condition_specs(args, tokenizer, model, country_key, background_contexts):
    cp_country = base.abcd.normalize_cp_country(country_key)
    (
        country_name,
        no_rot_history,
        no_rot_transcript,
        no_rot_reflection,
    ) = build_chat_template_memory(args, tokenizer, model, country_key, use_rot=False)
    save_chat_template_interaction_artifacts(
        args,
        country_key,
        country_name,
        "chat_template_no_rot_interaction",
        no_rot_history,
        no_rot_transcript,
        no_rot_reflection,
        use_rot=False,
    )

    (
        _rot_country_name,
        rot_history,
        rot_transcript,
        rot_reflection,
    ) = build_chat_template_memory(args, tokenizer, model, country_key, use_rot=True)
    save_chat_template_interaction_artifacts(
        args,
        country_key,
        country_name,
        "chat_template_rot_interaction",
        rot_history,
        rot_transcript,
        rot_reflection,
        use_rot=True,
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
        "A_chat_template_no_rot_interaction_persona_o": base.abcd.build_interaction_korea_system_prompt(
            cp_country,
            target_norms,
            no_rot_reflection,
        ),
        "B_chat_template_no_rot_debate_only_persona_o": build_debate_only_korea_system_prompt(
            cp_country,
            target_norms,
            no_rot_transcript,
            "chat-template no-Rule-of-Thumb",
        ),
        "C_persona_o_json_dialogue": build_json_dialogue_korea_messages(
            cp_country,
            target_norms,
            background_context,
        ),
        "D_chat_template_rot_interaction_persona_o": base.abcd.build_interaction_korea_system_prompt(
            cp_country,
            target_norms,
            rot_reflection,
        ),
        "E_chat_template_rot_debate_only_persona_o": build_debate_only_korea_system_prompt(
            cp_country,
            target_norms,
            rot_transcript,
            "chat-template Rule-of-Thumb",
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
    parser.add_argument("--output_dir", default="outputs/normad_chat_template_interaction_rot_no_rot_json_dialogue_5cond_qwen25_14b")
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
