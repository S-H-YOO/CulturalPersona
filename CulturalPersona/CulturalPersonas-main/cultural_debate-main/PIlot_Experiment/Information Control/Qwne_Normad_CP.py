'''
==========================================================================================================================================
NormAd 기반 문화 간 debate/interaction을 만든 뒤, 그 결과가 CulturalPersonas 성격/문화 평가에서 얼마나 target country에 가까워지는지 6개 조건으로 비교
==========================================================================================================================================

'''

import argparse
import datetime
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

import Qwen_ChatTemplate as chat_template
import Qwen_ChatTemplate_OEG_MCS as eval_utils
import Qwen_Normad_Debate as normad_debate
import qwen_no_rot_cp_probe_after_interaction as abcd
import qwen_no_rot_oeg_mcs_after_interaction as oeg_mcs
import qwen_no_rot_oeg_mcs_after_interaction_norm_control as norm_control
import qwen_persona_interaction as exp


CP_ROOT = "/scratch/lami2026/personal/sanghoon_2026/CulturalPersonas-main/CulturalPersona/CulturalPersonas-main"


def build_debate_only_system_prompt(target_country, debate_transcript, interaction_type):
    return f"""You are a helpful assistant.
You previously observed a {interaction_type} NormAd cross-cultural debate with a
{target_country} cultural agent.

[Raw NormAd debate transcript]
{debate_transcript}

Use only the raw debate transcript as context when answering later Cultural
Persona or NormAd questions. Do not assume any separate self-reflection, persona,
JSON background, or additional memory."""


def build_debate_reflection_system_prompt(
    target_country,
    debate_transcript,
    reflection,
    interaction_type,
):
    return f"""You are a helpful assistant.
You previously observed a {interaction_type} NormAd cross-cultural debate with a
{target_country} cultural agent, and then received a self-reflection summary
derived from that debate.

[Raw NormAd debate transcript]
{debate_transcript}

[Self-reflection summary]
{reflection}

Use the debate transcript and self-reflection as context when answering later
Cultural Persona or NormAd questions. Do not assume any persona, JSON background, or
additional memory."""


def build_korea_json_no_debate_messages(target_country, background_context):
    system_prompt = """You are a 28-year-old male.
Born and raised in South Korea: Yes
Current residence: South Korea

You are deeply familiar with everyday Korean social norms and expectations."""
    return {
        "kind": "messages",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Before the test, I will give you JSON-style cultural background about {target_country}. "
                    "Read it as contextual background for later judgments. This is not a debate transcript, "
                    "not a self-reflection memory, and not an interaction history.\n\n"
                    f"[JSON cultural etiquette background for {target_country}]\n{background_context}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"Understood. I will use the provided JSON cultural background about {target_country} "
                    "as context for later Cultural Persona questions, without assuming any prior debate, "
                    "self-reflection, or interaction memory."
                ),
            },
        ],
    }


def build_condition_specs(args, tokenizer, model, country_key, background_contexts):
    cp_country = abcd.normalize_cp_country(country_key)
    use_rot = args.use_rot
    interaction_type = "chat-template Rule-of-Thumb" if use_rot else "chat-template no-Rule-of-Thumb"

    (
        country_name,
        adaptation_history,
        debate_transcript,
        reflection,
    ) = chat_template.build_chat_template_memory(args, tokenizer, model, country_key, use_rot=use_rot)

    artifact_dir = "normad_chat_template_rot_interaction" if use_rot else "normad_chat_template_no_rot_interaction"
    chat_template.save_chat_template_interaction_artifacts(
        args,
        country_key,
        country_name,
        artifact_dir,
        adaptation_history,
        debate_transcript,
        reflection,
        use_rot=use_rot,
    )

    background_context = chat_template.json_background.get_wikipedia_context(
        background_contexts,
        country_key,
        country_name,
        cp_country,
        args.background_max_chars,
    )

    condition_specs = {
        "A_debate_only": build_debate_only_system_prompt(
            cp_country,
            debate_transcript,
            interaction_type,
        ),
        "B_debate_self_reflection": build_debate_reflection_system_prompt(
            cp_country,
            debate_transcript,
            reflection,
            interaction_type,
        ),
        "C_no_persona_no_debate": exp.PERSONAS["none"],
        "D_hybrid_persona_no_debate": exp.HYBRID_PERSONAS[country_key],
        "E_korea_persona_no_debate": exp.PERSONAS["korea"],
        "F_korea_persona_json_no_debate": build_korea_json_no_debate_messages(
            cp_country,
            background_context,
        ),
    }
    return country_name, cp_country, condition_specs


def run_one_country(args, tokenizer, model, embedder, gt, payload, country_key, background_contexts):
    country_name, cp_country, condition_specs = build_condition_specs(
        args,
        tokenizer,
        model,
        country_key,
        background_contexts,
    )
    country_code = abcd.country_code_for_gt(cp_country)
    output_dir = os.path.join(args.output_dir, country_key)
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_normad:
        normad_debate.run_normad_condition_evaluation(
            args,
            tokenizer,
            model,
            country_key,
            country_name,
            condition_specs,
        )

    metric_summaries = []
    for eval_mode in args.eval_modes:
        mode_dir = os.path.join(output_dir, eval_mode)
        os.makedirs(mode_dir, exist_ok=True)
        for condition, condition_spec in condition_specs.items():
            if eval_mode == "oeg":
                raw = eval_utils.run_oeg_condition(
                    tokenizer,
                    model,
                    embedder,
                    condition,
                    condition_spec,
                    payload,
                )
                dist = oeg_mcs.get_samples_oeg(
                    raw,
                    payload["traits"],
                    args.test_type,
                    samples=len(gt[gt.country == country_code]),
                )
            else:
                raw = eval_utils.run_mcs_condition(
                    tokenizer,
                    model,
                    condition,
                    condition_spec,
                    payload,
                )
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
            metrics = eval_utils.add_wasserstein_metrics(metrics, gt, dist, country_code)
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
            mean_wasserstein_distance=("wasserstein_distance", "mean"),
            mean_gt=("gt_mean", "mean"),
            mean_model=("model_mean", "mean"),
        )
    )
    summary.to_csv(os.path.join(output_dir, "oeg_mcs_metrics_summary.csv"), index=False)
    return all_metrics, summary


def condition_label(condition):
    labels = {
        "A_debate_only": "A Debate\nOnly",
        "B_debate_self_reflection": "B Debate\n+ Reflect",
        "C_no_persona_no_debate": "C No Persona\nNo Debate",
        "D_hybrid_persona_no_debate": "D Hybrid\nNo Debate",
        "E_korea_persona_no_debate": "E Korea\nNo Debate",
        "F_korea_persona_json_no_debate": "F Korea+JSON\nNo Debate",
    }
    return labels.get(condition, condition)


def condition_order(df):
    wanted = [
        "A_debate_only",
        "B_debate_self_reflection",
        "C_no_persona_no_debate",
        "D_hybrid_persona_no_debate",
        "E_korea_persona_no_debate",
        "F_korea_persona_json_no_debate",
    ]
    seen = set(df["condition"])
    return [condition for condition in wanted if condition in seen] + sorted(seen - set(wanted))


def write_cp_metric_plots(metrics, summary, output_dir):
    plot_dir = os.path.join(output_dir, "cp_eval_plots")
    os.makedirs(plot_dir, exist_ok=True)

    conditions = condition_order(metrics)
    traits = [trait for trait in ["O", "C", "E", "A", "N"] if trait in set(metrics["trait"])]
    eval_modes = [mode for mode in ["oeg", "mcs"] if mode in set(metrics["eval_mode"])]
    colors = {"oeg": "#3b82f6", "mcs": "#f97316"}
    metric_specs = [
        ("mean_kl_divergence", "KL Divergence"),
        ("mean_ks_stat", "KS Statistic"),
        ("mean_wasserstein_distance", "Wasserstein Distance"),
    ]
    trait_metric_specs = [
        ("kl_divergence", "KL Divergence"),
        ("ks_stat", "KS Statistic"),
        ("wasserstein_distance", "Wasserstein Distance"),
    ]
    written = []

    macro = (
        summary[summary["country_key"] != "ALL"]
        .groupby(["eval_mode", "condition"], as_index=False)
        .agg(
            mean_kl_divergence=("mean_kl_divergence", "mean"),
            mean_ks_stat=("mean_ks_stat", "mean"),
            mean_wasserstein_distance=("mean_wasserstein_distance", "mean"),
        )
    )
    x = np.arange(len(conditions))
    width = 0.34 if len(eval_modes) > 1 else 0.55
    for metric, title in metric_specs:
        if metric not in macro:
            continue
        fig, ax = plt.subplots(figsize=(13, 6), dpi=180)
        for idx, mode in enumerate(eval_modes):
            values = []
            for condition in conditions:
                row = macro[(macro["eval_mode"] == mode) & (macro["condition"] == condition)]
                values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            offset = (idx - (len(eval_modes) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=mode.upper(),
                color=colors.get(mode, "#22c55e"),
            )
            ax.bar_label(
                bars,
                labels=[f"{value:.3f}" if not np.isnan(value) else "" for value in values],
                fontsize=8,
                padding=2,
            )
        ax.set_title(f"Cultural Persona Mean {title} by Condition")
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([condition_label(c) for c in conditions], fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"macro_{metric}_by_condition.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    for metric, title in metric_specs:
        if metric not in summary:
            continue
        for mode in eval_modes:
            sub = summary[summary["eval_mode"] == mode]
            matrix = sub.pivot_table(
                index="target_country",
                columns="condition",
                values=metric,
                aggfunc="mean",
            )
            matrix = matrix.reindex(columns=conditions)
            fig, ax = plt.subplots(figsize=(13, 6), dpi=180)
            im = ax.imshow(matrix.values, aspect="auto", cmap="viridis_r")
            ax.set_title(f"{mode.upper()} Mean {title} by Country")
            ax.set_xticks(np.arange(len(conditions)))
            ax.set_xticklabels([condition_label(c).replace("\n", " ") for c in conditions], rotation=25, ha="right")
            ax.set_yticks(np.arange(len(matrix.index)))
            ax.set_yticklabels(matrix.index)
            threshold = np.nanmean(matrix.values)
            for row_idx in range(matrix.shape[0]):
                for col_idx in range(matrix.shape[1]):
                    value = matrix.values[row_idx, col_idx]
                    if not np.isnan(value):
                        ax.text(
                            col_idx,
                            row_idx,
                            f"{value:.3f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if value > threshold else "black",
                        )
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(title)
            fig.tight_layout()
            path = os.path.join(plot_dir, f"{mode}_country_{metric}_heatmap.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    for metric, title in trait_metric_specs:
        if metric not in metrics:
            continue
        for mode in eval_modes:
            sub = metrics[metrics["eval_mode"] == mode]
            matrix = sub.pivot_table(index="trait", columns="condition", values=metric, aggfunc="mean")
            matrix = matrix.reindex(index=traits, columns=conditions)
            fig, ax = plt.subplots(figsize=(13, 5), dpi=180)
            im = ax.imshow(matrix.values, aspect="auto", cmap="viridis_r")
            ax.set_title(f"{mode.upper()} Mean {title} by Trait")
            ax.set_xticks(np.arange(len(conditions)))
            ax.set_xticklabels([condition_label(c).replace("\n", " ") for c in conditions], rotation=25, ha="right")
            ax.set_yticks(np.arange(len(traits)))
            ax.set_yticklabels(traits)
            threshold = np.nanmean(matrix.values)
            for row_idx in range(matrix.shape[0]):
                for col_idx in range(matrix.shape[1]):
                    value = matrix.values[row_idx, col_idx]
                    if not np.isnan(value):
                        ax.text(
                            col_idx,
                            row_idx,
                            f"{value:.3f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if value > threshold else "black",
                        )
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(title)
            fig.tight_layout()
            path = os.path.join(plot_dir, f"{mode}_trait_{metric}_heatmap.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    for mode in eval_modes:
        sub = metrics[metrics["eval_mode"] == mode]
        grouped = sub.groupby(["condition", "trait"], as_index=False)[["gt_mean", "model_mean"]].mean()
        fig, axes = plt.subplots(1, len(conditions), figsize=(18, 5), dpi=180, sharey=True)
        if len(conditions) == 1:
            axes = [axes]
        x = np.arange(len(traits))
        width = 0.36
        for ax, condition in zip(axes, conditions):
            condition_rows = grouped[grouped["condition"] == condition].set_index("trait").reindex(traits)
            ax.bar(x - width / 2, condition_rows["gt_mean"], width=width, label="GT", color="#64748b")
            ax.bar(
                x + width / 2,
                condition_rows["model_mean"],
                width=width,
                label="Model",
                color=colors.get(mode, "#22c55e"),
            )
            ax.set_title(condition_label(condition).replace("\n", " "), fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(traits)
            ax.set_ylim(1, 5)
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Mean Big Five Score (1-5)")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"{mode.upper()} GT vs Model Mean by Trait", y=1.04)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"{mode}_gt_vs_model_mean_by_trait.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normad_input_path", default="data/normad.jsonl")
    parser.add_argument("--output_dir", default="outputs/cp_oeg_mcs_normad_6cond_qwen25_14b")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", exp.DEFAULT_CACHE_DIR))
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=[
            "india",
            "brazil",
            "saudi_arabia",
            "south_africa",
            "united_states_of_america",
            "japan",
        ],
    )
    parser.add_argument("--limit_per_country", type=int, default=None)
    parser.add_argument("--adapt_per_label", type=int, default=2)
    parser.add_argument("--test_per_label", type=int, default=None)
    parser.add_argument("--n_turns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reflection_tokens", type=int, default=700)
    parser.add_argument("--reflection_temp", type=float, default=0.3)
    parser.add_argument("--use_rot", action="store_true")
    parser.add_argument("--cp_root", default=CP_ROOT)
    parser.add_argument("--background_context_path", default="data/country_etiquette_backgrounds_detailed.json")
    parser.add_argument("--background_max_chars", type=int, default=6000)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--ground_truth", default=None)
    parser.add_argument("--test_type", choices=["standard", "trait", "cp", "big5chat"], default="trait")
    parser.add_argument("--max_questions", type=int, default=20)
    parser.add_argument("--eval_modes", nargs="+", choices=["oeg", "mcs"], default=["oeg", "mcs"])
    parser.add_argument("--skip_normad", action="store_true")
    args = parser.parse_args()

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

    aggregate_metrics = pd.concat(metrics, ignore_index=True)
    aggregate_summary = pd.concat(summaries, ignore_index=True)
    aggregate_metrics.to_csv(
        os.path.join(args.output_dir, "aggregate_oeg_mcs_metrics_by_trait.csv"),
        index=False,
    )
    aggregate_summary.to_csv(
        os.path.join(args.output_dir, "aggregate_oeg_mcs_metrics_summary.csv"),
        index=False,
    )
    cp_plots = write_cp_metric_plots(aggregate_metrics, aggregate_summary, args.output_dir)
    if not args.skip_normad:
        norm_control.update_aggregate_normad_eval_artifacts(args)
    print("CP plots:")
    for path in cp_plots:
        print(f"  {path}")
    finished_at = datetime.datetime.now()
    print(f"Started:  {started_at}")
    print(f"Finished: {finished_at}")
    print(f"Elapsed:  {finished_at - started_at}")
    print(f"Outputs:  {args.output_dir}")


if __name__ == "__main__":
    main()
