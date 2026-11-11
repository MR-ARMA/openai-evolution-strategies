# File: 03_Implementation/generate_report.py
import os
import csv
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

from es_core import ENV_CONFIGS


# ==============================================================================
# 1. DATA LOADERS (Single-Seed & 30-Seed Aggregated Telemetry)
# ==============================================================================
def load_single_metrics(csv_path: str) -> dict:
    data = {
        "generation": [],
        "timesteps": [],
        "pop_mean": [],
        "pop_std": [],
        "pop_max": [],
        "eval_ret": [],
        "best_eval": [],
        "wall_time": [],
    }
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["generation"].append(int(row["generation"]))
            data["timesteps"].append(int(row["total_timesteps"]))
            data["pop_mean"].append(float(row["pop_mean_return"]))
            data["pop_std"].append(float(row["pop_std_return"]))
            data["pop_max"].append(float(row["pop_max_return"]))
            data["eval_ret"].append(float(row["eval_return"]))
            data["best_eval"].append(float(row["best_eval_return"]))
            data["wall_time"].append(float(row["wall_time_sec"]))
    return {k: np.array(v) for k, v in data.items()}


def load_aggregated_30seeds(csv_path: str) -> dict:
    data = {
        "generation": [],
        "num_seeds": [],
        "mean_evaluations": [],
        "mean_timesteps": [],
        "eval_mean": [],
        "eval_std": [],
        "eval_median": [],
        "eval_q25": [],
        "eval_q75": [],
        "eval_iqr": [],
        "eval_min": [],
        "eval_max": [],
        "best_mean": [],
        "best_std": [],
        "pop_mean_avg": [],
        "wall_time": [],
    }
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["generation"].append(int(row["generation"]))
            data["num_seeds"].append(int(row["num_seeds"]))
            data["mean_evaluations"].append(int(row["mean_evaluations"]))
            data["mean_timesteps"].append(int(row["mean_timesteps"]))
            data["eval_mean"].append(float(row["eval_return_mean"]))
            data["eval_std"].append(float(row["eval_return_std"]))
            data["eval_median"].append(float(row["eval_return_median"]))
            data["eval_q25"].append(float(row["eval_return_q25"]))
            data["eval_q75"].append(float(row["eval_return_q75"]))
            data["eval_iqr"].append(float(row["eval_return_iqr"]))
            data["eval_min"].append(float(row["eval_return_min"]))
            data["eval_max"].append(float(row["eval_return_max"]))
            data["best_mean"].append(float(row["best_return_mean"]))
            data["best_std"].append(float(row["best_return_std"]))
            data["pop_mean_avg"].append(float(row["pop_mean_return_avg"]))
            data["wall_time"].append(float(row["mean_wall_time_sec"]))
    return {k: np.array(v) for k, v in data.items()}


def load_final_distribution(csv_path: str) -> dict:
    dist = {
        "seed": [],
        "env_id": [],
        "algo": [],
        "ac_bins": [],
        "generations": [],
        "evaluations": [],
        "timesteps": [],
        "final_return": [],
        "best_return": [],
        "wall_time": [],
    }
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist["seed"].append(int(row["seed"]))
            dist["env_id"].append(row["env_id"])
            dist["algo"].append(row["algo"])
            dist["ac_bins"].append(row["ac_bins"])
            dist["generations"].append(int(row["generations"]))
            dist["evaluations"].append(int(row["total_evaluations"]))
            dist["timesteps"].append(int(row["total_timesteps"]))
            dist["final_return"].append(float(row["final_eval_return"]))
            dist["best_return"].append(float(row["best_eval_return"]))
            dist["wall_time"].append(float(row["wall_time_sec"]))
    return dist


# ==============================================================================
# 2. CONVERGENCE CURVES (Mean ± 1 Std Shaded Bands + Paper Target Reference)
# ==============================================================================
def plot_30seed_convergence(exp_name: str, env_id: str, agg: dict, out_path: str):
    gens = agg["generation"]
    eval_mean = agg["eval_mean"]
    eval_std = agg["eval_std"]
    eval_median = agg["eval_median"]
    best_mean = agg["best_mean"]
    best_std = agg["best_std"]
    n_seeds = int(agg["num_seeds"][-1]) if len(agg["num_seeds"]) > 0 else 30
    target_score = ENV_CONFIGS.get(env_id, {}).get("target_score", None)

    plt.figure(figsize=(9.5, 5.2), dpi=150)
    plt.plot(gens, eval_mean, label=f"Parent Eval Return (Mean over N={n_seeds} seeds)", color="#1f77b4", linewidth=2.2)
    plt.fill_between(
        gens,
        eval_mean - eval_std,
        eval_mean + eval_std,
        color="#1f77b4",
        alpha=0.22,
        label="±1 Standard Deviation Band",
    )
    plt.plot(gens, eval_median, label=f"Parent Eval Return (Median, N={n_seeds})", color="#9467bd", linestyle=":", linewidth=1.8)
    plt.plot(gens, best_mean, label=f"Best Return So Far (Mean, N={n_seeds})", color="#2ca02c", linestyle="--", linewidth=2.0)
    plt.fill_between(gens, best_mean - best_std, best_mean + best_std, color="#2ca02c", alpha=0.10)

    if target_score is not None:
        plt.axhline(
            y=target_score,
            color="#d62728",
            linestyle="-.",
            linewidth=1.6,
            label=f"Paper Baseline Target ({target_score:.0f})",
        )

    plt.title(f"OpenAI ES — {exp_name} Convergence ({env_id}, N={n_seeds} Independent Seeds)", fontsize=12.5, fontweight="bold")
    plt.xlabel("Generation (t)", fontsize=11)
    plt.ylabel("Episode Return F(θ)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_single_seed_curve(env_name: str, data: dict, out_path: str):
    gens = data["generation"]
    pop_mean = data["pop_mean"]
    pop_std = data["pop_std"]
    eval_ret = data["eval_ret"]
    best_eval = data["best_eval"]

    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(gens, pop_mean, label="Population Mean Return", color="#1f77b4", alpha=0.7, linewidth=1.5)
    plt.fill_between(gens, pop_mean - pop_std, pop_mean + pop_std, color="#1f77b4", alpha=0.15, label="Population ±1 Std")
    plt.plot(gens, eval_ret, label="Parent Policy Return (Deterministic)", color="#ff7f0e", linewidth=2.0)
    plt.plot(gens, best_eval, label="Best Parent Return So Far", color="#2ca02c", linestyle="--", linewidth=2.0)

    plt.title(f"OpenAI Evolution Strategies — {env_name.capitalize()} Learning Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Generation (t)", fontsize=11)
    plt.ylabel("Episode Return F(θ)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ==============================================================================
# 3. PUBLICATION-GRADE BOXPLOTS & ABLATION OVERLAY PLOTS
# ==============================================================================
def plot_multi_experiment_boxplots(distributions: dict, out_path: str, title: str = "Final Return Distribution Across 30 Independent Seeds"):
    if not distributions:
        return

    labels = list(distributions.keys())
    final_Arrays = [np.array(distributions[k]["best_return"], dtype=np.float64) for k in labels]

    plt.figure(figsize=(max(8.5, 2.0 * len(labels)), 5.5), dpi=150)
    bp = plt.boxplot(
        final_Arrays,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "#d62728", "markeredgecolor": "black", "markersize": 7},
        medianprops={"color": "#111827", "linewidth": 2.2},
        boxprops={"facecolor": "#93c5fd", "edgecolor": "#1e3a8a", "alpha": 0.75, "linewidth": 1.4},
        whiskerprops={"color": "#1e3a8a", "linewidth": 1.4},
        capprops={"color": "#1e3a8a", "linewidth": 1.4},
        flierprops={"marker": "o", "markerfacecolor": "#ef4444", "markersize": 5, "alpha": 0.7},
    )

    # Overlay individual seed jitter points
    rng = np.random.RandomState(42)
    for i, arr in enumerate(final_Arrays, start=1):
        jitter = rng.normal(loc=i, scale=0.06, size=len(arr))
        plt.scatter(jitter, arr, color="#1e40af", alpha=0.55, s=22, zorder=3, label="Individual Seed (N=30)" if i == 1 else None)

    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("Best Evaluation Return F(θ)", fontsize=11)
    plt.xticks(rotation=15, fontsize=10)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_ablation_overlay(env_id: str, exp_agg_map: dict, out_path: str):
    if len(exp_agg_map) < 2:
        return

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    plt.figure(figsize=(9.5, 5.2), dpi=150)

    for idx, (label, agg) in enumerate(exp_agg_map.items()):
        color = palette[idx % len(palette)]
        gens = agg["generation"]
        mean = agg["eval_mean"]
        std = agg["eval_std"]
        n_seeds = int(agg["num_seeds"][-1]) if len(agg["num_seeds"]) > 0 else 30
        plt.plot(gens, mean, label=f"{label} (Mean ± 1 Std, N={n_seeds})", color=color, linewidth=2.2)
        plt.fill_between(gens, mean - std, mean + std, color=color, alpha=0.18)

    target_score = ENV_CONFIGS.get(env_id, {}).get("target_score", None)
    if target_score is not None:
        plt.axhline(y=target_score, color="#111827", linestyle="-.", linewidth=1.5, label=f"Paper Target ({target_score:.0f})")

    plt.title(f"Controlled Ablation & Baseline Comparison on {env_id} (Equal Evaluation Budget)", fontsize=12.5, fontweight="bold")
    plt.xlabel("Generation (t)", fontsize=11)
    plt.ylabel("Deterministic Evaluation Return F(θ)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ==============================================================================
# 4. STATISTICAL HYPOTHESIS TESTS & SUMMARY TABLE GENERATION
# ==============================================================================
def compute_two_sample_pvalues(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple:
    """Computes two-sided Welch's t-test and Mann-Whitney U test p-values."""
    if scipy_stats is None or len(sample_a) < 2 or len(sample_b) < 2:
        return np.nan, np.nan
    try:
        _, p_welch = scipy_stats.ttest_ind(sample_a, sample_b, equal_var=False)
        _, p_mwu = scipy_stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        return float(p_welch), float(p_mwu)
    except Exception:
        return np.nan, np.nan


def generate_all_reports(runs_root: str = "runs"):
    if not os.path.exists(runs_root):
        print(f"Directory '{runs_root}/' does not exist.")
        return

    exp_dirs = sorted([d for d in glob.glob(os.path.join(runs_root, "*")) if os.path.isdir(d)])
    if not exp_dirs:
        print(f"No experiment subdirectories found in '{runs_root}/'.")
        return

    summary_rows_csv = []
    md_table_lines = [
        "# OpenAI Evolution Strategies (ES) — 30-Seed Statistical Benchmark Report\n",
        "## 1. Multi-Seed Statistical Performance Summary (`N=30` Independent Runs)\n",
        "| Experiment / Environment | Algo | Action Space | Seeds ($N$) | Budget (Evals / Steps) | Mean ± Std Return | Median (IQR) | Best / Worst | Paper Target | Mean Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    all_distributions = {}
    env_grouped_aggs = {}
    env_grouped_dists = {}
    milestone_sections = ["\n## 2. Per-Generation Convergence Milestones\n"]

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        agg_csv = os.path.join(exp_dir, "aggregated_30seeds_summary.csv")
        dist_csv = os.path.join(exp_dir, "final_30seeds_distribution.csv")
        single_csv = os.path.join(exp_dir, "metrics.csv")
        meta_json = os.path.join(exp_dir, "benchmark_metadata.json")

        meta = {}
        if os.path.exists(meta_json):
            with open(meta_json, "r") as f:
                meta = json.load(f)

        if os.path.exists(agg_csv) and os.path.exists(dist_csv):
            agg = load_aggregated_30seeds(agg_csv)
            dist = load_final_distribution(dist_csv)
            if len(agg["generation"]) == 0 or len(dist["seed"]) == 0:
                continue

            env_id = meta.get("env_id", dist["env_id"][0])
            algo = meta.get("algo", dist["algo"][0])
            ac_bins = meta.get("ac_bins", dist["ac_bins"][0])
            ac_label = f"{ac_bins} Bins" if (ac_bins not in (None, "continuous", "None", 0)) else "Continuous (tanh)"
            paper_target = ENV_CONFIGS.get(env_id, {}).get("target_score", 0.0)

            # Save 30-seed convergence curve ( both as <exp>_learning_curve.png and convergence_30seeds.png )
            curve_path_1 = os.path.join(exp_dir, f"{exp_name}_learning_curve.png")
            curve_path_2 = os.path.join(exp_dir, "convergence_30seeds.png")
            plot_30seed_convergence(exp_name, env_id, agg, curve_path_1)
            plot_30seed_convergence(exp_name, env_id, agg, curve_path_2)
            print(f"[Plot Saved] {curve_path_1}")

            # Save single-experiment boxplot
            single_box_path = os.path.join(exp_dir, "boxplot_final_returns.png")
            plot_multi_experiment_boxplots({exp_name: dist}, single_box_path, title=f"{exp_name} ({env_id}) — 30-Seed Return Boxplot")

            all_distributions[exp_name] = dist
            env_grouped_aggs.setdefault(env_id, {})[exp_name] = agg
            env_grouped_dists.setdefault(env_id, {})[exp_name] = dist

            best_rets = np.array(dist["best_return"], dtype=np.float64)
            final_rets = np.array(dist["final_return"], dtype=np.float64)
            n_seeds = len(best_rets)
            mean_ret = float(np.mean(best_rets))
            std_ret = float(np.std(best_rets))
            q25, median_ret, q75 = np.percentile(best_rets, [25, 50, 75])
            iqr_ret = float(q75 - q25)
            max_ret = float(np.max(best_rets))
            min_ret = float(np.min(best_rets))
            mean_evals = int(np.mean(dist["evaluations"]))
            mean_steps = int(np.mean(dist["timesteps"]))
            mean_time = float(np.mean(dist["wall_time"]))

            md_table_lines.append(
                f"| **{exp_name}** (`{env_id}`) | `{algo}` | {ac_label} | {n_seeds} | "
                f"{mean_evals:,} / {mean_steps:,} | **{mean_ret:.1f} ± {std_ret:.1f}** | "
                f"{median_ret:.1f} ({iqr_ret:.1f}) | {max_ret:.1f} / {min_ret:.1f} | "
                f"~{paper_target:.0f} | {mean_time:.1f}s |"
            )

            summary_rows_csv.append({
                "experiment": exp_name,
                "env_id": env_id,
                "algo": algo,
                "action_space": ac_label,
                "num_seeds": n_seeds,
                "mean_evaluations": mean_evals,
                "mean_timesteps": mean_steps,
                "best_return_mean": f"{mean_ret:.2f}",
                "best_return_std": f"{std_ret:.2f}",
                "best_return_median": f"{median_ret:.2f}",
                "best_return_iqr": f"{iqr_ret:.2f}",
                "best_return_max": f"{max_ret:.2f}",
                "best_return_min": f"{min_ret:.2f}",
                "final_gen_return_mean": f"{np.mean(final_rets):.2f}",
                "final_gen_return_std": f"{np.std(final_rets):.2f}",
                "paper_target": f"{paper_target:.1f}",
                "mean_wall_time_sec": f"{mean_time:.2f}",
            })

            milestone_idxs = np.linspace(0, len(agg["generation"]) - 1, min(5, len(agg["generation"])), dtype=int)
            milestone_sections.append(f"### {exp_name} (`{env_id}`, N={n_seeds} Seeds)\n")
            milestone_sections.append("| Generation | Mean Timesteps | Eval Return (Mean ± Std) | Eval Return (Median / IQR) | Best Return (Mean ± Std) |")
            milestone_sections.append("| :---: | :---: | :---: | :---: | :---: |")
            for idx in milestone_idxs:
                milestone_sections.append(
                    f"| Gen {int(agg['generation'][idx]):04d} | {int(agg['mean_timesteps'][idx]):,} | "
                    f"{agg['eval_mean'][idx]:.1f} ± {agg['eval_std'][idx]:.1f} | "
                    f"{agg['eval_median'][idx]:.1f} ({agg['eval_iqr'][idx]:.1f}) | "
                    f"**{agg['best_mean'][idx]:.1f} ± {agg['best_std'][idx]:.1f}** |"
                )
            milestone_sections.append("")

        elif os.path.exists(single_csv):
            data = load_single_metrics(single_csv)
            if len(data["generation"]) == 0:
                continue
            plot_path = os.path.join(exp_dir, f"{exp_name}_learning_curve.png")
            plot_single_seed_curve(exp_name, data, plot_path)
            print(f"[Single-Seed Plot Saved] {plot_path}")

    # Combined multi-environment boxplot
    if all_distributions:
        global_box_path = os.path.join(runs_root, "boxplot_final_returns.png")
        plot_multi_experiment_boxplots(all_distributions, global_box_path)
        print(f"[Boxplot Saved] {global_box_path}")

    # Ablation overlays & statistical significance tables per environment
    ablation_md_lines = ["\n## 3. Controlled Ablation & Baseline Hypothesis Tests\n"]
    has_ablation = False
    for env_id, group_aggs in env_grouped_aggs.items():
        if len(group_aggs) >= 2:
            has_ablation = True
            env_short = env_id.split("-")[0].lower()
            overlay_path = os.path.join(runs_root, f"ablation_comparison_{env_short}.png")
            box_ablation_path = os.path.join(runs_root, f"boxplot_ablation_{env_short}.png")
            plot_ablation_overlay(env_id, group_aggs, overlay_path)
            plot_multi_experiment_boxplots(
                env_grouped_dists[env_id],
                box_ablation_path,
                title=f"{env_id} Controlled Comparison — 30-Seed Return Distribution",
            )
            print(f"[Ablation Plots Saved] {overlay_path} & {box_ablation_path}")

            exp_keys = list(env_grouped_dists[env_id].keys())
            ref_key = exp_keys[0]
            ref_sample = np.array(env_grouped_dists[env_id][ref_key]["best_return"], dtype=np.float64)

            ablation_md_lines.append(f"### Pairwise Comparisons on `{env_id}` (Reference: **{ref_key}**)\n")
            ablation_md_lines.append("| Comparison Pair | Reference Mean ± Std | Variant Mean ± Std | Welch's $t$-test ($p$-value) | Mann-Whitney $U$ ($p$-value) |")
            ablation_md_lines.append("| :--- | :---: | :---: | :---: | :---: |")
            for other_key in exp_keys[1:]:
                other_sample = np.array(env_grouped_dists[env_id][other_key]["best_return"], dtype=np.float64)
                p_welch, p_mwu = compute_two_sample_pvalues(ref_sample, other_sample)
                ablation_md_lines.append(
                    f"| **{ref_key}** vs. **{other_key}** | "
                    f"{np.mean(ref_sample):.1f} ± {np.std(ref_sample):.1f} | "
                    f"{np.mean(other_sample):.1f} ± {np.std(other_sample):.1f} | "
                    f"`{p_welch:.4e}` | `{p_mwu:.4e}` |"
                )
            ablation_md_lines.append("")

    # Save CSV summary table
    if summary_rows_csv:
        csv_table_path = os.path.join(runs_root, "statistical_benchmark_table.csv")
        with open(csv_table_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows_csv[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows_csv)
        print(f"[CSV Table Saved] {csv_table_path}")

    # Save Markdown reports
    report_sections = md_table_lines + (ablation_md_lines if has_ablation else []) + milestone_sections
    for report_filename in ("statistical_benchmark_report.md", "evolution_summary_report.md"):
        report_path = os.path.join(runs_root, report_filename)
        with open(report_path, "w") as f:
            f.write("\n".join(report_sections))
        print(f"[Report Saved] {report_path}")


if __name__ == "__main__":
    generate_all_reports()