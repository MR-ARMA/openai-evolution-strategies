# File: 03_Implementation/generate_report.py
import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_metrics(csv_path: str):
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


def plot_environment_curve(env_name: str, data: dict, out_path: str):
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


def generate_all_reports(runs_root: str = "runs"):
    csv_files = sorted(glob.glob(os.path.join(runs_root, "*", "metrics.csv")))
    if not csv_files:
        print(f"No metrics.csv files found in '{runs_root}/'. Run train_es_mujoco.py first.")
        return

    report_lines = [
        "# OpenAI Evolution Strategies (ES) — Experimental Reproduction Report\n",
        "## 1. Final Performance Summary (Table 2 Comparison)\n",
        "| Environment | Generations | Total Timesteps | Initial Return (Gen 1) | Final Best Return | Training Time (min) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    milestone_sections = ["\n## 2. Evolutionary Progression Across Generations\n"]

    for csv_path in csv_files:
        env_dir = os.path.dirname(csv_path)
        env_name = os.path.basename(env_dir)
        data = load_metrics(csv_path)
        if len(data["generation"]) == 0:
            continue

        plot_path = os.path.join(env_dir, f"{env_name}_learning_curve.png")
        plot_environment_curve(env_name, data, plot_path)
        print(f"[Plot Saved] {plot_path}")

        total_gens = int(data["generation"][-1])
        total_steps = int(data["timesteps"][-1])
        init_ret = float(data["eval_ret"][0])
        best_ret = float(data["best_eval"][-1])
        total_min = float(data["wall_time"][-1]) / 60.0

        report_lines.append(
            f"| **{env_name.capitalize()}-v4** | {total_gens} | {total_steps:,} | {init_ret:.1f} | **{best_ret:.1f}** | {total_min:.1f}m |"
        )

        # Sample 5 milestone rows for behavioral progression table
        milestone_idxs = np.linspace(0, len(data["generation"]) - 1, min(5, len(data["generation"])), dtype=int)
        milestone_sections.append(f"### {env_name.capitalize()}-v4 Milestone Trajectory\n")
        milestone_sections.append("| Generation | Timesteps | Population Mean | Parent Eval Return | Best Return |")
        milestone_sections.append("| :---: | :---: | :---: | :---: | :---: |")
        for idx in milestone_idxs:
            milestone_sections.append(
                f"| Gen {int(data['generation'][idx]):04d} | {int(data['timesteps'][idx]):,} | "
                f"{data['pop_mean'][idx]:.1f} | {data['eval_ret'][idx]:.1f} | **{data['best_eval'][idx]:.1f}** |"
            )
        milestone_sections.append("")

    report_path = os.path.join(runs_root, "evolution_summary_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines + milestone_sections))

    print(f"[Report Saved] {report_path}")


if __name__ == "__main__":
    generate_all_reports()