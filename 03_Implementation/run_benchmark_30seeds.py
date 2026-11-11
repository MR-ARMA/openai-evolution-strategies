# File: 03_Implementation/run_benchmark_30seeds.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import time
import json
import pickle
import argparse
import numpy as np
import gymnasium as gym
from multiprocessing import Pool, RawArray

from es_core import BENCHMARK_SEEDS, ENV_CONFIGS, MujocoMLPPolicy
from train_es_mujoco import init_worker, run_single_seed_training
from evaluate_and_record import evaluate_and_record
from generate_report import generate_all_reports


def load_seed_csv(csv_path: str) -> list:
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "seed": int(r["seed"]),
                "generation": int(r["generation"]),
                "total_evaluations": int(r["total_evaluations"]),
                "total_timesteps": int(r["total_timesteps"]),
                "pop_mean_return": float(r["pop_mean_return"]),
                "pop_std_return": float(r["pop_std_return"]),
                "pop_min_return": float(r["pop_min_return"]),
                "pop_max_return": float(r["pop_max_return"]),
                "eval_return": float(r["eval_return"]),
                "best_eval_return": float(r["best_eval_return"]),
                "eval_steps": int(r["eval_steps"]),
                "wall_time_sec": float(r["wall_time_sec"]),
            })
    return rows


def aggregate_experiment_telemetry(exp_dir: str, env_id: str, algo: str, ac_bins, seeds: list):
    raw_dir = os.path.join(exp_dir, "raw_seeds")
    seed_histories = {}
    for s in seeds:
        p = os.path.join(raw_dir, f"seed_{s}_metrics.csv")
        if os.path.exists(p):
            rows = load_seed_csv(p)
            if rows:
                seed_histories[s] = rows

    if not seed_histories:
        return

    num_seeds = len(seed_histories)
    min_gens = min(len(h) for h in seed_histories.values())

    agg_csv_path = os.path.join(exp_dir, "aggregated_30seeds_summary.csv")
    compat_csv_path = os.path.join(exp_dir, "metrics.csv")
    final_dist_csv_path = os.path.join(exp_dir, "final_30seeds_distribution.csv")

    with open(agg_csv_path, "w", newline="") as agg_f, open(compat_csv_path, "w", newline="") as comp_f:
        agg_writer = csv.writer(agg_f)
        comp_writer = csv.writer(comp_f)

        agg_writer.writerow([
            "generation",
            "num_seeds",
            "mean_evaluations",
            "mean_timesteps",
            "eval_return_mean",
            "eval_return_std",
            "eval_return_median",
            "eval_return_q25",
            "eval_return_q75",
            "eval_return_iqr",
            "eval_return_min",
            "eval_return_max",
            "best_return_mean",
            "best_return_std",
            "pop_mean_return_avg",
            "pop_std_return_avg",
            "mean_wall_time_sec",
        ])

        comp_writer.writerow([
            "generation",
            "total_timesteps",
            "pop_mean_return",
            "pop_std_return",
            "pop_max_return",
            "eval_return",
            "best_eval_return",
            "eval_steps",
            "wall_time_sec",
        ])

        for g_idx in range(min_gens):
            gen_num = g_idx + 1
            eval_rets = np.array([h[g_idx]["eval_return"] for h in seed_histories.values()], dtype=np.float64)
            best_rets = np.array([h[g_idx]["best_eval_return"] for h in seed_histories.values()], dtype=np.float64)
            pop_means = np.array([h[g_idx]["pop_mean_return"] for h in seed_histories.values()], dtype=np.float64)
            pop_stds = np.array([h[g_idx]["pop_std_return"] for h in seed_histories.values()], dtype=np.float64)
            pop_maxs = np.array([h[g_idx]["pop_max_return"] for h in seed_histories.values()], dtype=np.float64)
            timesteps = np.array([h[g_idx]["total_timesteps"] for h in seed_histories.values()], dtype=np.float64)
            evals = np.array([h[g_idx]["total_evaluations"] for h in seed_histories.values()], dtype=np.float64)
            steps_arr = np.array([h[g_idx]["eval_steps"] for h in seed_histories.values()], dtype=np.float64)
            wall_arr = np.array([h[g_idx]["wall_time_sec"] for h in seed_histories.values()], dtype=np.float64)

            q25, median, q75 = np.percentile(eval_rets, [25, 50, 75])
            iqr = q75 - q25

            agg_writer.writerow([
                gen_num,
                num_seeds,
                int(np.mean(evals)),
                int(np.mean(timesteps)),
                f"{np.mean(eval_rets):.2f}",
                f"{np.std(eval_rets):.2f}",
                f"{median:.2f}",
                f"{q25:.2f}",
                f"{q75:.2f}",
                f"{iqr:.2f}",
                f"{np.min(eval_rets):.2f}",
                f"{np.max(eval_rets):.2f}",
                f"{np.mean(best_rets):.2f}",
                f"{np.std(best_rets):.2f}",
                f"{np.mean(pop_means):.2f}",
                f"{np.mean(pop_stds):.2f}",
                f"{np.mean(wall_arr):.2f}",
            ])

            comp_writer.writerow([
                gen_num,
                int(np.mean(timesteps)),
                f"{np.mean(pop_means):.2f}",
                f"{np.std(eval_rets):.2f}",
                f"{np.mean(pop_maxs):.2f}",
                f"{np.mean(eval_rets):.2f}",
                f"{np.mean(best_rets):.2f}",
                int(np.mean(steps_arr)),
                f"{np.mean(wall_arr):.2f}",
            ])

    with open(final_dist_csv_path, "w", newline="") as dist_f:
        dist_writer = csv.writer(dist_f)
        dist_writer.writerow([
            "seed",
            "env_id",
            "algo",
            "ac_bins",
            "generations",
            "total_evaluations",
            "total_timesteps",
            "final_eval_return",
            "best_eval_return",
            "wall_time_sec",
        ])
        for s, h in sorted(seed_histories.items()):
            last = h[-1]
            dist_writer.writerow([
                s,
                env_id,
                algo,
                ac_bins if ac_bins is not None else "continuous",
                last["generation"],
                last["total_evaluations"],
                last["total_timesteps"],
                f"{last['eval_return']:.2f}",
                f"{last['best_eval_return']:.2f}",
                f"{last['wall_time_sec']:.2f}",
            ])


def run_30seed_experiment(
    env_id: str,
    exp_name: str = None,
    seeds: list = None,
    generations: int = 150,
    max_timesteps: int = None,
    max_evaluations: int = None,
    pop_size: int = None,
    sigma: float = None,
    lr: float = None,
    l2coeff: float = None,
    ac_bins_override: int = -1,
    algo: str = "es",
    workers: int = None,
    noise_table_size: int = 30_000_000,
    resume: bool = True,
):
    if seeds is None:
        seeds = BENCHMARK_SEEDS
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 16

    cfg = ENV_CONFIGS[env_id]
    pop_size = pop_size if pop_size is not None else cfg["pop_size"]
    sigma = sigma if sigma is not None else cfg["sigma"]
    lr = lr if lr is not None else cfg["lr"]
    l2coeff = l2coeff if l2coeff is not None else cfg["l2coeff"]

    if ac_bins_override == -1:
        ac_bins = cfg["ac_bins"]
    elif ac_bins_override == 0:
        ac_bins = None
    else:
        ac_bins = int(ac_bins_override)

    env_short = env_id.split("-")[0].lower()
    if exp_name is None:
        exp_name = env_short
    exp_dir = os.path.join("runs", exp_name)
    raw_seeds_dir = os.path.join(exp_dir, "raw_seeds")
    os.makedirs(raw_seeds_dir, exist_ok=True)

    dummy_env = gym.make(env_id)
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]
    dummy_env.close()

    dummy_policy = MujocoMLPPolicy(obs_dim, act_dim, hidden_dims=(64, 64), ac_bins=ac_bins)
    num_params = dummy_policy.num_params

    meta_path = os.path.join(exp_dir, "benchmark_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "exp_name": exp_name,
                "env_id": env_id,
                "algo": algo,
                "ac_bins": ac_bins,
                "num_seeds": len(seeds),
                "seeds": seeds,
                "generations": generations,
                "max_timesteps": max_timesteps,
                "max_evaluations": max_evaluations,
                "pop_size": pop_size,
                "sigma": sigma,
                "lr": lr,
                "l2coeff": l2coeff,
                "workers": workers,
                "target_score": cfg["target_score"],
                "paper_es_score": cfg["paper_es_score"],
            },
            f,
            indent=2,
        )

    print("\n" + "#" * 92)
    print(f"STARTING 30-SEED BENCHMARK: {exp_name} ({env_id} | Algo={algo} | Bins={ac_bins})")
    print(f"Seeds ({len(seeds)}): {seeds[0]}..{seeds[-1]} | Workers: {workers} Logical Cores | Zero-Copy Shared Memory: ON")
    print("#" * 92)

    shared_buffers = {
        "noise": RawArray("f", noise_table_size),
        "theta": RawArray("f", num_params),
        "ob_mean": RawArray("f", obs_dim),
        "ob_std": RawArray("f", obs_dim),
    }
    noise_np = np.frombuffer(shared_buffers["noise"], dtype=np.float32)

    global_best_return = -np.inf
    best_ckpt_path = os.path.join(exp_dir, f"{env_short}_best.pkl")
    suite_t0 = time.time()

    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(
            shared_buffers["noise"],
            shared_buffers["theta"],
            shared_buffers["ob_mean"],
            shared_buffers["ob_std"],
            env_id,
            obs_dim,
            act_dim,
            ac_bins,
        ),
    ) as pool:

        for idx, s in enumerate(seeds, start=1):
                seed_csv_path = os.path.join(raw_seeds_dir, f"seed_{s}_metrics.csv")
                video_dir = os.path.join(exp_dir, "videos")
                need_milestone_videos = (idx == 1) and (not os.path.exists(video_dir) or len(os.listdir(video_dir)) < 5)

                if resume and os.path.exists(seed_csv_path) and not need_milestone_videos:
                    existing_rows = load_seed_csv(seed_csv_path)
                    if len(existing_rows) >= generations or (
                        max_evaluations is not None and existing_rows and existing_rows[-1]["total_evaluations"] >= max_evaluations
                    ) or (
                        max_timesteps is not None and existing_rows and existing_rows[-1]["total_timesteps"] >= max_timesteps
                    ):
                        final_ret = existing_rows[-1]["eval_return"]
                        best_ret = existing_rows[-1]["best_eval_return"]
                        print(
                            f"[{idx:02d}/{len(seeds):02d}] Seed {s} — SKIPPED (already completed: "
                            f"Final={final_ret:.1f}, Best={best_ret:.1f})"
                        )
                        continue

                noise_np[:] = np.random.RandomState(s).randn(noise_table_size).astype(np.float32)

                # Record 5 milestone progression videos ONLY for the first seed (idx == 1)
                record_this_seed = (idx == 1)

                res = run_single_seed_training(
                    env_id=env_id,
                    seed=s,
                    generations=generations,
                    max_timesteps=max_timesteps,
                    max_evaluations=max_evaluations,
                    pop_size=pop_size,
                    sigma=sigma,
                    lr=lr,
                    l2coeff=l2coeff,
                    ac_bins_override=ac_bins_override,
                    algo=algo,
                    workers=workers,
                    num_videos=5 if record_this_seed else 0,
                    no_video=not record_this_seed,
                    noise_table_size=noise_table_size,
                    run_dir=exp_dir,
                    metrics_csv_path=seed_csv_path,
                    save_checkpoints=False,
                    external_pool=pool,
                    external_shared_buffers=shared_buffers,
                    verbose=record_this_seed,
                )

                if res["best_eval_return"] > global_best_return and res["best_ckpt_payload"] is not None:
                    global_best_return = res["best_eval_return"]
                    with open(best_ckpt_path, "wb") as f:
                        pickle.dump(res["best_ckpt_payload"], f)

                print(
                    f"[{idx:02d}/{len(seeds):02d}] Seed {s} Complete | "
                    f"Gens: {res['total_generations']:3d} | Evals: {res['total_evaluations']:6d} | "
                    f"Steps: {res['total_timesteps']:8,d} | FinalRet: {res['final_eval_return']:7.1f} | "
                    f"BestRet: {res['best_eval_return']:7.1f} | Time: {res['wall_time_sec']:.1f}s"
                )

                aggregate_experiment_telemetry(exp_dir, env_id, algo, ac_bins, seeds)

        aggregate_experiment_telemetry(exp_dir, env_id, algo, ac_bins, seeds)

        # Generate the final showcase video and 5 PNG snapshots from the best checkpoint
        if os.path.exists(best_ckpt_path):
            evaluate_and_record(checkpoint_path=best_ckpt_path, env_id=env_id, num_episodes=3)

        total_min = (time.time() - suite_t0) / 60.0
        print(f"--> Finished 30-seed experiment '{exp_name}' in {total_min:.2f} minutes. Aggregated CSVs saved in {exp_dir}/")

def main():
    parser = argparse.ArgumentParser(description="30-Seed Statistical Benchmark & Ablation Runner for OpenAI ES")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["Swimmer-v4", "HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"],
        choices=list(ENV_CONFIGS.keys()),
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="standard",
        choices=["standard", "ablation", "both"],
    )
    parser.add_argument("--num_seeds", type=int, default=30)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--max_evaluations", type=int, default=9600)
    parser.add_argument("--max_timesteps", type=int, default=None)
    parser.add_argument("--pop_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 16, help="Defaults to all logical CPU cores (16 on i5-12500H)")
    parser.add_argument("--noise_table_size", type=int, default=10_000_000)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    seeds = BENCHMARK_SEEDS[: args.num_seeds]
    resume = not args.no_resume

    if args.suite in ("standard", "both"):
        for env_id in args.envs:
            env_short = env_id.split("-")[0].lower()
            run_30seed_experiment(
                env_id=env_id,
                exp_name=env_short,
                seeds=seeds,
                generations=args.generations,
                max_timesteps=args.max_timesteps,
                max_evaluations=args.max_evaluations,
                pop_size=args.pop_size,
                ac_bins_override=-1,
                algo="es",
                workers=args.workers,
                noise_table_size=args.noise_table_size,
                resume=resume,
            )

    if args.suite in ("ablation", "both"):
        ablation_env = args.envs[0]
        env_short = ablation_env.split("-")[0].lower()

        run_30seed_experiment(
            env_id=ablation_env,
            exp_name=f"{env_short}_es_bins10",
            seeds=seeds,
            generations=args.generations,
            max_timesteps=args.max_timesteps,
            max_evaluations=args.max_evaluations,
            pop_size=args.pop_size,
            ac_bins_override=10,
            algo="es",
            workers=args.workers,
            noise_table_size=args.noise_table_size,
            resume=resume,
        )

        run_30seed_experiment(
            env_id=ablation_env,
            exp_name=f"{env_short}_es_continuous",
            seeds=seeds,
            generations=args.generations,
            max_timesteps=args.max_timesteps,
            max_evaluations=args.max_evaluations,
            pop_size=args.pop_size,
            ac_bins_override=0,
            algo="es",
            workers=args.workers,
            noise_table_size=args.noise_table_size,
            resume=resume,
        )

        run_30seed_experiment(
            env_id=ablation_env,
            exp_name=f"{env_short}_random_search",
            seeds=seeds,
            generations=args.generations,
            max_timesteps=args.max_timesteps,
            max_evaluations=args.max_evaluations,
            pop_size=args.pop_size,
            ac_bins_override=-1,
            algo="random_search",
            workers=args.workers,
            noise_table_size=args.noise_table_size,
            resume=resume,
        )

    generate_all_reports("runs")


if __name__ == "__main__":
    main()