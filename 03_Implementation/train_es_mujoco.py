# File: 03_Implementation/train_es_mujoco.py
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
import imageio

from es_core import (
    ENV_CONFIGS,
    Adam,
    RunningStat,
    MujocoMLPPolicy,
    compute_centered_ranks,
    compute_standardized_returns,
    make_tracked_env,
    render_tracked_frame,
)

# ==============================================================================
# 1. ZERO-COPY MULTIPROCESSING ROLLOUT WORKER
# ==============================================================================
GLOBAL_NOISE_TABLE = None
GLOBAL_THETA = None
GLOBAL_OB_MEAN = None
GLOBAL_OB_STD = None
GLOBAL_ENV = None
GLOBAL_POLICY = None


def init_worker(
    shared_noise_base,
    shared_theta_base,
    shared_ob_mean_base,
    shared_ob_std_base,
    env_id: str,
    obs_dim: int,
    act_dim: int,
    ac_bins,
):
    """Initializes each CPU worker with zero-copy shared memory views of noise, theta, and obs stats."""
    global GLOBAL_NOISE_TABLE, GLOBAL_THETA, GLOBAL_OB_MEAN, GLOBAL_OB_STD, GLOBAL_ENV, GLOBAL_POLICY
    GLOBAL_NOISE_TABLE = np.frombuffer(shared_noise_base, dtype=np.float32)
    GLOBAL_THETA = np.frombuffer(shared_theta_base, dtype=np.float32)
    GLOBAL_OB_MEAN = np.frombuffer(shared_ob_mean_base, dtype=np.float32)
    GLOBAL_OB_STD = np.frombuffer(shared_ob_std_base, dtype=np.float32)
    GLOBAL_ENV = gym.make(env_id)
    GLOBAL_POLICY = MujocoMLPPolicy(obs_dim, act_dim, ac_bins=ac_bins)


def evaluate_antithetic_pair(args):
    """
    Evaluates a mirrored pair (+eps, -eps) reading theta, ob_mean, and ob_std directly
    from zero-copy shared RAM without IPC pickling overhead.
    """
    noise_idx, noise_stdev, max_steps, seed, record_obs = args
    num_params = GLOBAL_POLICY.num_params
    eps = GLOBAL_NOISE_TABLE[noise_idx : noise_idx + num_params]
    theta = GLOBAL_THETA
    ob_mean = GLOBAL_OB_MEAN
    ob_std = GLOBAL_OB_STD

    rets = []
    lengths = []
    obs_sum = np.zeros(GLOBAL_POLICY.obs_dim, dtype=np.float64)
    obs_sq = np.zeros(GLOBAL_POLICY.obs_dim, dtype=np.float64)
    obs_cnt = 0

    env_step = GLOBAL_ENV.step
    policy_act = GLOBAL_POLICY.act

    for sign in (1.0, -1.0):
        theta_perturbed = theta + (sign * noise_stdev) * eps
        obs, _ = GLOBAL_ENV.reset(seed=int(seed))
        total_ret = 0.0
        steps = 0

        for _ in range(max_steps):
            if record_obs:
                obs_sum += obs
                obs_sq += np.square(obs)
                obs_cnt += 1
            action = policy_act(obs, theta_perturbed, ob_mean, ob_std)
            obs, reward, terminated, truncated, _ = env_step(action)
            total_ret += float(reward)
            steps += 1
            if terminated or truncated:
                break

        rets.append(total_ret)
        lengths.append(steps)

    return noise_idx, rets[0], rets[1], lengths[0], lengths[1], obs_sum, obs_sq, obs_cnt


# ==============================================================================
# 2. PROGRAMMABLE SINGLE-SEED TRAINING ENGINE
# ==============================================================================
def run_single_seed_training(
    env_id: str,
    seed: int = 42,
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
    num_videos: int = 10,
    no_video: bool = False,
    dynamic_cap: bool = False,
    noise_table_size: int = 30_000_000,
    run_dir: str = None,
    metrics_csv_path: str = None,
    save_checkpoints: bool = True,
    external_pool=None,
    external_shared_buffers: dict = None,
    verbose: bool = True,
):
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

    if pop_size % 2 != 0:
        raise ValueError(f"pop_size ({pop_size}) must be an even number for antithetic sampling.")

    env_short_name = env_id.split("-")[0].lower()
    if run_dir is None:
        run_dir = os.path.join("runs", env_short_name)
    os.makedirs(run_dir, exist_ok=True)

    video_dir = os.path.join(run_dir, "videos")
    if not no_video and num_videos > 0:
        os.makedirs(video_dir, exist_ok=True)

    if metrics_csv_path is None:
        metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)

    best_ckpt_path = os.path.join(run_dir, f"{env_short_name}_best.pkl")
    latest_ckpt_path = os.path.join(run_dir, f"{env_short_name}_latest.pkl")
    config_json_path = os.path.join(run_dir, "config.json")

    dummy_env = gym.make(env_id)
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]
    env_max_steps = dummy_env.spec.max_episode_steps or 1000
    dummy_env.close()

    if not no_video and num_videos > 0:
        video_schedule = sorted(list(set(np.linspace(1, generations, num_videos, dtype=int).tolist())))
    else:
        video_schedule = []

    if save_checkpoints:
        with open(config_json_path, "w") as f:
            json.dump(
                {
                    "env_id": env_id,
                    "algo": algo,
                    "seed": seed,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "ac_bins": ac_bins,
                    "generations": generations,
                    "max_timesteps": max_timesteps,
                    "max_evaluations": max_evaluations,
                    "pop_size": pop_size,
                    "sigma": sigma,
                    "lr": lr,
                    "l2coeff": l2coeff,
                    "workers": workers,
                    "no_video": no_video,
                    "video_schedule": video_schedule,
                    "target_score": cfg["target_score"],
                },
                f,
                indent=2,
            )

    if verbose:
        print("=" * 88)
        print(f"OpenAI Evolution Strategies (ES) — Training Pipeline [Algo: {algo.upper()} | Seed: {seed}]")
        print(f"Environment    : {env_id} (Obs: {obs_dim}, Act: {act_dim}, Action Bins: {ac_bins})")
        print(f"Budget Limits  : MaxGens={generations}, MaxSteps={max_timesteps}, MaxEvals={max_evaluations}")
        print(f"Hyperparams    : Pop={pop_size} ({pop_size // 2} pairs), Sigma={sigma}, LR={lr}, L2={l2coeff}, Workers={workers}")
        print(f"Output Log     : {metrics_csv_path}")
        print("=" * 88)

    policy = MujocoMLPPolicy(obs_dim, act_dim, hidden_dims=(64, 64), seed=seed, ac_bins=ac_bins)
    optimizer = Adam(policy.theta, stepsize=lr)
    ob_stat = RunningStat(shape=(obs_dim,))

    # Set up or reuse zero-copy shared memory buffers
    if external_shared_buffers is not None:
        shared_noise_base = external_shared_buffers["noise"]
        shared_theta_base = external_shared_buffers["theta"]
        shared_ob_mean_base = external_shared_buffers["ob_mean"]
        shared_ob_std_base = external_shared_buffers["ob_std"]
    else:
        shared_noise_base = RawArray("f", noise_table_size)
        shared_theta_base = RawArray("f", policy.num_params)
        shared_ob_mean_base = RawArray("f", obs_dim)
        shared_ob_std_base = RawArray("f", obs_dim)
        np.frombuffer(shared_noise_base, dtype=np.float32)[:] = (
            np.random.RandomState(seed).randn(noise_table_size).astype(np.float32)
        )

    noise_np = np.frombuffer(shared_noise_base, dtype=np.float32)
    theta_shared_np = np.frombuffer(shared_theta_base, dtype=np.float32)
    ob_mean_shared_np = np.frombuffer(shared_ob_mean_base, dtype=np.float32)
    ob_std_shared_np = np.frombuffer(shared_ob_std_base, dtype=np.float32)
    actual_noise_size = len(noise_np)

    rng = np.random.RandomState(seed)
    total_timesteps = 0
    total_evaluations = 0
    best_eval_return = -np.inf
    best_random_theta = policy.theta.copy()
    best_random_return = -np.inf
    num_pairs = pop_size // 2
    current_max_steps = env_max_steps

    render_mode = None if (no_video or num_videos <= 0) else "rgb_array"
    eval_env = make_tracked_env(env_id, render_mode=render_mode)

    OwnPool = external_pool is None
    pool = (
        Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(
                shared_noise_base,
                shared_theta_base,
                shared_ob_mean_base,
                shared_ob_std_base,
                env_id,
                obs_dim,
                act_dim,
                ac_bins,
            ),
        )
        if OwnPool
        else external_pool
    )

    # Optimal chunksize so each worker receives its exact batch of pairs in one IPC call
    chunksize = max(1, num_pairs // workers)
    history = []

    try:
        with open(metrics_csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                "seed",
                "generation",
                "total_evaluations",
                "total_timesteps",
                "pop_mean_return",
                "pop_std_return",
                "pop_min_return",
                "pop_max_return",
                "eval_return",
                "best_eval_return",
                "eval_steps",
                "wall_time_sec",
            ])

            train_start_time = time.time()

            for gen in range(1, generations + 1):
                t0 = time.time()

                # Sync current parent weights and normalization stats to zero-copy shared RAM once per generation
                theta_shared_np[:] = policy.theta
                ob_mean_shared_np[:] = ob_stat.mean
                ob_std_shared_np[:] = ob_stat.std

                noise_inds = rng.randint(0, actual_noise_size - policy.num_params, size=num_pairs)
                seeds = rng.randint(0, 2**31 - 1, size=num_pairs)
                record_flags = rng.rand(num_pairs) < 0.10

                tasks = [
                    (
                        int(noise_inds[i]),
                        float(sigma),
                        int(current_max_steps),
                        int(seeds[i]),
                        bool(record_flags[i]),
                    )
                    for i in range(num_pairs)
                ]

                # Phase 1: Parallel Black-Box Rollouts across all CPU threads
                results = pool.map(evaluate_antithetic_pair, tasks, chunksize=chunksize)

                noisy_rets = np.zeros((num_pairs, 2), dtype=np.float32)
                ep_lengths = []
                for i, (idx, r_pos, r_neg, l_pos, l_neg, o_sum, o_sq, o_cnt) in enumerate(results):
                    noisy_rets[i, 0] = r_pos
                    noisy_rets[i, 1] = r_neg
                    ep_lengths.extend([l_pos, l_neg])
                    total_timesteps += (l_pos + l_neg)
                    total_evaluations += 2
                    if o_cnt > 0:
                        ob_stat.increment(o_sum, o_sq, o_cnt)

                    if algo == "random_search":
                        if r_pos > best_random_return:
                            best_random_return = r_pos
                            eps = noise_np[idx : idx + policy.num_params]
                            best_random_theta = (policy.theta + sigma * eps).copy()
                        if r_neg > best_random_return:
                            best_random_return = r_neg
                            eps = noise_np[idx : idx + policy.num_params]
                            best_random_theta = (policy.theta - sigma * eps).copy()

                if dynamic_cap and len(ep_lengths) > 0:
                    current_max_steps = min(env_max_steps, max(100, int(2.0 * np.mean(ep_lengths))))

                # Phase 2: Parameter Update
                if algo in ("es", "es_no_rank"):
                    if algo == "es":
                        proc_rets = compute_centered_ranks(noisy_rets)
                    else:
                        proc_rets = compute_standardized_returns(noisy_rets)

                    grad = np.zeros(policy.num_params, dtype=np.float32)
                    for i in range(num_pairs):
                        idx = noise_inds[i]
                        eps = noise_np[idx : idx + policy.num_params]
                        grad += (proc_rets[i, 0] - proc_rets[i, 1]) * eps

                    grad /= (pop_size * sigma)
                    globalg = -grad + l2coeff * policy.theta
                    policy.theta = optimizer.update(globalg)
                elif algo == "random_search":
                    policy.theta = best_random_theta.copy()
                else:
                    raise ValueError(f"Unknown algo '{algo}'.")

                # Phase 3: Deterministic Parent Policy Evaluation
                obs, _ = eval_env.reset(seed=10000 + seed)
                eval_ret = 0.0
                eval_steps = 0
                frames = []
                is_video_gen = (not no_video) and (gen in video_schedule)

                cur_mean, cur_std = ob_stat.mean, ob_stat.std
                for _ in range(env_max_steps):
                    if is_video_gen:
                        frames.append(render_tracked_frame(eval_env))
                    act = policy.act(obs, policy.theta, cur_mean, cur_std)
                    obs, r, term, trunc, _ = eval_env.step(act)
                    eval_ret += float(r)
                    eval_steps += 1
                    if term or trunc:
                        break

                if is_video_gen and len(frames) > 0:
                    v_file = os.path.join(video_dir, f"{env_short_name}_gen_{gen:04d}_ret_{int(eval_ret)}.mp4")
                    imageio.mimsave(v_file, frames, fps=30)
                    if verbose:
                        print(f"   [VIDEO SAVED] {v_file} (Return: {eval_ret:.1f}, Steps: {eval_steps})")

                ckpt_payload = {
                    "env_id": env_id,
                    "algo": algo,
                    "seed": seed,
                    "theta": policy.theta.copy(),
                    "ob_mean": ob_stat.mean.copy(),
                    "ob_std": ob_stat.std.copy(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "ac_bins": ac_bins,
                    "eval_return": eval_ret,
                    "generation": gen,
                    "evaluations": total_evaluations,
                    "timesteps": total_timesteps,
                }

                if save_checkpoints:
                    with open(latest_ckpt_path, "wb") as f:
                        pickle.dump(ckpt_payload, f)

                if eval_ret > best_eval_return:
                    best_eval_return = eval_ret
                    if save_checkpoints:
                        with open(best_ckpt_path, "wb") as f:
                            pickle.dump(ckpt_payload, f)

                dt = time.time() - t0
                wall_elapsed = time.time() - train_start_time
                pop_mean = float(noisy_rets.mean())
                pop_std = float(noisy_rets.std())
                pop_min = float(noisy_rets.min())
                pop_max = float(noisy_rets.max())

                row_dict = {
                    "seed": seed,
                    "generation": gen,
                    "total_evaluations": total_evaluations,
                    "total_timesteps": total_timesteps,
                    "pop_mean_return": pop_mean,
                    "pop_std_return": pop_std,
                    "pop_min_return": pop_min,
                    "pop_max_return": pop_max,
                    "eval_return": eval_ret,
                    "best_eval_return": best_eval_return,
                    "eval_steps": eval_steps,
                    "wall_time_sec": wall_elapsed,
                }
                history.append(row_dict)

                csv_writer.writerow([
                    seed,
                    gen,
                    total_evaluations,
                    total_timesteps,
                    f"{pop_mean:.2f}",
                    f"{pop_std:.2f}",
                    f"{pop_min:.2f}",
                    f"{pop_max:.2f}",
                    f"{eval_ret:.2f}",
                    f"{best_eval_return:.2f}",
                    eval_steps,
                    f"{wall_elapsed:.2f}",
                ])
                csvfile.flush()

                if verbose and (gen % 10 == 0 or gen == 1 or is_video_gen):
                    print(
                        f"[Seed {seed}] Gen {gen:04d}/{generations:04d} | Evals: {total_evaluations:6d} | "
                        f"Steps: {total_timesteps:8d} | PopMean: {pop_mean:7.1f} | "
                        f"EvalRet: {eval_ret:7.1f} | Best: {best_eval_return:7.1f} | Time: {dt:.2f}s"
                    )

                if max_evaluations is not None and total_evaluations >= max_evaluations:
                    break
                if max_timesteps is not None and total_timesteps >= max_timesteps:
                    break

    finally:
        eval_env.close()
        if OwnPool:
            pool.close()
            pool.join()

    return {
        "seed": seed,
        "env_id": env_id,
        "algo": algo,
        "ac_bins": ac_bins,
        "final_eval_return": history[-1]["eval_return"] if history else 0.0,
        "best_eval_return": best_eval_return,
        "total_generations": history[-1]["generation"] if history else 0,
        "total_evaluations": total_evaluations,
        "total_timesteps": total_timesteps,
        "wall_time_sec": history[-1]["wall_time_sec"] if history else 0.0,
        "best_ckpt_payload": ckpt_payload if history else None,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="OpenAI Evolution Strategies (ES) for MuJoCo Control")
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4", choices=list(ENV_CONFIGS.keys()))
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--max_timesteps", type=int, default=None)
    parser.add_argument("--max_evaluations", type=int, default=None)
    parser.add_argument("--pop_size", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--l2coeff", type=float, default=None)
    parser.add_argument("--ac_bins", type=int, default=-1)
    parser.add_argument("--algo", type=str, default="es", choices=["es", "es_no_rank", "random_search"])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 16, help="Parallel CPU workers (defaults to all logical cores)")
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--noise_table_size", type=int, default=30_000_000)
    parser.add_argument("--dynamic_cap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    run_single_seed_training(
        env_id=args.env_id,
        seed=args.seed,
        generations=args.generations,
        max_timesteps=args.max_timesteps,
        max_evaluations=args.max_evaluations,
        pop_size=args.pop_size,
        sigma=args.sigma,
        lr=args.lr,
        l2coeff=args.l2coeff,
        ac_bins_override=args.ac_bins,
        algo=args.algo,
        workers=args.workers,
        num_videos=args.num_videos,
        no_video=args.no_video,
        dynamic_cap=args.dynamic_cap,
        noise_table_size=args.noise_table_size,
        run_dir=args.run_dir,
        save_checkpoints=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()