# File: 03_Implementation/train_es_mujoco.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
    make_tracked_env,
    render_tracked_frame,
)

# ==============================================================================
# 1. MULTIPROCESSING ROLLOUT WORKER (Shared Memory Noise Table)
# ==============================================================================
GLOBAL_NOISE_TABLE = None
GLOBAL_ENV = None
GLOBAL_POLICY = None


def init_worker(shared_noise_base, env_id: str, obs_dim: int, act_dim: int, ac_bins):
    """Initializes each CPU worker process with a zero-copy view of the shared noise table."""
    global GLOBAL_NOISE_TABLE, GLOBAL_ENV, GLOBAL_POLICY
    GLOBAL_NOISE_TABLE = np.frombuffer(shared_noise_base, dtype=np.float32)
    GLOBAL_ENV = gym.make(env_id)
    GLOBAL_POLICY = MujocoMLPPolicy(obs_dim, act_dim, ac_bins=ac_bins)


def evaluate_antithetic_pair(args):
    """Evaluates a mirrored pair of perturbations (+eps and -eps) in the black-box environment."""
    noise_idx, theta, ob_mean, ob_std, noise_stdev, max_steps, seed, record_obs = args
    num_params = len(theta)
    eps = GLOBAL_NOISE_TABLE[noise_idx : noise_idx + num_params]

    rets = []
    lengths = []
    obs_sum = np.zeros(GLOBAL_POLICY.obs_dim, dtype=np.float64)
    obs_sq = np.zeros(GLOBAL_POLICY.obs_dim, dtype=np.float64)
    obs_cnt = 0

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
            action = GLOBAL_POLICY.act(obs, theta_perturbed, ob_mean, ob_std)
            obs, reward, terminated, truncated, _ = GLOBAL_ENV.step(action)
            total_ret += float(reward)
            steps += 1
            if terminated or truncated:
                break

        rets.append(total_ret)
        lengths.append(steps)

    return noise_idx, rets[0], rets[1], lengths[0], lengths[1], obs_sum, obs_sq, obs_cnt


# ==============================================================================
# 2. MAIN TRAINING, EVALUATION & MILESTONE VIDEO PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="OpenAI Evolution Strategies (ES) for MuJoCo Control")
    parser.add_argument(
        "--env_id",
        type=str,
        default="HalfCheetah-v4",
        choices=list(ENV_CONFIGS.keys()),
        help="MuJoCo environment ID from Table 2",
    )
    parser.add_argument("--generations", type=int, default=150, help="Number of evolutionary generations")
    parser.add_argument("--pop_size", type=int, default=None, help="Population size (must be even for antithetic pairs)")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian noise standard deviation")
    parser.add_argument("--lr", type=float, default=None, help="Adam learning rate")
    parser.add_argument("--l2coeff", type=float, default=None, help="Weight decay (L2 regularization) coefficient")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel CPU worker processes")
    parser.add_argument("--num_videos", type=int, default=10, help="Number of milestone progress MP4 videos to save")
    parser.add_argument("--noise_table_size", type=int, default=30_000_000, help="Size of pre-allocated Gaussian noise block")
    parser.add_argument("--dynamic_cap", action="store_true", help="Enable dynamic episode cap m = 2 * mean_steps (Section 2.1)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    args = parser.parse_args()

    # Load environment-specific defaults from Table 2 / Section 4.1
    cfg = ENV_CONFIGS[args.env_id]
    pop_size = args.pop_size if args.pop_size is not None else cfg["pop_size"]
    sigma = args.sigma if args.sigma is not None else cfg["sigma"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    l2coeff = args.l2coeff if args.l2coeff is not None else cfg["l2coeff"]
    ac_bins = cfg["ac_bins"]

    if pop_size % 2 != 0:
        raise ValueError(f"pop_size ({pop_size}) must be an even number for antithetic sampling.")

    # Setup structured output directories
    env_short_name = args.env_id.split("-")[0].lower()
    run_dir = os.path.join("runs", env_short_name)
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    best_ckpt_path = os.path.join(run_dir, f"{env_short_name}_best.pkl")
    latest_ckpt_path = os.path.join(run_dir, f"{env_short_name}_latest.pkl")
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    config_json_path = os.path.join(run_dir, "config.json")

    dummy_env = gym.make(args.env_id)
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]
    env_max_steps = dummy_env.spec.max_episode_steps or 1000
    dummy_env.close()

    # Compute ~10 evenly spaced milestone generations from gen 1 to args.generations
    video_schedule = sorted(list(set(np.linspace(1, args.generations, args.num_videos, dtype=int).tolist())))

    # Save experiment metadata
    with open(config_json_path, "w") as f:
        json.dump(
            {
                "env_id": args.env_id,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "ac_bins": ac_bins,
                "generations": args.generations,
                "pop_size": pop_size,
                "sigma": sigma,
                "lr": lr,
                "l2coeff": l2coeff,
                "workers": args.workers,
                "video_schedule": video_schedule,
                "target_score": cfg["target_score"],
            },
            f,
            indent=2,
        )

    print("=" * 88)
    print(f"OpenAI Evolution Strategies (ES) — Local Parallel Training Pipeline")
    print(f"Environment    : {args.env_id} (Obs: {obs_dim}, Act: {act_dim}, Action Bins: {ac_bins})")
    print(f"Output Dir     : {run_dir}/")
    print(f"Hyperparams    : Pop={pop_size} ({pop_size // 2} pairs), Sigma={sigma}, LR={lr}, L2={l2coeff}, Workers={args.workers}")
    print(f"Video Schedule : {video_schedule} -> {video_dir}/")
    print("=" * 88)

    # Pre-allocate shared Gaussian noise block (Algorithm 2 shared seed table)
    shared_noise_base = RawArray("f", args.noise_table_size)
    noise_np = np.frombuffer(shared_noise_base, dtype=np.float32)
    noise_np[:] = np.random.RandomState(args.seed).randn(args.noise_table_size).astype(np.float32)

    policy = MujocoMLPPolicy(obs_dim, act_dim, hidden_dims=(64, 64), seed=args.seed, ac_bins=ac_bins)
    optimizer = Adam(policy.theta, stepsize=lr)
    ob_stat = RunningStat(shape=(obs_dim,))

    rng = np.random.RandomState(args.seed)
    total_timesteps = 0
    best_eval_return = -np.inf
    num_pairs = pop_size // 2
    current_max_steps = env_max_steps

    # Persistent tracked evaluation environment to avoid EGL context exhaustion
    eval_env = make_tracked_env(args.env_id, render_mode="rgb_array")

    with open(metrics_csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
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

        with Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(shared_noise_base, args.env_id, obs_dim, act_dim, ac_bins),
        ) as pool:

            train_start_time = time.time()

            for gen in range(1, args.generations + 1):
                t0 = time.time()
                ob_mean, ob_std = ob_stat.mean, ob_stat.std

                noise_inds = rng.randint(0, args.noise_table_size - policy.num_params, size=num_pairs)
                seeds = rng.randint(0, 2**31 - 1, size=num_pairs)
                record_flags = rng.rand(num_pairs) < 0.10

                tasks = [
                    (
                        int(noise_inds[i]),
                        policy.theta,
                        ob_mean,
                        ob_std,
                        sigma,
                        int(current_max_steps),
                        int(seeds[i]),
                        bool(record_flags[i]),
                    )
                    for i in range(num_pairs)
                ]

                # Phase 1: Parallel Black-Box Rollouts
                results = pool.map(evaluate_antithetic_pair, tasks)

                noisy_rets = np.zeros((num_pairs, 2), dtype=np.float32)
                ep_lengths = []
                for i, (idx, r_pos, r_neg, l_pos, l_neg, o_sum, o_sq, o_cnt) in enumerate(results):
                    noisy_rets[i, 0] = r_pos
                    noisy_rets[i, 1] = r_neg
                    ep_lengths.extend([l_pos, l_neg])
                    total_timesteps += (l_pos + l_neg)
                    if o_cnt > 0:
                        ob_stat.increment(o_sum, o_sq, o_cnt)

                # Dynamic episode length cap m = 2 * mean_steps (Section 2.1)
                if args.dynamic_cap and len(ep_lengths) > 0:
                    current_max_steps = min(env_max_steps, max(100, int(2.0 * np.mean(ep_lengths))))

                # Phase 2: Fitness Shaping (Centered Rank Transformation) & Recombination
                proc_rets = compute_centered_ranks(noisy_rets)

                grad = np.zeros(policy.num_params, dtype=np.float32)
                for i in range(num_pairs):
                    idx = noise_inds[i]
                    eps = noise_np[idx : idx + policy.num_params]
                    grad += (proc_rets[i, 0] - proc_rets[i, 1]) * eps

                grad /= (pop_size * sigma)
                globalg = -grad + l2coeff * policy.theta
                policy.theta = optimizer.update(globalg)

                # Phase 3: Deterministic Parent Policy Evaluation & Chasing-Camera Video Capture
                obs, _ = eval_env.reset(seed=1337)
                eval_ret = 0.0
                eval_steps = 0
                frames = []
                is_video_gen = gen in video_schedule

                for _ in range(env_max_steps):
                    if is_video_gen:
                        frames.append(render_tracked_frame(eval_env))
                    act = policy.act(obs, policy.theta, ob_stat.mean, ob_stat.std)
                    obs, r, term, trunc, _ = eval_env.step(act)
                    eval_ret += float(r)
                    eval_steps += 1
                    if term or trunc:
                        break

                if is_video_gen and len(frames) > 0:
                    v_file = os.path.join(video_dir, f"{env_short_name}_gen_{gen:04d}_ret_{int(eval_ret)}.mp4")
                    imageio.mimsave(v_file, frames, fps=30)
                    print(f"   [VIDEO SAVED] {v_file} (Return: {eval_ret:.1f}, Steps: {eval_steps}, Frames: {len(frames)})")

                ckpt_payload = {
                    "env_id": args.env_id,
                    "theta": policy.theta.copy(),
                    "ob_mean": ob_stat.mean.copy(),
                    "ob_std": ob_stat.std.copy(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "ac_bins": ac_bins,
                    "eval_return": eval_ret,
                    "generation": gen,
                    "timesteps": total_timesteps,
                }

                with open(latest_ckpt_path, "wb") as f:
                    pickle.dump(ckpt_payload, f)

                if eval_ret > best_eval_return:
                    best_eval_return = eval_ret
                    with open(best_ckpt_path, "wb") as f:
                        pickle.dump(ckpt_payload, f)

                dt = time.time() - t0
                wall_elapsed = time.time() - train_start_time
                pop_mean = float(noisy_rets.mean())
                pop_std = float(noisy_rets.std())
                pop_max = float(noisy_rets.max())

                csv_writer.writerow([
                    gen,
                    total_timesteps,
                    f"{pop_mean:.2f}",
                    f"{pop_std:.2f}",
                    f"{pop_max:.2f}",
                    f"{eval_ret:.2f}",
                    f"{best_eval_return:.2f}",
                    eval_steps,
                    f"{wall_elapsed:.2f}",
                ])
                csvfile.flush()

                if gen % 5 == 0 or gen == 1 or is_video_gen:
                    print(
                        f"Gen {gen:04d}/{args.generations:04d} | Steps: {total_timesteps:8d} | "
                        f"PopMean: {pop_mean:7.1f} | PopMax: {pop_max:7.1f} | "
                        f"EvalRet: {eval_ret:7.1f} | Best: {best_eval_return:7.1f} | Time: {dt:.2f}s"
                    )

    eval_env.close()
    print("=" * 88)
    print(f"Training Complete for {args.env_id}!")
    print(f"  Best Evaluation Return : {best_eval_return:.2f} (Paper Target: ~{cfg['target_score']:.1f})")
    print(f"  Best Checkpoint Saved  : {best_ckpt_path}")
    print(f"  Metrics Log Saved      : {metrics_csv_path}")
    print(f"  Milestone Videos Dir   : {video_dir}/")
    print("=" * 88)


if __name__ == "__main__":
    main()