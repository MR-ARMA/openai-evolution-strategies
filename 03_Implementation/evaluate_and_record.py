# File: 03_Implementation/evaluate_and_record.py
import os
import pickle
import argparse
import numpy as np
import imageio

from es_core import MujocoMLPPolicy, make_tracked_env, render_tracked_frame


def evaluate_and_record(
    checkpoint_path: str,
    env_id: str = None,
    video_path: str = None,
    num_episodes: int = 5,
    seed: int = 100,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    env_id = env_id or ckpt.get("env_id", "HalfCheetah-v4")
    theta = ckpt["theta"]
    ob_mean = ckpt["ob_mean"]
    ob_std = ckpt["ob_std"]
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    ac_bins = ckpt.get("ac_bins", None)

    env_short_name = env_id.split("-")[0].lower()
    out_dir = os.path.dirname(checkpoint_path) or os.path.join("runs", env_short_name)
    if video_path is None:
        video_path = os.path.join(out_dir, f"{env_short_name}_final_showcase.mp4")
    frames_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(frames_dir, exist_ok=True)

    print("=" * 80)
    print(f"Loaded Checkpoint : {checkpoint_path}")
    print(f"Environment       : {env_id} (Generation {ckpt['generation']}, Timesteps: {ckpt['timesteps']})")
    print(f"Best Train Eval   : {ckpt['eval_return']:.2f} | Action Bins: {ac_bins}")
    print("=" * 80)

    policy = MujocoMLPPolicy(obs_dim, act_dim, hidden_dims=(64, 64), ac_bins=ac_bins)
    env = make_tracked_env(env_id, render_mode="rgb_array")

    best_frames = []
    best_ep_ret = -np.inf
    returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        done = False
        step = 0
        ep_frames = []

        while not done:
            frame = render_tracked_frame(env)
            ep_frames.append(frame)

            action = policy.act(obs, theta, ob_mean, ob_std)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
            step += 1

        returns.append(ep_ret)
        print(f"Test Episode {ep + 1:02d}/{num_episodes:02d} | Return = {ep_ret:8.2f} | Steps = {step:4d}")

        if ep_ret > best_ep_ret:
            best_ep_ret = ep_ret
            best_frames = ep_frames

    env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    print("-" * 80)
    print(f"Evaluation Summary over {num_episodes} episodes: {mean_ret:.2f} +/- {std_ret:.2f} (Max: {best_ep_ret:.2f})")

    if len(best_frames) > 0:
        print(f"Saving Chasing-Camera MP4 video ({len(best_frames)} frames @ 30fps) to {video_path}...")
        imageio.mimsave(video_path, best_frames, fps=30)

        indices = np.linspace(0, len(best_frames) - 1, 5, dtype=int)
        for idx in indices:
            frame_file = os.path.join(frames_dir, f"{env_short_name}_step_{idx:04d}.png")
            imageio.imwrite(frame_file, best_frames[idx])
        print(f"Saved 5 keyframe PNG snapshots to {frames_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and Record Trained ES Checkpoint with Chasing Camera")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to saved .pkl checkpoint")
    parser.add_argument("--env_id", type=str, default=None, help="Optional override for environment ID")
    parser.add_argument("--video_out", type=str, default=None, help="Output MP4 path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    args = parser.parse_args()

    evaluate_and_record(
        checkpoint_path=args.ckpt,
        env_id=args.env_id,
        video_path=args.video_out,
        num_episodes=args.episodes,
    )