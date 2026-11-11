# File: 03_Implementation/convert_videos_to_gif.py

import os
import glob

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


def convert_mp4_to_gif(video_path: str, gif_path: str, fps: int = 15):
    if os.path.exists(gif_path):
        print(f"   [Skipped] Already exists: {gif_path}")
        return

    print(
        f"   [Converting] {os.path.basename(video_path)} "
        f"-> {os.path.basename(gif_path)} ..."
    )

    clip = VideoFileClip(video_path)

    try:
        clip.write_gif(gif_path, fps=fps, logger=None)
    except TypeError:
        clip.write_gif(gif_path, fps=fps)
    finally:
        clip.close()

    print(f"   [Saved] {gif_path}")


def convert_selected_videos(
    runs_root: str = "runs",
    max_milestone_gifs: int = 5
):
    if not os.path.exists(runs_root):
        print(f"Error: Directory '{runs_root}' does not exist.")
        return

    env_dirs = sorted(
        [
            d for d in glob.glob(os.path.join(runs_root, "*"))
            if os.path.isdir(d)
        ]
    )

    converted_any = False

    for env_dir in env_dirs:
        video_dir = os.path.join(env_dir, "videos")
        gifs_dir = os.path.join(env_dir, "gifs")

        mp4_files = (
            sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
            if os.path.exists(video_dir)
            else []
        )

        showcase_mp4s = sorted(
            glob.glob(os.path.join(env_dir, "*_final_showcase.mp4"))
        )

        if not mp4_files and not showcase_mp4s:
            continue

        os.makedirs(gifs_dir, exist_ok=True)
        converted_any = True

        print(f"\nProcessing environment folder: {env_dir}")

        # 1. Select up to 5 evenly-spaced milestone progression videos
        if len(mp4_files) > max_milestone_gifs:
            step_idxs = [
                round(i * (len(mp4_files) - 1) / (max_milestone_gifs - 1))
                for i in range(max_milestone_gifs)
            ]

            selected_mp4s = [mp4_files[idx] for idx in step_idxs]
        else:
            selected_mp4s = mp4_files

        for video_path in selected_mp4s:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            gif_path = os.path.join(gifs_dir, f"{base_name}.gif")

            convert_mp4_to_gif(video_path, gif_path, fps=15)

        # 2. Convert the final showcase MP4 into the gifs/ folder as well
        for showcase_path in showcase_mp4s:
            base_name = os.path.splitext(os.path.basename(showcase_path))[0]
            gif_path = os.path.join(gifs_dir, f"{base_name}.gif")

            convert_mp4_to_gif(showcase_path, gif_path, fps=20)

    if not converted_any:
        print("No MP4 videos found under 'runs/'.")
    else:
        print("\nAll selected milestone and showcase GIFs created!")


if __name__ == "__main__":
    convert_selected_videos("runs")