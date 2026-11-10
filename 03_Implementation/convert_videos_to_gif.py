import os
from moviepy import VideoFileClip

def convert_videos_in_directory(root_dir="."):
    print(f"Searching for video directories inside: {root_dir} ...\n")
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    found_videos = False
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == "videos":
            print(f"📁 Found video folder: {dirpath}")
            
            parent_dir = os.path.dirname(dirpath)
            gifs_output_dir = os.path.join(parent_dir, "gifs")
            os.makedirs(gifs_output_dir, exist_ok=True)
            
            mp4_files = [f for f in filenames if f.lower().endswith('.mp4')]
            
            if not mp4_files:
                print(f"   -> No MP4 files found in {dirpath}")
                continue
                
            found_videos = True
            for video_file in mp4_files:
                video_path = os.path.join(dirpath, video_file)
                base_name = os.path.splitext(video_file)[0]
                gif_filename = f"{base_name}.gif"
                gif_path = os.path.join(gifs_output_dir, gif_filename)
                
                if os.path.exists(gif_path):
                    print(f"   Skipped (already exists): {gif_filename}")
                    continue
                
                print(f"   🔄 Converting: {video_file} -> {gif_filename} ...")
                try:
                    clip = VideoFileClip(video_path)
                    clip.write_gif(gif_path, fps=15)
                    clip.close()
                    print(f"   ✅ Saved to: {gif_path}")
                except Exception as e:
                    print(f"   ❌ Error converting {video_file}: {e}")

    if not found_videos:
        print("No 'videos' folders were found.")
    else:
        print("\n✨ All video conversions completed successfully!")

if __name__ == "__main__":
    convert_videos_in_directory(".")
