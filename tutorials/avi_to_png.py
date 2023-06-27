import cv2
import os
from tqdm import tqdm

def save_frames(video_path, save_dir):
    # Create the save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load the video
    vidcap = cv2.VideoCapture(video_path)
    
    # Get total frames
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    success, image = vidcap.read()
    count = 0

    # Progress bar for each video
    pbar = tqdm(total=total_frames, desc="Saving frames", unit="frame")

    while success:
        # Save frame as PNG file
        cv2.imwrite(os.path.join(save_dir, f"frame{count}.png"), image)
        success, image = vidcap.read()  # Get next frame
        pbar.update(1)  # Update progress bar
        count += 1
    
    pbar.close()

def process_all_videos_in_folder(folder_path, save_dir_base):
    # Get list of all .avi files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]

    # Process each file, with progress bar
    for filename in tqdm(video_files, desc="Processing videos", unit="video"):
        video_path = os.path.join(folder_path, filename)
        save_dir = os.path.join(save_dir_base, os.path.splitext(filename)[0])
        save_frames(video_path, save_dir)


exp_root = os.getenv('NUPLAN_EXP_ROOT')
video_folder_path = f"{exp_root}/training/urban_driver_closed_loop_experiment/urban_driver_closed_loop_model/2023.06.20.11.49.56_final_200000sc/simulation/urban_driver_closed_loop_experiment/open_loop_boxes/2023.06.21.10.53.07/video_screenshot"

if __name__ == "__main__":
    process_all_videos_in_folder(video_folder_path, video_folder_path)
