
import os
from moviepy.editor import VideoFileClip
import glob

if __name__ == "__main__":
    
    root_dir = os.getenv('NUPLAN_EXP_ROOT') + '/training/scenario_renderings'
    time_sorted_videos = sorted(glob.glob(f'{root_dir}/avi/*.avi'), key=os.path.getmtime)
    
    for i, video in enumerate(time_sorted_videos):
        filename = os.path.splitext(os.path.basename(video))[0]
        print(f'File {i+1}/{len(time_sorted_videos)}:')
        videoClip = VideoFileClip(video, audio=False)
        videoClip.write_gif(f'{root_dir}/gif/{filename}.gif')
        videoClip.close()
        
  