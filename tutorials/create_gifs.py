import os
from PIL import Image
import glob
import shutil

def scenario_visualzation():
    # Create the frames
    frames = []
    exp_root = os.getenv('NUPLAN_EXP_ROOT')
    gif_root = f'{exp_root}/training/scenario_visualization'
    save_dir = f'{exp_root}/training/create_gifs'
    
    # remove previous create_gifs folder
    shutil.rmtree(f'{save_dir}', ignore_errors=True)
    
    # Get the list of all files and folders in the specified directory
    dir_contents = os.listdir(gif_root)
    # Filter out only the folders from the list
    dirs = sorted([f for f in dir_contents if os.path.isdir(os.path.join(gif_root, f))])
    
    for dir in dirs:
        time_sorted_images = sorted(glob.glob(f'{gif_root}/{dir}/*.png'), key=os.path.getmtime)
        frames = [Image.open(i) for i in time_sorted_images]
        if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        # Save into a GIF file that loops forever
        rest_images = [frames[0]] * 3 + frames[1:]
        try:
            frames[0].save(f'{save_dir}/{dir}.gif', format='GIF', append_images=rest_images, save_all=True, duration=500, loop=0)
        except:
            print(f'Could not create {dir}.gif :(')
            
if __name__ == "__main__":
    scenario_visualzation()