# Importing all necessary libraries
import cv2
import imageio
import os


root_dir = os.getenv('NUPLAN_EXP_ROOT') + '/training/scenario_renderings'


# os.listdir(f'{root_dir}/avi')


# Open the AVI video file
video = cv2.VideoCapture(f'{root_dir}/avi/starting_left_turn-unknown_tutorial_planner_93b9b0c9aaba597b.avi')

# Initialize the imageio writer
writer = imageio.get_writer(f'{root_dir}/gif/output.gif', mode='I', duration=0.1)

# Loop through the frames in the video
while True:
    # Read the next frame
    ret, frame = video.read()

    # If there are no more frames, break out of the loop
    if not ret:
        break

    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Add the frame to the imageio writer
    writer.append_data(frame)

# Close the imageio writer and release the video capture
writer.close()
video.release()
cv2.destroyAllWindows()