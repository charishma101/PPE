
import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import io
import os
import imageio
# Load YOLOv5 model
model= YOLO('./best.pt')  # Adjust file path if needed
# Load YOLO model


# app.py
def detect_objects(image):
    results = model(image)
    for r in results:
        im_array = r.plot()  # Get the BGR NumPy array

    # Convert to RGB format for OpenCV:
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

    # Display the image:
        
   

    return im_rgb


# app.py
def main():
    frame_counter = 0
    sample_rate = 10 
    st.title('PPE Detection')

    uploaded_file = st.sidebar.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        
        if uploaded_file.type.startswith("video/"):
            video_bytes = uploaded_file.read()

            video_np_array = np.frombuffer(video_bytes, dtype=np.uint8)

        # Save the video locally
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(video_bytes)

        # Create video writer using FFmpeg
            output_video_path = "output_video.mp4"
            video_writer = imageio.get_writer(output_video_path, fps=30)

        # Process each frame and write to the output video
            with imageio.get_reader(video_path, 'ffmpeg') as video_reader:
                for frame in video_reader:
                # Perform operations on the frame if needed (replace this with your detection logic)
                    detected_frame = detect_objects(frame)

                # Write the processed frame to the output video
                    video_writer.append_data(detected_frame)

        # Close the video writer
            video_writer.close()

        # Display the output video in Streamlit
            st.video(output_video_path)

        # Clean up: remove the local video file
            os.remove(video_path)

        else:  # Image processing
            image_bytes = uploaded_file.read()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
            #image = cv2.imread(uploaded_file.name)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            detected_image = detect_objects(image)
            st.image(detected_image, caption="Object Detection Result", use_column_width=True)

if __name__ == '__main__':
    main()



