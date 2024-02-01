
import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO

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

    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        if uploaded_file.type.startswith("video/"):
            cap = cv2.VideoCapture(uploaded_file.name)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Create an output video with the same dimensions and FPS
            out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

    

                #frame_counter += 1
                #if frame_counter % sample_rate == 0:  # Process every nth frame
                detected_frame = detect_objects(frame)
                out.write(detected_frame)  # Write the processed frame to the output video

            cap.release()
            out.release()

            st.video('output_video.mp4')  # Display the output video in Streamlit

        else:  # Image processing
            image = cv2.imread(uploaded_file.name)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            detected_image = detect_objects(image)
            st.image(detected_image, caption="Object Detection Result", use_column_width=True)

if __name__ == '__main__':
    main()



