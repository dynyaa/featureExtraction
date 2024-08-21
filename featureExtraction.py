import cv2
import numpy as np
import os

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def calculate_deformation(frame, reference_frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_frame, gray_reference)
    deformation = np.sum(diff)
    return deformation

def find_most_deformed_frame(frames):
    reference_frame = frames[0]
    max_deformation = 0
    most_deformed_frame = reference_frame
    
    for frame in frames:
        deformation = calculate_deformation(frame, reference_frame)
        if deformation > max_deformation:
            max_deformation = deformation
            most_deformed_frame = frame
    
    return most_deformed_frame

def save_frame(frame, output_path):
    cv2.imwrite(output_path, frame)

def process_videos(video_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(video_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_directory, filename)
            frames = extract_frames(video_path)
            most_deformed_frame = find_most_deformed_frame(frames)
            output_path = os.path.join(output_directory, filename.replace(".mp4", ".jpg"))
            save_frame(most_deformed_frame, output_path)
            print(f"Processed {filename}")

if __name__ == "__main__":
    video_directory = "videos"
    output_directory = "output_frames"

    process_videos(video_directory, output_directory)