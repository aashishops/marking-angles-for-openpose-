import csv
import cv2
from pose_parser import PoseParser, Body25Joints
import numpy as np

def load_angles_from_csv(csv_path):
    """
    Load angles from the CSV file and return a dictionary indexed by frame and joint triplet name.
    """
    angles_dict = {}
    with open(csv_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frame = int(row['Frame']) - 1  # Adjust to zero-index
            angles_dict[frame] = {k: float(v) if v else None for k, v in row.items() if k != 'Frame'}
    return angles_dict

def overlay_angles_on_video(video_path, csv_path, json_folder, joint_triplets, output_video_path):
    # Load angles from CSV
    angles_dict = load_angles_from_csv(csv_path)
    
    # Load joint data from JSON files
    parser = PoseParser()
    parser.load_json(json_folder)
    
    # Set up video capture and output
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_idx = 0
    while cap.isOpened():
        print(frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay angles on each frame at the specified joint triplets
        if frame_idx in angles_dict:
            for triplet in joint_triplets:
                triplet_name = f"{triplet[0].name}_{triplet[1].name}_{triplet[2].name}"
                angle = angles_dict[frame_idx].get(triplet_name)
                
                if angle is not None:
                    # Get coordinates for the middle joint (e.g., elbow in shoulder-elbow-wrist)
                    middle_joint_coords = parser.get_keypoints()[frame_idx][triplet[1]]
                    cv2.putText(frame, f"{angle:.1f}°", (int(middle_joint_coords[0]), int(middle_joint_coords[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    # Release video resources
    cap.release()
    out.release()

if __name__ == "__main__":
    # Define the path to input files
    video_path = "output.avi"
    csv_path = "joint_angles.csv"
    json_folder = "pose_json_data1"
    output_video_path = "output_video_with_angles.mp4"

    # Define joint triplets for angle calculations
    joint_triplets = [
        (Body25Joints.R_SHOULDER, Body25Joints.R_ELBOW, Body25Joints.R_WRIST),  # Right Arm
        (Body25Joints.L_SHOULDER, Body25Joints.L_ELBOW, Body25Joints.L_WRIST),  # Left Arm
        (Body25Joints.R_HIP, Body25Joints.R_KNEE, Body25Joints.R_ANKLE),        # Right Leg
        (Body25Joints.L_HIP, Body25Joints.L_KNEE, Body25Joints.L_ANKLE),        # Left Leg
        (Body25Joints.NECK, Body25Joints.R_SHOULDER, Body25Joints.L_SHOULDER),  # Upper body (Neck and Shoulders)
        (Body25Joints.R_HIP, Body25Joints.R_SHOULDER, Body25Joints.L_SHOULDER),  # Hips and Shoulders
        (Body25Joints.R_HIP, Body25Joints.R_KNEE, Body25Joints.L_KNEE)           # Hips and Knees
    ]

    # Run the overlay function
    overlay_angles_on_video(video_path, csv_path, json_folder, joint_triplets, output_video_path)
