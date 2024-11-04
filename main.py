import csv
from pose_parser import PoseParser
from pose_parser import Body25Joints
from pose_math import PoseMath
from pose_plot import PosePlot
import numpy as np


def main():
    parser = PoseParser()

    # Load output JSON files from OpenPose
    parser.load_json("pose_json_data1")

    # Define specific triplets of joints for angle calculation
    joint_triplets = [
        (Body25Joints.R_SHOULDER, Body25Joints.R_ELBOW, Body25Joints.R_WRIST),  # Right Arm
        (Body25Joints.L_SHOULDER, Body25Joints.L_ELBOW, Body25Joints.L_WRIST),  # Left Arm
        (Body25Joints.R_HIP, Body25Joints.R_KNEE, Body25Joints.R_ANKLE),        # Right Leg
        (Body25Joints.L_HIP, Body25Joints.L_KNEE, Body25Joints.L_ANKLE),        # Left Leg
        (Body25Joints.NECK, Body25Joints.R_SHOULDER, Body25Joints.L_SHOULDER),  # Upper body (Neck and Shoulders)
        (Body25Joints.R_HIP, Body25Joints.R_SHOULDER, Body25Joints.L_SHOULDER),  # Hips and Shoulders
        (Body25Joints.R_HIP, Body25Joints.R_KNEE, Body25Joints.L_KNEE)           # Hips and Knees
    ]

    # Initialize a dictionary to store angles for each triplet
    angles_dict = {}

    # Iterate through each joint triplet and calculate angles
    for joint1, joint2, joint3 in joint_triplets:
        joint_angles = calculate_joint_angles(parser, joint1, joint2, joint3)
        if joint_angles:
            # Store the angles in the dictionary with a key for the triplet
            angles_dict[f"{joint1.name}_{joint2.name}_{joint3.name}"] = joint_angles
            plotter = PosePlot(joint_angles,
                               200,
                               f"Angle Between {joint1.name}, {joint2.name}, and {joint3.name}",
                               "Angle (deg)",
                               "Frame Number")
            plotter.animate()


    # Write the angles to a CSV file
    write_angles_to_csv(angles_dict, "joint_angles.csv")

def calculate_joint_angles(parser, joint1, joint2, joint3):
    # Extract coordinates for the three joints
    joint1_x, joint1_y, _ = parser.get_joint_coords(joint1)
    joint2_x, joint2_y, _ = parser.get_joint_coords(joint2)
    joint3_x, joint3_y, _ = parser.get_joint_coords(joint3)

    # Initialize a list to store angles
    joint_angles = []
    
    # Verify length of extracted coords are the same
    if len(joint1_x) == len(joint2_x) == len(joint3_x):
        num_frames = len(joint1_x)
        for frame_num in range(num_frames):
            # Define points for the joints
            p_joint1 = [joint1_x[frame_num], joint1_y[frame_num]]
            p_joint2 = [joint2_x[frame_num], joint2_y[frame_num]]
            p_joint3 = [joint3_x[frame_num], joint3_y[frame_num]]

            # Define vectors for joint2->joint1 and joint2->joint3
            v1 = PoseMath.make_vector(p_joint2, p_joint1)  # Vector from joint2 to joint1
            v2 = PoseMath.make_vector(p_joint2, p_joint3)  # Vector from joint2 to joint3

            # Get angle between the vectors
           
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                # Get angle between the vectors
                angle = PoseMath.get_angle_between(v1, v2, in_deg=True)
                joint_angles.append(angle)
            else:
                # Append None or a specific value for undefined angles
                joint_angles.append(None)


        return joint_angles
    else:
        print(f"Length of coordinates for joints {joint1}, {joint2}, and {joint3} do not match.")
        return None



def write_angles_to_csv(angles_dict, filename):
    # Open a new CSV file for writing
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row with triplet names
        header = ["Frame"] + list(angles_dict.keys())
        writer.writerow(header)

        # Get the maximum number of frames
        num_frames = max(len(angles) for angles in angles_dict.values())

        # Write angles for each frame
        for frame in range(num_frames):
            row = [frame + 1]  # Start frame numbering from 1
            for angles in angles_dict.values():
                if frame < len(angles):
                    row.append(angles[frame])
                else:
                    row.append(None)  # Handle missing data for frames
            writer.writerow(row)

if __name__ == "__main__":
    main()
