# Author: Isaac Chang
# Contact: chang.isaac@outlook.com
# Date: 03/16/2021

from enum import IntEnum
import json
import os

# Length of one joint keypoint data in 2D (x, y, confidence)
KEYPOINT_LENGTH = 3
# Order of x coordinate in a keypoint
X_SPACING = 0
# Order of y coordinate in a keypoint
Y_SPACING = 1
# Order of confidence in a keypoint
CONF_SPACING = 2

class Body25Joints(IntEnum):
    """
    This is an enum to represent the ordering of the joint keypoint data from OpenPose
    for the BODY_25 UI mapping.
    """
    NOSE = 0
    NECK = 1
    R_SHOULDER = 2
    R_ELBOW = 3
    R_WRIST = 4
    L_SHOULDER = 5
    L_ELBOW = 6
    L_WRIST = 7
    MIDHIP = 8
    R_HIP = 9
    R_KNEE = 10
    R_ANKLE = 11
    L_HIP = 12
    L_KNEE = 13
    L_ANKLE = 14
    R_EYE = 15
    L_EYE = 16
    R_EAR = 17
    L_EAR = 18
    L_BIG_TOE = 19
    L_SMALL_TOE = 20
    L_HEEL = 21
    R_BIG_TOE = 22
    R_SMALL_TOE = 23
    R_HEEL = 24

class PoseParser:
    """
    This class is used to load and parse data from OpenPose output JSON files.
    It currently only supports 2D points and UI mapping BODY_25.
    """

    def __init__(self):
        """
        Initializes class variables.
        """
        self._is_dir = False  # Represents whether the data loaded is multiple frames or a single image
        self._data = None  # Holds all the JSON data loaded

    def load_json(self, data_path):
        """
        Verifies that the path exists on the host system and loads a single file or a directory
        full of output JSON files into the class.
        """
        if os.path.isdir(data_path):
            # If the path is a directory
            self._data = []
            self._is_dir = True
            filenames = os.listdir(data_path)
            filenames.sort()  # Sort filenames alphanumerically

            for filename in filenames:
                if filename.endswith('.json'):  # Ensure the file is a JSON file
                    try:
                        with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
                            self._data.append(json.load(f))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {filename}: {e}")
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")

        elif os.path.isfile(data_path):
            # If the path is a single file
            self._is_dir = False
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {data_path}: {e}")
            except Exception as e:
                print(f"Error loading {data_path}: {e}")

        else:
            raise FileNotFoundError("File or directory does not exist")

    def get_joint_coords(self, joint, person_id=0):
        """
        Returns the x, y, and confidence values for all given frames for a particular joint.
        """
        if self._is_dir:
            x, y, confidence = [], [], []

            for frame_num in range(len(self._data)):
             
                frame_x, frame_y, frame_conf = self._parse_pose_frame(joint, person_id, frame_num)
                x.append(frame_x)
                y.append(frame_y)
                confidence.append(frame_conf)

            if len(x) == len(y) == len(confidence):
                return x, y, confidence
            else:
                raise ValueError("Length of x, y, and confidence values from frames do not match")
        else:
            # Extract coordinates from a single image
            frame_x, frame_y, frame_conf = self._parse_pose_frame(joint, person_id)
            return frame_x, frame_y, frame_conf

    def _parse_pose_frame(self, joint, person_index, frame_num=None):
        """
        Parses a single frame of data to return the x, y, and confidence.
        """
        keypoints = None

        if frame_num is None:
            keypoints = self._data["people"][person_index]["pose_keypoints_2d"]
        else:
            keypoints = self._data[frame_num]["people"][person_index]["pose_keypoints_2d"]

        # Verify length of keypoints
        if len(keypoints) != len(Body25Joints) * KEYPOINT_LENGTH:
            raise ValueError("Keypoints do not match format for BODY_25")

        x = keypoints[(joint * KEYPOINT_LENGTH) + X_SPACING]
        y = keypoints[(joint * KEYPOINT_LENGTH) + Y_SPACING]
        confidence = keypoints[(joint * KEYPOINT_LENGTH) + CONF_SPACING]

        return x, y, confidence
