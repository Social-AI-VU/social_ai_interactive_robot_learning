import cv2
import numpy as np

def run_eye_in_hand_calibration(R_base_to_flange, t_base_to_flange, R_cam_to_target, t_cam_to_target):
    """
    Calibrates the wrist camera.
    Finds the transformation from the Robot Flange to the Wrist Camera.
    """
    # OpenCV expects matrices moving from flange to base for the robot chain
    R_flange_to_base = []
    t_flange_to_base = []
    
    for R, t in zip(R_base_to_flange, t_base_to_flange):
        # Invert the base-to-flange matrices
        R_inv = R.T
        t_inv = -R_inv @ t
        R_flange_to_base.append(R_inv)
        t_flange_to_base.append(t_inv)

    # Solve AX = XB
    R_flange_to_cam, t_flange_to_cam = cv2.calibrateHandEye(
        R_gripper2base=R_flange_to_base,
        t_gripper2base=t_flange_to_base,
        R_target2cam=R_cam_to_target,
        t_target2cam=t_cam_to_target,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    return R_flange_to_cam, t_flange_to_cam

def run_eye_on_base_calibration(R_base_to_flange, t_base_to_flange, R_cam_to_target, t_cam_to_target):
    """
    Calibrates the static room/wall camera.
    Finds the transformation from the Robot Base to the Static Camera.
    """
    # For Eye-on-Base, the target is fixed to the gripper. 
    # cv2.calibrateHandEye can solve this if we pass base_to_flange directly.
    R_base_to_cam, t_base_to_cam = cv2.calibrateHandEye(
        R_gripper2base=R_base_to_flange,
        t_gripper2base=t_base_to_flange,
        R_target2cam=R_cam_to_target,
        t_target2cam=t_cam_to_target,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    return R_base_to_cam, t_base_to_cam

def make_homogeneous_matrix(R, t):
    """Helper to convert Rotation and Translation into a 4x4 matrix"""
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.reshape(3,)
    return T

# ==========================================
# EXAMPLE USAGE WITH DUMMY DATA
# ==========================================
if __name__ == "__main__":
    # Generate mock data for 15 calibration poses
    num_poses = 15
    
    # 1. Robot telemetry (from Franka arm)
    # List of 3x3 rotation matrices
    mock_R_base_to_flange = [np.eye(3) for _ in range(num_poses)]
    # List of 3x1 translation vectors (in meters)
    mock_t_base_to_flange = [np.array([[0.4], [0.0], [0.5]]) for _ in range(num_poses)]
    
    # 2. Vision telemetry (from cv2.estimatePoseSingleMarkers or ChArUco)
    mock_R_cam_to_target = [np.eye(3) for _ in range(num_poses)]
    mock_t_cam_to_target = [np.array([[0.0], [0.0], [0.3]]) for _ in range(num_poses)]

    # --- Run Wrist Camera Calibration ---
    R_wrist, t_wrist = run_eye_in_hand_calibration(
        mock_R_base_to_flange, mock_t_base_to_flange, 
        mock_R_cam_to_target, mock_t_cam_to_target
    )
    T_flange_to_wrist = make_homogeneous_matrix(R_wrist, t_wrist)
    print("--- WRIST CAMERA CALIBRATION (T_flange_to_camera) ---")
    print(T_flange_to_wrist)

    # --- Run Static Camera Calibration ---
    R_static, t_static = run_eye_on_base_calibration(
        mock_R_base_to_flange, mock_t_base_to_flange, 
        mock_R_cam_to_target, mock_t_cam_to_target
    )
    T_base_to_static = make_homogeneous_matrix(R_static, t_static)
    print("\n--- STATIC CAMERA CALIBRATION (T_base_to_camera) ---")
    print(T_base_to_static)
