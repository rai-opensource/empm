import pickle
import json
import numpy as np
from scipy.spatial.transform import Rotation as R


def process_cam_extrinsic():

    # # file_path = "camera/calibrate_custom.pkl"
    # file_path = "../data_custom/data/different_types/double_lift_sloth_test/calibrate.pkl"
    # with open(file_path, "rb") as f:
    #     data = pickle.load(f)
    #     print(data)

    file_path = "camera/hand_eye_calibration.json"  # Bdai Fraka calibr
    with open(file_path, "r") as file:
        data = json.load(file)

    cam_extrinsic = []
    # for i in range(3):
    for i in [1, 2, 0]:  # [24.. 23.. 21..] -> [23.. 21.. 24..] ******
        camera_info = data[f"camera{i}"]
        # print("Position (x, y, z):", camera_info["pos"])
        # print("Quaternion (x, y, z, w):", camera_info["quat"])

        # The quaternion in SciPy is expected as (x, y, z, w)
        quat = np.array(camera_info["quat"])
        rotation = R.from_quat(quat)
        rotation_matrix = rotation.as_matrix()
        # Get the position vector
        position_vector = np.array(camera_info["pos"])

        # Create the 4x4 transformation matrix
        mat4x4 = np.identity(4)
        mat4x4[:3, :3] = rotation_matrix
        mat4x4[:3, 3] = position_vector

        # convert from ros convention to cv convention
        mat4x4 = mat4x4 @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0],
                                    [0.0, 0.0, 0.0, 1.0]])

        cam_extrinsic.append(mat4x4)

    print(cam_extrinsic)

    # with open("camera/calibrate_cv.pkl", 'wb') as file:
    #     pickle.dump(cam_extrinsic, file)


process_cam_extrinsic()
