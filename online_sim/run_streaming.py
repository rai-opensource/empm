"""
    python online_sim/run_streaming.py --base_path="./data_custom/data/different_types" --case_name="static_sloth_test" --category "sloth" --shape_prior
"""
import os
from argparse import ArgumentParser
import time
import logging
import json
import glob
import numpy as np
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phys_sim import InvPhyTrainerWarpMPM
from phys_sim.utils import logger, cfg

DATA_PATH = "./data_custom/ours"
parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    default=f"{DATA_PATH}/data/different_types",
)
parser.add_argument("--case_name", type=str, default="double_hand_rope_test")
parser.add_argument("--category", type=str, default="sloth")
parser.add_argument("--n_ctrl_parts", type=int, default=2)
parser.add_argument("--inv_ctrl",
                    action="store_true",
                    help="invert horizontal control direction")
parser.add_argument("--shape_prior", action="store_true", default=False)
parser.add_argument(
    "--gaussian_path",
    type=str,
    default=f"{DATA_PATH}/gaussian_output",
)
parser.add_argument(
    "--bg_img_path",
    type=str,
    default=f"{DATA_PATH}/data/bg.png",
)
args = parser.parse_args()

from realsense_stream import CameraStreamer, Args

import torch
import pyrealsense2 as rs
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, List
import groundingdino.datasets.transforms as T
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image

# viser visualization
import viser

viser_server = viser.ViserServer()


def getPcdFromDepth(
    depth,
    intrinsic,
):
    # Depth in meters
    height, width = np.shape(depth)

    # Reshape the depth array to invert the depth values
    depth = -depth

    # Create a grid of (x, y) coordinates
    x_coords = np.arange(width)
    y_coords = np.arange(height)

    # Create a meshgrid for x and y coordinates
    X, Y = np.meshgrid(x_coords, y_coords)

    # Calculate points using vectorized operations
    old_points = np.stack([(width - X) * depth, Y * depth, depth], axis=-1)

    # Flatten the old_points array and calculate the new points using matrix multiplication
    points = np.dot(np.linalg.inv(intrinsic),
                    old_points.reshape(-1, 3).T).T.reshape(old_points.shape)

    points[:, :, 1] *= -1
    points[:, :, 2] *= -1

    return points


def transform_image(image_source) -> Tuple[np.array, torch.Tensor]:
    """
        input: PIL image
        output: original image, transformed image
    """
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def test_multi_streaming():
    # Initialize context and check connected devices
    context = rs.context()
    connected_devices = context.query_devices()

    if len(connected_devices) < 3:
        print(
            f"Only {len(connected_devices)} camera(s) found. Please connect 3 RealSense cameras."
        )
        exit(1)

    # Store pipelines and serials
    pipelines = []
    serials = []
    intrinsics_all = []

    # load cam calibrations
    base_path = args.base_path
    case_name = args.case_name
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        T_wc_all = pickle.load(f)

    # Initialize 3 camera pipelines
    for dev in connected_devices[:3]:
        serial = dev.get_info(rs.camera_info.serial_number)
        serials.append(serial)

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        pipelines.append(pipeline)

        # Get intrinsics
        profile = pipeline.get_active_profile()
        color_profile = profile.get_stream(rs.stream.color)
        intrin = color_profile.as_video_stream_profile().get_intrinsics()
        intrin_dict = {
            "height": intrin.height,
            "width": intrin.width,
            "fx": intrin.fx,
            "fy": intrin.fy,
            "cx": intrin.ppx,
            "cy": intrin.ppy,
            "model": str(intrin.model),
            "coeffs": intrin.coeffs
        }
        intrinsics_all.append(intrin_dict)

    try:
        while True:
            frames_cv2 = []
            frame_depth = []
            frames_rgb = []

            # loop through each pipeline to get frames
            for i, pipeline in enumerate(pipelines):
                frameset = pipeline.wait_for_frames()
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()

                if color_frame:
                    img_cv2 = np.asanyarray(color_frame.get_data())
                    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

                    # Overlay serial number on frame
                    cv2.putText(img_cv2, f"Serial: {serials[i]}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    frames_cv2.append(img_cv2)
                    frames_rgb.append(img_rgb)

                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    frame_depth.append(depth_image)

            # Combine images side by side
            if len(frames_cv2) == 3:
                combined = np.hstack(frames_cv2)
                cv2.imshow("RealSense Cameras", combined)

            # add viser visualization of four frames
            pcd = []
            for (i, frame_rgb) in enumerate(frames_rgb):
                # Add the camera pose as a coordinate frame
                viser_server.scene.add_frame(
                    name=f"/world/camera_{i}",
                    wxyz=Rotation.from_matrix(
                        T_wc_all[i][:3, :3]).as_quat(scalar_first=True),
                    position=T_wc_all[i][:3, 3],
                    axes_length=0.1,
                    axes_radius=0.01,
                )

                # Add the visual frustum for the camera
                intrinsics = intrinsics_all[i]
                fov = 2 * np.arctan2(intrinsics['height'],
                                     2 * intrinsics['fy']) * 180 / np.pi
                aspect = intrinsics['width'] / intrinsics['height']
                viser_server.scene.add_camera_frustum(
                    name=f"/world/camera_{serials[i]}_frustum",
                    image=frame_rgb,
                    fov=fov,
                    aspect=aspect,
                    scale=0.1,
                    wxyz=Rotation.from_matrix(
                        T_wc_all[i][:3, :3]).as_quat(scalar_first=True),
                    position=T_wc_all[i][:3, 3],
                    visible=True)

                # get the pointcloud fusing 3 views
                depth = frame_depth[i]
                K = np.array([[intrinsics['fx'], 0, intrinsics['cx']],
                              [0, intrinsics['fy'], intrinsics['cy']],
                              [0, 0, 1]])
                pcd_i = getPcdFromDepth(depth, K)
                pcd.append(pcd_i)

                # TODO: fuse and display pcd

            # if press 'q', exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if True:  # Initialize camera streamer
        cam_arg = Args()
        # cam_arg.realsense_serial = ["234222302175", "215122251521", "243122300947"]
        cam_arg.realsense_serial = ["234222302175"]
        # cam_arg.object_prompt = args.category
        cam_arg.object_prompt = "rope"
        cam_streamer = CameraStreamer(cam_arg)

    # test_multi_streaming()

    if True:  # Initialize trainer
        base_path = args.base_path
        case_name = args.case_name

        # Set the intrinsic and extrinsic parameters for visualization
        with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        cfg.c2ws = np.array(c2ws)
        cfg.w2cs = np.array(w2cs)
        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            data = json.load(f)
        cfg.intrinsics = np.array(data["intrinsics"])
        cfg.WH = data["WH"]
        cfg.bg_img_path = args.bg_img_path
        cfg.WH = [640, 480]
        cfg.case_name = case_name

        exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
        # gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"
        gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_500/point_cloud.ply"

        base_dir = f"{DATA_PATH}/temp_experiments/{case_name}"
        logger.set_log_file(path=base_dir, name="inference_log")
        trainer = InvPhyTrainerWarpMPM(
            data_path=f"{base_path}/{case_name}/final_data.pkl",
            base_dir=base_dir,
            pure_inference_mode=True,
            online=True,
        )
        # trainer.interactive_playground(None, gaussians_path, args.n_ctrl_parts,
        #                                args.inv_ctrl)

    cam_streamer.sim_realtime_callback = trainer.realtime_streaming_step
    # cam_streamer.sim_realtime_callback = trainer.initialize_pointcloud

    # cam_streamer.real_time_multi_streaming()
    while True:
        trainer.realtime_streaming_step(None)
