"""
    Get RealSense Cameras Info
"""

import pyrealsense2 as rs
import numpy as np
import json

# Create a context object to access connected devices
ctx = rs.context()
connected_devices = ctx.query_devices()

serial_numbers = []
intrinsics = []

if len(connected_devices) == 0:
    print("No RealSense devices found.")
else:
    for dev in connected_devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)

        # Create a pipeline for each device
        pipeline = rs.pipeline(ctx)
        config = rs.config()
        config.enable_device(serial)

        # Enable a common resolution (you can modify this if needed)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start the pipeline
        pipeline_profile = pipeline.start(config)

        # Get the active stream profile
        color_profile = pipeline_profile.get_stream(rs.stream.color)
        depth_profile = pipeline_profile.get_stream(rs.stream.depth)

        # Build and print the 3x3 intrinsic matrix
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        intr_matrix = [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy],
                       [0, 0, 1]]

        # # Get extrinsics from depth to color
        # depth_to_color = depth_profile.get_extrinsics_to(color_profile)
        # R = np.array(depth_to_color.rotation).reshape(3, 3)
        # t = np.array(depth_to_color.translation).reshape(3, 1)
        # extrinsic_d2c = np.eye(4)
        # extrinsic_d2c[:3, :3] = R
        # extrinsic_d2c[:3, 3] = t.flatten()
        # print(extrinsic_d2c)

        # # Format rotation matrix
        # R_matrix = [R[0:3], R[3:6], R[6:9]]
        # print("\nDepth-to-Color Rotation Matrix (3x3):")
        # for row in R_matrix:
        #     print(row)

        # print("\nDepth-to-Color Translation Vector (in meters):")
        # print(t)

        print(f"\n📷 Device: {name}, Serial Number: {serial}")
        # # Print camera intrinsics
        # print("  ▶ Resolution: {}x{}".format(intr.width, intr.height))
        # print("  ▶ Focal Length: fx = {:.2f}, fy = {:.2f}".format(
        #     intr.fx, intr.fy))
        # print("  ▶ Principal Point: cx = {:.2f}, cy = {:.2f}".format(
        #     intr.ppx, intr.ppy))
        # print("  ▶ Distortion Model:", intr.model)
        # print("  ▶ Distortion Coefficients:", intr.coeffs)
        # print("  ▶ Intrinsic Matrix:")
        # for row in intr_matrix:
        #     print("    ", row)

        serial_numbers.append(serial)
        intrinsics.append(intr_matrix)
        print(intr_matrix)

        # Stop the pipeline (we only needed it for intrinsics)
        pipeline.stop()


def process_cam_metadata():
    # file_path = "camera/metadata.json"
    # with open(file_path, "r") as file:
    #     metadata = json.load(file)

    metadata = {}
    metadata["intrinsics"] = intrinsics
    metadata["serial_numbers"] = serial_numbers
    metadata["fps"] = 30
    metadata["WH"] = [640, 480]
    metadata["frame_num"] = 62
    metadata["start_step"] = 827
    metadata["end_step"] = 888

    # with open("camera/metadata_custom.json", "w") as f:
    #     json.dump(metadata, f, indent=4)

    print(metadata)


process_cam_metadata()
