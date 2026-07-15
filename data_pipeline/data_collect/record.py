"""
    Record with RealSense Cameras
"""
import pyrealsense2 as rs
import cv2
import time
import os
import numpy as np

# Parameters
RECORD_DURATION = 12  # seconds
RESOLUTION = (640, 480)
FPS = 30
OUTPUT_DIR = "videos_raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. List RealSense devices
ctx = rs.context()
connected_devices = ctx.query_devices()
serials = [
    dev.get_info(rs.camera_info.serial_number) for dev in connected_devices
]

if not serials:
    raise Exception("No RealSense devices found!")

print(f"Found devices: {serials}")

pipelines = []
video_writers = []
depth_dirs = []
align_objects = []  # For aligning depth to color

# 2. Setup pipelines and writers
for serial in serials:
    # Create subfolders
    cam_dir = os.path.join(OUTPUT_DIR, serial)
    os.makedirs(cam_dir, exist_ok=True)
    depth_dir = os.path.join(cam_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    depth_dirs.append(depth_dir)

    # Setup pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1],
                         rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1],
                         rs.format.z16, FPS)
    pipeline.start(config)
    pipelines.append(pipeline)

    # Print camera intrinsics
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    print(f"Camera {serial} intrinsics:")
    print(f"  Width: {intr.width}, Height: {intr.height}")
    print(f"  fx: {intr.fx}, fy: {intr.fy}")
    print(f"  ppx: {intr.ppx}, ppy: {intr.ppy}")
    print(f"  Distortion model: {intr.model}, Coeffs: {intr.coeffs}")

    # Setup align object
    align_to = rs.stream.color
    align = rs.align(align_to)
    align_objects.append(align)

    # Setup video writer
    color_video_path = os.path.join(cam_dir, "color.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(color_video_path, fourcc, FPS, RESOLUTION)
    video_writers.append(writer)

# 3. Start recording
print("Recording started...")
time.sleep(3.0)  # Wait for 2 seconds before starting
start_time = time.time()
frame_index = 0

try:
    while time.time() - start_time < RECORD_DURATION:
        for i, pipeline in enumerate(pipelines):
            frames = pipeline.wait_for_frames()

            # Align depth to color
            aligned_frames = align_objects[i].process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if color_frame and depth_frame:
                # Save color frame
                color_image = np.asanyarray(color_frame.get_data())
                video_writers[i].write(color_image)

                # Save depth frame as .npy
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_path = os.path.join(depth_dirs[i],
                                          f"{frame_index:05d}.npy")
                np.save(depth_path, depth_image)

        frame_index += 1

except KeyboardInterrupt:
    print("Recording interrupted by user!")

# 4. Cleanup
print("Stopping pipelines and saving data...")
for pipeline in pipelines:
    pipeline.stop()
for writer in video_writers:
    writer.release()

cv2.destroyAllWindows()
print("Recording complete. Files saved in:", OUTPUT_DIR)
