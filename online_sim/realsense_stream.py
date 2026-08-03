import cv2
import numpy as np
import os
import time
import tyro
from typing import Optional
from dataclasses import dataclass

from core.sensor.realsense.multi_realsense import MultiRealsense
from core.perception.sam.predictor import SAM2ImageSegmenter
from core.perception.cotracker.tracker import (
    CoTrackerOnlineTracker,
    draw_tracks_on_video,
)
from matplotlib import cm
import torch

DEFAULT_DEVICE = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class Args:
    # SAM2
    hf_model: str = "facebook/sam2.1-hiera-large"
    sam_device: str = DEFAULT_DEVICE
    # CoTracker
    checkpoint: Optional[str] = None
    cot_device: str = DEFAULT_DEVICE
    # Camera
    width: int = 640
    height: int = 480
    fps: int = 30
    realsense_serial: Optional[str] = "234222302175"
    # UI/vis
    alpha: float = 0.5
    trace_length: int = 2
    # Point sampling from mask
    point_stride: int = 12  # pixels between candidate points on grid
    max_points: int = 800

    object_prompt: str = "object"


def sample_points_from_mask(mask: np.ndarray, stride: int,
                            max_points: int) -> np.ndarray:
    h, w = mask.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    inside = mask[grid_points[:, 1].clip(0, h - 1),
                  grid_points[:, 0].clip(0, w - 1)] > 0
    sampled = grid_points[inside]
    if sampled.shape[0] > max_points:
        idx = np.random.choice(sampled.shape[0],
                               size=max_points,
                               replace=False)
        sampled = sampled[idx]
    return sampled.astype(np.float32)


def select_mask_on_first_frame(first_bgr, segmenter, alpha):
    window_name = "Select mask (click or drag box). Enter=confirm, c=clear, q=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    selecting = False
    start_pt = (0, 0)
    current_pt = (0, 0)
    selected_point = None
    selected_box = None

    def on_mouse(event, x, y, flags, param):
        nonlocal selecting, start_pt, current_pt, selected_point, selected_box
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_pt = (x, y)
            current_pt = (x, y)
            selected_point = (x, y)
            selected_box = None
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            current_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and selecting:
            selecting = False
            x1, y1 = start_pt
            x2, y2 = x, y
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                x_min, y_min = min(x1, x2), min(y1, y2)
                x_max, y_max = max(x1, x2), max(y1, y2)
                selected_box = (x_min, y_min, x_max, y_max)
                selected_point = None

    cv2.setMouseCallback(window_name, on_mouse)
    segmenter.set_image(first_bgr)

    while True:
        disp = first_bgr.copy()
        if selecting:
            cv2.rectangle(disp, start_pt, current_pt, (0, 255, 255), 2)
        elif selected_box is not None:
            x1, y1, x2, y2 = selected_box
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
        elif selected_point is not None:
            cv2.circle(disp, selected_point, 5, (0, 255, 0), -1)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            selected_point = None
            selected_box = None
            selecting = False
        elif key in (13, 10):  # Enter
            if selected_box is not None:
                masks, scores, _ = segmenter.predict_from_box(
                    np.array(selected_box, dtype=np.float32))
            elif selected_point is not None:
                pt = np.array([[selected_point[0], selected_point[1]]],
                              dtype=np.float32)
                labels = np.array([1], dtype=np.int32)
                masks, scores, _ = segmenter.predict_from_points(
                    pt, labels, multimask_output=True)
            else:
                continue
            best_mask, _ = segmenter.get_best_mask(masks,
                                                   scores,
                                                   strategy="highest_score")
            best_mask = (best_mask > 0).astype(np.uint8)
            overlay = segmenter.overlay_masks(first_bgr,
                                              best_mask,
                                              alpha=alpha)
            cv2.imshow(window_name, overlay)
            cv2.waitKey(100)
            return best_mask, overlay
        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Selection canceled by user")


class CameraStreamer():

    def __init__(self, args: Args):
        self.args = args
        self.sim_realtime_callback = None

    def real_time_tracking(self):
        hf_model = self.args.hf_model
        checkpoint = self.args.checkpoint
        sam_device = self.args.sam_device
        cot_device = self.args.cot_device
        width = self.args.width
        height = self.args.height
        fps = self.args.fps
        realsense_serial = self.args.realsense_serial
        alpha = self.args.alpha
        trace_length = self.args.trace_length
        point_stride = self.args.point_stride
        max_points = self.args.max_points

        try:
            cot_tracker = CoTrackerOnlineTracker(
                checkpoint=checkpoint,
                device=cot_device,
                grid_size=15,
                trace_length=trace_length,
            )
        except RuntimeError as e:
            print(str(e))
            return

        with MultiRealsense(
                enable_color=True,
                enable_depth=False,
                resolution=(width, height),
                capture_fps=fps,
                serial_numbers=[realsense_serial]
                if realsense_serial is not None else None,
        ) as cameras:
            if cameras.n_cameras == 0:
                print("No RealSense cameras detected.")
                return

            print(f"Using camera {list(cameras.cameras.keys())[0]}")
            first_bgr = None
            print(
                "Waiting 0.5 seconds for camera white balance to stabilize...")
            time.sleep(0.5)
            while first_bgr is None:
                frames_data = cameras.get_vis()
                if frames_data is not None and "color" in frames_data and frames_data[
                        "color"].shape[0] > 0:
                    first_bgr = frames_data["color"][0]

            segmenter = SAM2ImageSegmenter(hf_model_id=hf_model,
                                           device=sam_device)
            try:
                mask01, overlay = select_mask_on_first_frame(first_bgr,
                                                             segmenter,
                                                             alpha=alpha)
            except RuntimeError as e:
                print(str(e))
                return

            points_xy = sample_points_from_mask(mask01,
                                                stride=point_stride,
                                                max_points=max_points)
            if points_xy.shape[0] == 0:
                print("No points sampled from the selected mask. Aborting.")
                return

            t_col = np.zeros((points_xy.shape[0], 1), dtype=np.float32)
            queries_np = np.concatenate(
                [t_col, points_xy.astype(np.float32)], axis=1)
            queries = torch.from_numpy(queries_np[None,
                                                  ...]).float().to(cot_device)

            h, w = height, width
            tracking_view = np.zeros((h, w, 3), dtype=np.uint8)
            last_vis_time = 0.0

            window_name = "CoTracker Tracking (from SAM2 selection)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            while True:
                frames_data = cameras.get_vis()
                if (frames_data is None or "color" not in frames_data
                        or frames_data["color"].shape[0] == 0):
                    continue

                camera_view = frames_data["color"][0]
                pred_tracks, pred_visibility = cot_tracker.track_frame(
                    camera_view, queries=queries)

                if pred_tracks is not None:
                    vis_start = time.time()
                    video_chunk = cot_tracker.get_last_video_chunk()
                    video_chunk_len = video_chunk.shape[1]
                    tracks_chunk = pred_tracks[:, -video_chunk_len:]
                    visibility_chunk = pred_visibility[:, -video_chunk_len:]

                    if cot_tracker.point_colors is None:
                        num_points = pred_tracks.shape[2]
                        cot_tracker.point_colors = (
                            cm.get_cmap("gist_rainbow")(np.linspace(
                                0, 1, num_points))[:, :3] * 255).astype(
                                    np.uint8)

                    last_frame_with_tracks = draw_tracks_on_video(
                        video=video_chunk,
                        tracks=tracks_chunk,
                        visibility=visibility_chunk,
                        point_colors=cot_tracker.point_colors,
                        tracks_leave_trace=trace_length,
                    )
                    # print(tracks_chunk.shape, visibility_chunk.shape)
                    tracking_view = cv2.cvtColor(last_frame_with_tracks,
                                                 cv2.COLOR_RGB2BGR)
                    last_vis_time = time.time() - vis_start

                stats_text = (
                    f"Model: {cot_tracker.last_model_time*1000:.1f}ms | Vis: {last_vis_time*1000:.1f}ms"
                )
                display_frame = tracking_view if cot_tracker.get_last_video_chunk(
                ) is not None else camera_view
                hh = display_frame.shape[0]
                cv2.putText(
                    display_frame,
                    stats_text,
                    (10, hh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        cv2.destroyAllWindows()

    def real_time_mask_tracking(self):
        hf_model = self.args.hf_model
        sam_device = self.args.sam_device
        width = self.args.width
        height = self.args.height
        fps = self.args.fps
        realsense_serial = self.args.realsense_serial
        alpha = self.args.alpha

        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

        with MultiRealsense(
                enable_color=True,
                enable_depth=False,
                resolution=(width, height),
                capture_fps=fps,
                serial_numbers=[realsense_serial]
                if realsense_serial is not None else None,
        ) as cameras:
            if cameras.n_cameras == 0:
                print("No RealSense cameras detected.")
                return

            print(f"Using camera {list(cameras.cameras.keys())[0]}")
            first_bgr = None
            print(
                "Waiting 0.5 seconds for camera white balance to stabilize...")
            time.sleep(0.5)
            while first_bgr is None:
                frames_data = cameras.get_vis()
                if frames_data is not None and "color" in frames_data and frames_data[
                        "color"].shape[0] > 0:
                    first_bgr = frames_data["color"][0]

            segmenter = SAM2ImageSegmenter(hf_model_id=hf_model,
                                           device=sam_device)
            try:
                mask01, overlay = select_mask_on_first_frame(first_bgr,
                                                             segmenter,
                                                             alpha=alpha)
            except RuntimeError as e:
                print(str(e))
                return

            best_mask = mask01.copy()  # Initialize with first mask

            window_name = "SAM2 Mask Tracking"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            while True:
                frames_data = cameras.get_vis()
                if frames_data is None or "color" not in frames_data or frames_data[
                        "color"].shape[0] == 0:
                    continue

                camera_view = frames_data["color"][0]
                segmenter.set_image(camera_view)

                # Sample points from previous mask and use as prompt
                points = sample_points_from_mask(best_mask,
                                                 stride=12,
                                                 max_points=10)
                if points.shape[0] > 0:
                    labels = np.ones(points.shape[0], dtype=np.int32)
                    masks, scores, _ = segmenter.predict_from_points(
                        points, labels, multimask_output=True)
                    best_mask, _ = segmenter.get_best_mask(
                        masks, scores, strategy="highest_score")
                    best_mask = (best_mask > 0).astype(np.uint8)
                else:
                    best_mask = mask01  # fallback to initial mask

                overlay = segmenter.overlay_masks(camera_view,
                                                  best_mask,
                                                  alpha=alpha)

                ################################################
                rgb = cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                hand_marks = []
                if results.multi_hand_landmarks:
                    h, w, _ = camera_view.shape
                    for hand_lms in results.multi_hand_landmarks:
                        x_list = [int(lm.x * w) for lm in hand_lms.landmark]
                        y_list = [int(lm.y * h) for lm in hand_lms.landmark]
                        x_min, x_max = min(x_list), max(x_list)
                        y_min, y_max = min(y_list), max(y_list)
                        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max),
                                      (0, 255, 0), 2)
                        hand_marks.append([(x_min + x_max) // 2,
                                           (y_min + y_max) // 2])
                ################################################

                ################################################
                # Call sim real-time callback to sync if provided
                real_data = {
                    "object_mask": best_mask,
                    "handler_mark": hand_marks
                }
                if self.sim_realtime_callback is not None:
                    self.sim_realtime_callback(real_data)
                ################################################

                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        cv2.destroyAllWindows()

    def list_camera_info(self):
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        if not devices:
            print("No RealSense device connected.")
            exit()

        device = devices[0]
        for sensor in device.query_sensors():
            print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")
            for profile in sensor.get_stream_profiles():
                if profile.is_video_stream_profile():
                    vsp = profile.as_video_stream_profile()
                    fmt = profile.format()
                    print(f"Stream: {vsp.stream_type()}, "
                          f"{vsp.width()}x{vsp.height()} @ {vsp.fps()} FPS, "
                          f"Format: {fmt.name}")

    def test_visualization(self):
        with MultiRealsense(enable_color=True,
                            enable_depth=True,
                            resolution=(640, 480),
                            capture_fps=30) as cameras:
            print(f"Number of cameras detected: {cameras.n_cameras}")
            print(f"Camera serials: {list(cameras.cameras.keys())}")

            while True:
                frames = cameras.get_vis()
                if frames is None or 'color' not in frames:
                    continue

                rgb_frames = frames['color']
                n_cameras = rgb_frames.shape[0]
                grid_size = int(np.ceil(np.sqrt(n_cameras)))
                h, w = rgb_frames.shape[1:3]
                grid = np.zeros((h * grid_size, w * grid_size, 3),
                                dtype=np.uint8)

                for idx in range(n_cameras):
                    i, j = idx // grid_size, idx % grid_size
                    grid[i * h:(i + 1) * h,
                         j * w:(j + 1) * w] = rgb_frames[idx]
                    serial = list(cameras.cameras.keys())[idx]
                    cv2.putText(grid, f"Camera {serial}",
                                (j * w + 10, i * h + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('MultiRealsense Cameras', grid)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()

    def test_recording(self):
        import pathlib
        output_dir = str(
            pathlib.Path(__file__).parent.parent / "data" / "tests" /
            "camera_recordings")
        os.makedirs(output_dir, exist_ok=True)

        with MultiRealsense(enable_color=True,
                            enable_depth=False,
                            resolution=(640, 480),
                            capture_fps=30) as cameras:
            print(f"Testing recording with {cameras.n_cameras} cameras")
            cameras.start_recording(output_dir, start_time=-1)
            print(f"Recording started. Saving to {output_dir}")

            start_time = time.time()
            recording_duration = 5  # seconds

            while time.time() - start_time < recording_duration:
                frames = cameras.get_vis()
                if frames is None or 'color' not in frames:
                    continue

                rgb_frames = frames['color']
                n_cameras = rgb_frames.shape[0]
                if n_cameras > 0:
                    preview = rgb_frames[0].copy()
                    cv2.circle(preview, (30, 30), 15, (0, 0, 255), -1)
                    remaining = recording_duration - (time.time() - start_time)
                    cv2.putText(preview, f"Recording: {remaining:.1f}s",
                                (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    cv2.imshow('Recording Preview', preview)
                    cv2.waitKey(1)

            cameras.stop_recording()
            print(f"Recording completed. Files saved to {output_dir}")
            cv2.destroyAllWindows()

    def cotracker_sam_select(self):
        hf_model = self.args.hf_model
        checkpoint = self.args.checkpoint
        sam_device = self.args.sam_device
        cot_device = self.args.cot_device
        width = self.args.width
        height = self.args.height
        fps = self.args.fps
        realsense_serial = self.args.realsense_serial
        alpha = self.args.alpha
        trace_length = self.args.trace_length
        point_stride = self.args.point_stride
        max_points = self.args.max_points

        try:
            cot_tracker = CoTrackerOnlineTracker(
                checkpoint=checkpoint,
                device=cot_device,
                grid_size=15,
                trace_length=trace_length,
            )
        except RuntimeError as e:
            print(str(e))
            return

        with MultiRealsense(
                enable_color=True,
                enable_depth=False,
                resolution=(width, height),
                capture_fps=fps,
                serial_numbers=[realsense_serial]
                if realsense_serial is not None else None,
        ) as cameras:
            if cameras.n_cameras == 0:
                print("No RealSense cameras detected.")
                return

            print(f"Using camera {list(cameras.cameras.keys())[0]}")
            first_bgr = None
            print(
                "Waiting 0.5 seconds for camera white balance to stabilize...")
            time.sleep(0.5)
            while first_bgr is None:
                frames_data = cameras.get_vis()
                if frames_data is not None and "color" in frames_data and frames_data[
                        "color"].shape[0] > 0:
                    first_bgr = frames_data["color"][0]

            segmenter = SAM2ImageSegmenter(hf_model_id=hf_model,
                                           device=sam_device)
            try:
                mask01, overlay = select_mask_on_first_frame(first_bgr,
                                                             segmenter,
                                                             alpha=alpha)
            except RuntimeError as e:
                print(str(e))
                return

            points_xy = sample_points_from_mask(mask01,
                                                stride=point_stride,
                                                max_points=max_points)
            if points_xy.shape[0] == 0:
                print("No points sampled from the selected mask. Aborting.")
                return

            t_col = np.zeros((points_xy.shape[0], 1), dtype=np.float32)
            queries_np = np.concatenate(
                [t_col, points_xy.astype(np.float32)], axis=1)
            queries = torch.from_numpy(queries_np[None,
                                                  ...]).float().to(cot_device)

            h, w = height, width
            tracking_view = np.zeros((h, w, 3), dtype=np.uint8)
            last_vis_time = 0.0

            window_name = "CoTracker Tracking (from SAM2 selection)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            while True:
                frames_data = cameras.get_vis()
                if (frames_data is None or "color" not in frames_data
                        or frames_data["color"].shape[0] == 0):
                    continue

                camera_view = frames_data["color"][0]
                pred_tracks, pred_visibility = cot_tracker.track_frame(
                    camera_view, queries=queries)

                if pred_tracks is not None:
                    vis_start = time.time()
                    video_chunk = cot_tracker.get_last_video_chunk()
                    video_chunk_len = video_chunk.shape[1]
                    tracks_chunk = pred_tracks[:, -video_chunk_len:]
                    visibility_chunk = pred_visibility[:, -video_chunk_len:]

                    if cot_tracker.point_colors is None:
                        num_points = pred_tracks.shape[2]
                        cot_tracker.point_colors = (
                            cm.get_cmap("gist_rainbow")(np.linspace(
                                0, 1, num_points))[:, :3] * 255).astype(
                                    np.uint8)

                    last_frame_with_tracks = draw_tracks_on_video(
                        video=video_chunk,
                        tracks=tracks_chunk,
                        visibility=visibility_chunk,
                        point_colors=cot_tracker.point_colors,
                        tracks_leave_trace=trace_length,
                    )
                    print(tracks_chunk.shape, visibility_chunk.shape)
                    tracking_view = cv2.cvtColor(last_frame_with_tracks,
                                                 cv2.COLOR_RGB2BGR)
                    last_vis_time = time.time() - vis_start

                stats_text = (
                    f"Model: {cot_tracker.last_model_time*1000:.1f}ms | Vis: {last_vis_time*1000:.1f}ms"
                )
                display_frame = tracking_view if cot_tracker.get_last_video_chunk(
                ) is not None else camera_view
                hh = display_frame.shape[0]
                cv2.putText(
                    display_frame,
                    stats_text,
                    (10, hh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        cv2.destroyAllWindows()

    def real_time_multi_streaming(self):
        hf_model = self.args.hf_model
        sam_device = self.args.sam_device
        width = self.args.width
        height = self.args.height
        fps = self.args.fps
        # Accept a list of serial numbers for up to 3 cameras
        realsense_serials = self.args.realsense_serial
        if isinstance(realsense_serials, str):
            realsense_serials = [
                s.strip() for s in realsense_serials.split(",") if s.strip()
            ]
        alpha = self.args.alpha

        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

        # --- Automatic mask selection using Grounding DINO and SAM2 ---
        from core.perception.sam.predictor import SAM2ImageSegmenter
        from groundingdino.util.inference import load_model, load_image, predict
        from torchvision.ops import box_convert
        import torch

        # Set up Grounding DINO and SAM2
        GROUNDING_DINO_CONFIG = "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT = "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
        BOX_THRESHOLD = 0.35
        TEXT_THRESHOLD = 0.25
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE,
        )

        def run_grounded_sam2(bgr_image, segmenter):
            obj = self.args.object_prompt
            TEXT_PROMPT = f"{obj}."
            cv2.imwrite("outputs/realsense_frame.png", bgr_image)
            image_source, image = load_image("outputs/realsense_frame.png")
            h, w, _ = image_source.shape

            boxes, confidences, labels = predict(
                model=grounding_model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes,
                                      in_fmt="cxcywh",
                                      out_fmt="xyxy").numpy()

            if len(input_boxes) == 0:
                input_boxes = [[w * 0.4, h * 0.4, w * 0.6, h * 0.6]]

            masks, scores, logits = segmenter.predict_from_box(
                np.array(input_boxes[0], dtype=np.float32))
            best_mask, _ = segmenter.get_best_mask(masks,
                                                   scores,
                                                   strategy="highest_score")
            best_mask = (best_mask > 0).astype(np.uint8)

            return best_mask

        # Support up to 3 cameras
        with MultiRealsense(
                enable_color=True,
                enable_depth=True,
                resolution=(width, height),
                capture_fps=fps,
                serial_numbers=realsense_serials
                if realsense_serials else None,
        ) as cameras:
            if cameras.n_cameras == 0:
                print("No RealSense cameras detected.")
                return

            print(f"Using cameras: {list(cameras.cameras.keys())}")
            first_bgrs = [None] * cameras.n_cameras

            print(
                "Waiting 0.5 seconds for camera white balance to stabilize...")
            time.sleep(0.5)
            while any(bgr is None for bgr in first_bgrs):
                frames_data = cameras.get_vis()
                if frames_data is not None and "color" in frames_data:
                    for idx in range(cameras.n_cameras):
                        if frames_data["color"].shape[0] > idx and first_bgrs[
                                idx] is None:
                            first_bgrs[idx] = frames_data["color"][idx]

            segmenters = [
                SAM2ImageSegmenter(hf_model_id=hf_model, device=sam_device)
                for _ in range(cameras.n_cameras)
            ]
            best_masks = []
            for idx in range(cameras.n_cameras):
                segmenters[idx].set_image(first_bgrs[idx])
                best_mask = run_grounded_sam2(first_bgrs[idx], segmenters[idx])
                best_masks.append(best_mask)

            overlays = [
                segmenters[idx].overlay_masks(first_bgrs[idx],
                                              best_masks[idx],
                                              alpha=alpha)
                for idx in range(cameras.n_cameras)
            ]

            window_names = [
                f"SAM2 Mask Tracking Camera {serial}"
                for serial in cameras.cameras.keys()
            ]
            for wn in window_names:
                cv2.namedWindow(wn, cv2.WINDOW_NORMAL)

            frame_count = 0
            while True:
                frame_count += 1
                frames_data = cameras.get_vis()
                if (frames_data is None or "color" not in frames_data
                        or frames_data["color"].shape[0] < cameras.n_cameras
                        or "depth" not in frames_data
                        or frames_data["depth"].shape[0] < cameras.n_cameras):
                    continue

                overlays = []
                real_data_list = []
                for idx in range(cameras.n_cameras):
                    camera_view = frames_data["color"][idx]
                    depth_view = frames_data["depth"][idx]
                    segmenters[idx].set_image(camera_view)

                    # Redo Grounding DINO every 5 frames
                    # if frame_count % 5 == 0 or best_masks[idx] is None:
                    if True:
                        best_masks[idx] = run_grounded_sam2(
                            camera_view, segmenters[idx])

                    # Sample points from previous mask and use as prompt
                    points = sample_points_from_mask(best_masks[idx],
                                                     stride=12,
                                                     max_points=10)
                    if points.shape[0] > 0:
                        labels = np.ones(points.shape[0], dtype=np.int32)
                        masks, scores, _ = segmenters[idx].predict_from_points(
                            points, labels, multimask_output=True)
                        best_masks[idx], _ = segmenters[idx].get_best_mask(
                            masks, scores, strategy="highest_score")
                        best_masks[idx] = (best_masks[idx]
                                           > 0).astype(np.uint8)

                    overlay = segmenters[idx].overlay_masks(camera_view,
                                                            best_masks[idx],
                                                            alpha=alpha)

                    # Hand detection and green rectangle drawing
                    rgb = cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    hand_marks = []
                    if results.multi_hand_landmarks:
                        h, w, _ = camera_view.shape
                        for hand_lms in results.multi_hand_landmarks:
                            x_list = [
                                int(lm.x * w) for lm in hand_lms.landmark
                            ]
                            y_list = [
                                int(lm.y * h) for lm in hand_lms.landmark
                            ]
                            x_min, x_max = min(x_list), max(x_list)
                            y_min, y_max = min(y_list), max(y_list)
                            cv2.rectangle(overlay, (x_min, y_min),
                                          (x_max, y_max), (0, 255, 0), 2)
                            hand_marks.append([(x_min + x_max) // 2,
                                               (y_min + y_max) // 2])

                    # Collect real_data for each camera
                    real_data = {
                        "color": camera_view,
                        "depth": depth_view,
                        "object_mask": best_masks[idx],
                        "handler_mark": hand_marks
                    }
                    real_data_list.append(real_data)
                    overlays.append(overlay)

                # Call sim real-time callback to sync if provided
                if self.sim_realtime_callback is not None:
                    self.sim_realtime_callback(real_data_list)

                # Show overlays for all cameras
                for idx, wn in enumerate(window_names):
                    cv2.imshow(wn, overlays[idx])
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            for wn in window_names:
                cv2.destroyWindow(wn)


def main(args: Args):
    camstream = CameraStreamer(args)
    print("1. Test visualization")
    print("2. Test recording")
    print("3. List camera info")
    print("4. CoTracker SAM Select")
    print("5. Real-time mask tracking")
    choice = input("Enter your choice (1, 2, 3, 4, or 5): ")
    args.realsense_serial = ["234222302175", "215122251521", "243122300947"]
    args.object_prompt = "rope"
    if choice == '1':
        camstream.test_visualization()
    elif choice == '2':
        camstream.test_recording()
    elif choice == '3':
        camstream.list_camera_info()
    elif choice == '4':
        camstream.cotracker_sam_select()
    elif choice == '5':
        camstream.real_time_multi_streaming()
    else:
        raise ValueError("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main(tyro.cli(Args))
