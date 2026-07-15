import os
import sys
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
if str(Path("./Grounded-SAM-2").resolve()) not in sys.path:
    sys.path.append(str(Path("./Grounded-SAM-2").resolve()))
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
import json
from argparse import ArgumentParser


def segment_image(img_path: str, output_path: str, text_prompt: str):
    """
        image segmentation with GroundedSAM2
    """
    # hyperparam for GroundedSAM2
    SAM2_CHECKPOINT = "./data_pipeline/data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = (
        "./data_pipeline/data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
    )
    GROUNDING_DINO_CHECKPOINT = (
        "./data_pipeline/data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
    )
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )


    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    image_source, image = load_image(img_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    # FIXME: figure how does this influence the G-DINO model
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = confidences.numpy().tolist()
    class_names = labels

    OBJECTS = class_names

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS)}

    print(f"Detected {len(masks)} objects")

    raw_img = cv2.imread(img_path)
    mask_img = (masks[0] * 255).astype(np.uint8)

    ref_img = np.zeros((h, w, 4), dtype=np.uint8)
    mask_bool = mask_img > 0
    ref_img[mask_bool, :3] = raw_img[mask_bool]
    ref_img[:, :, 3] = mask_bool.astype(np.uint8) * 255
    cv2.imwrite(output_path, ref_img)


def segment_video(base_path: str, case_name: str, text_prompt: str, camera_idx: int, output_path: str):
    """
        video segmentation with GroundedSAM2
    """
    output_path = f"{base_path}/{case_name}" if output_path is None else output_path

    # hyperparam for GroundedSAM2
    GROUNDING_DINO_CONFIG = "./data_pipeline/data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "./data_pipeline/data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    PROMPT_TYPE_FOR_VIDEO = "box"  # choose from ["point", "box", "mask"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    VIDEO_PATH = f"{base_path}/{case_name}/color/{camera_idx}.mp4"
    os.makedirs(f"{base_path}/{case_name}/tmp_data", exist_ok=True)
    os.makedirs(f"{base_path}/{case_name}/tmp_data/{case_name}", exist_ok=True)
    os.makedirs(f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}", exist_ok=True)

    SOURCE_VIDEO_FRAME_DIR = f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}"

    """
    Step 1: Environment settings and model initialization for Grounding DINO and SAM 2
    """
    # build grounding dino model from local path
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )

    # init sam image predictor and video predictor model
    sam2_checkpoint = "./data_pipeline/data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)


    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
    print(video_info)
    frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

    # saving video to frames
    source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
    source_frames.mkdir(parents=True, exist_ok=True)

    with sv.ImageSink(
        target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p
        for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

    ann_frame_idx = 0  # the frame index we interact with
    """
    Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
    """

    # prompt grounding dino to get the box coordinates on specific frame
    img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
    image_source, image = load_image(img_path)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    confidences = confidences.numpy().tolist()
    class_names = labels

    print(input_boxes)

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(image_source)

    # process the detection results
    OBJECTS = class_names

    print(OBJECTS)

    # FIXME: figure how does this influence the G-DINO model
    # torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    torch.autocast(device_type=DEVICE, dtype=torch.float32).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Step 3: Register each object's positive points to video predictor with seperate add_new_points call
    """

    assert PROMPT_TYPE_FOR_VIDEO in [
        "point",
        "box",
        "mask",
    ], "SAM 2 video predictor only support point/box/mask prompt"

    if PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes)):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    else:
        raise NotImplementedError(
            "SAM 2 video predictor only support point/box/mask prompts"
        )

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    """
    Step 5: Visualize the segment results across the video and save them
    """

    os.makedirs(f"{output_path}/mask/", exist_ok=True)
    os.makedirs(f"{output_path}/mask/{camera_idx}", exist_ok=True)

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS)}

    # Save the id_to_objects into json
    with open(f"{output_path}/mask/mask_info_{camera_idx}.json", "w") as f:
        json.dump(ID_TO_OBJECTS, f)

    for frame_idx, masks in video_segments.items():
        for obj_id, mask in masks.items():
            os.makedirs(f"{output_path}/mask/{camera_idx}/{obj_id}", exist_ok=True)
            # mask is 1 * H * W
            Image.fromarray((mask[0] * 255).astype(np.uint8)).save(
                f"{output_path}/mask/{camera_idx}/{obj_id}/{frame_idx}.png"
            )