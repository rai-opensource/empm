import os
import shutil
import subprocess
from pathlib import Path

FRAME_COUNT = 165
SOURCE_START_FRAME = 60


def run(cam_idx=0, exp_name = "none"):
    """ 0: 243122300947
        1: 234222302175
        2: 215122251521
        -> 23, 21, 24
    """
    serial_list = ["234222302175", "215122251521", "243122300947"]
    serial = serial_list[cam_idx]
    raw_dir = f"videos_raw_pita_1/{serial}"
    tmp_dir = Path("tmp")
    # out_dir = f"videos/{serial}"
    out_dir = f"data_custom/pet/data/different_types/{exp_name}"

    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(Path(raw_dir) / "color.mp4"),
            str(tmp_dir / "%04d.png"),
        ],
        check=True,
    )
    os.makedirs(f"{out_dir}/color/{cam_idx}", exist_ok=True)
    os.makedirs(f"{out_dir}/depth/{cam_idx}", exist_ok=True)
    for f in range(FRAME_COUNT):
        src_index = SOURCE_START_FRAME + f
        src_file = os.path.join(tmp_dir, f"{src_index:04d}.png")
        dst_file = os.path.join(out_dir, f"color/{cam_idx}/{f}.png")
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} → {dst_file}")
        else:
            print(f"Skipped (not found): {src_file}")

        src_file = os.path.join(raw_dir, f"depth/{src_index:05d}.npy")
        dst_file = os.path.join(out_dir, f"depth/{cam_idx}/{f}.npy")
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} → {dst_file}")
        else:
            print(f"Skipped (not found): {src_file}")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "30",
            "-i",
            str(Path(out_dir) / "color" / str(cam_idx) / "%d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(Path(out_dir) / "color" / f"{cam_idx}.mp4"),
        ],
        check=True,
    )


if __name__ == "__main__":
    
    cam_ids = [0, 1, 2]
    exp_name = "double_tear_pita_test"
    for i in cam_ids:
        run(cam_idx=i, exp_name=exp_name)
