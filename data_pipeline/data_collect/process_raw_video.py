import os
import shutil


def run(cam_idx=0, exp_name = "none"):
    """ 0: 243122300947
        1: 234222302175
        2: 215122251521
        -> 23, 21, 24
    """
    serial_list = ["234222302175", "215122251521", "243122300947"]
    serial = serial_list[cam_idx]
    raw_dir = f"videos_raw_pita_1/{serial}"
    tmp_dir = "tmp"
    # out_dir = f"videos/{serial}"
    out_dir = f"data_custom/pet/data/different_types/{exp_name}"

    os.system(f"rm -rf {tmp_dir}")
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"ffmpeg -i {raw_dir}/color.mp4 {tmp_dir}/%04d.png")
    os.makedirs(f"{out_dir}/color/{cam_idx}", exist_ok=True)
    os.makedirs(f"{out_dir}/depth/{cam_idx}", exist_ok=True)
    for f in range(165):
        src_index = 60 + f * 1
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

    os.system(
        f"ffmpeg -y -framerate 30 -i {out_dir}/color/{cam_idx}/%d.png -c:v libx264 -pix_fmt yuv420p {out_dir}/color/{cam_idx}.mp4"
    )


if __name__ == "__main__":
    
    cam_ids = [0, 1, 2]
    exp_name = "double_tear_pita_test"
    for i in cam_ids:
        run(cam_idx=i, exp_name=exp_name)