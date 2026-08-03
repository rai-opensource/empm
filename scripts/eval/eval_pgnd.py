import argparse
from pathlib import Path

import torch
import pyvista as pv
import pickle
import numpy as np

PGND_FRAME_COUNT = 27


def run(
    reference_data_path: Path,
    pgnd_dir: Path,
    output_dir: Path,
    frame_count: int = PGND_FRAME_COUNT,
):
    with reference_data_path.open("rb") as f:
        data = pickle.load(f)
        print(data.shape)

    points = []
    for i in range(frame_count):
        point = torch.load(pgnd_dir / f"{i:04d}.pt")
        point = point["x"].cpu().numpy()
        point = point[:, [0, 2, 1]]
        point[:, 1] = -point[:, 1]
        point[:, 2] = -point[:, 2]
        points.append(point)
        points.append(point)
        points.append(point)
        points.append(point)
        points.append(point)

    for _ in range(20):
        points.append(points[-1])

    points = np.asarray(points)
    points = points - np.mean(points[0], axis=0, keepdims=True) + np.mean(
        data[0], axis=0, keepdims=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(np.min(data[0], axis=0), np.max(data[0], axis=0))
    print(np.min(points[0], axis=0), np.max(points[0], axis=0))
    point_cloud = pv.PolyData(points[0])
    point_cloud.save(str(output_dir / "pgnd.ply"), binary=False)
    point_cloud = pv.PolyData(data[0])
    point_cloud.save(str(output_dir / "data.ply"), binary=False)

    print(points.shape)  # (T, N, 3)

    with (output_dir / "inference_pgnd.pkl").open("wb") as f:
        pickle.dump(points, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PGND evaluation output.")
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--pgnd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=PGND_FRAME_COUNT)
    args = parser.parse_args()
    run(
        reference_data_path=args.reference_data,
        pgnd_dir=args.pgnd_dir,
        output_dir=args.output_dir,
        frame_count=args.frame_count,
    )
