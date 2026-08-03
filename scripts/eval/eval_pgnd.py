import torch
import pyvista as pv
import pickle
import numpy as np

PGND_FRAME_COUNT = 27


def run():

    with open(
            f"data_custom/ours/experiments/double_stretch_dough_test/inference.pkl",
            "rb") as f:
        data = pickle.load(f)
        print(data.shape)

    points = []
    for i in range(PGND_FRAME_COUNT):
        point = torch.load(f"pgnd_dough/{i:04d}.pt")
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

    points = points - np.mean(points[0], axis=0, keepdims=True) + np.mean(
        data[0], axis=0, keepdims=True)

    print(np.min(data[0], axis=0), np.max(data[0], axis=0))
    print(np.min(points[0], axis=0), np.max(points[0], axis=0))
    point_cloud = pv.PolyData(points[0])
    point_cloud.save(f"episode_0000/pgnd.ply", binary=False)
    point_cloud = pv.PolyData(data[0])
    point_cloud.save(f"episode_0000/data.ply", binary=False)

    print(points.shape)  # (T, N, 3)

    with open(
            f"data_custom/ours/experiments/double_stretch_dough_test/inference_pgnd.pkl",
            "wb") as f:
        pickle.dump(points, f)


run()
