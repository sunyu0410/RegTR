import sys
sys.path.append('../src')
import numpy as np
import cvhelpers.visualization as cvv
import torch

import pickle
from matplotlib.pyplot import cm as colormap
import cvhelpers.colors as colors
from utils.se3_torch import se3_transform


def visualize_result(src_xyz: np.ndarray, tgt_xyz: np.ndarray,
                     src_kp: np.ndarray, src2tgt: np.ndarray,
                     src_overlap: np.ndarray,
                     pose: np.ndarray,
                     threshold: float = 0.5):
    """Visualizes the registration result:
       - Top-left: Source point cloud and keypoints
       - Top-right: Target point cloud and predicted corresponding kp positions
       - Bottom-left: Source and target point clouds before registration
       - Bottom-right: Source and target point clouds after registration

    Press 'q' to exit.

    Args:
        src_xyz: Source point cloud (M x 3)
        tgt_xyz: Target point cloud (N x 3)
        src_kp: Source keypoints (M' x 3)
        src2tgt: Corresponding positions of src_kp in target (M' x 3)
        src_overlap: Predicted probability the point lies in the overlapping region
        pose: Estimated rigid transform
        threshold: For clarity, we only show predicted overlapping points (prob > 0.5).
                   Set to 0 to show all keypoints, and a larger number to show
                   only points strictly within the overlap region.
    """

    large_pt_size = 4
    color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('coolwarm'))
    overlap_colors = (color_mapper.to_rgba(src_overlap[:, 0])[:, :3] * 255).astype(np.uint8)
    m = src_overlap[:, 0] > threshold

    vis = cvv.Visualizer(
        win_size=(1600, 1000),
        num_renderers=4)

    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.RED),
        renderer_idx=0
    )
    vis.add_object(
        cvv.create_point_cloud(src_kp[m, :], colors=overlap_colors[m, :], pt_size=large_pt_size),
        renderer_idx=0
    )

    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=1
    )
    vis.add_object(
        cvv.create_point_cloud(src2tgt[m, :], colors=overlap_colors[m, :], pt_size=large_pt_size),
        renderer_idx=1
    )

    # Before registration
    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.RED),
        renderer_idx=2
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=2
    )

    # After registration
    vis.add_object(
        cvv.create_point_cloud(se3_transform(pose, src_xyz), colors=colors.RED),
        renderer_idx=3
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=3
    )

    vis.set_titles(['Source point cloud (with keypoints)',
                    'Target point cloud (with predicted source keypoint positions)',
                    'Before Registration',
                    'After Registration'])

    vis.reset_camera()
    vis.start()



args = pickle.load(open('../dev/args.pkl', 'rb'))
args = [torch.from_numpy(i).double() for i in args]
visualize_result(*args, threshold=0.5)
