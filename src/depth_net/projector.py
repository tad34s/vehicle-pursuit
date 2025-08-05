import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch3d.io import IO

# Util function for loading point clouds|
# Data structures and functions for rendering
from pytorch3d.renderer import (
    PerspectiveCameras,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import axis_angle_to_matrix


class Projector:
    CAMERA_POS = torch.tensor([[0, 2, 2]], dtype=torch.float32)
    CAMERA_ROT = torch.tensor([[25, 0, 0]], dtype=torch.float32)

    CAR_SIZES = [2.188, 1.273, 5.416]  # in x,y,z direction
    IMAGE_SIZE = (1080, 1080)

    def __init__(self, car_model_path: str, device=torch.device("cpu")) -> None:
        pc = IO().load_pointcloud(car_model_path, device)
        scale, move_y = self.calc_scale_move(pc)
        print(move_y)
        pc.offset_(torch.tensor([0, move_y, 0], device=device))

        if pc.features_packed() is None:
            feats_list = [torch.ones((len(p), 3), device=p.device) for p in pc.points_list()]
            pc = Pointclouds(
                points=pc.points_list(),
                features=feats_list,
            )

        self.car_pc = pc.scale(scale)

        self.device = device
        self.car_pc.to(self.device)

        self.camera = self.create_camera()

    def create_camera(self) -> PerspectiveCameras:
        focal_length_mm = 50
        sensor_width_mm = 70
        W, H = self.IMAGE_SIZE
        focal_len = (focal_length_mm * W) / sensor_width_mm

        cam_R = axis_angle_to_matrix(math.pi * self.CAMERA_ROT / 180)
        cam_T = self.CAMERA_POS * -1

        focal_length = torch.tensor(
            [focal_len, focal_len], device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        principal_point = torch.tensor(
            [self.IMAGE_SIZE[0] / 2, self.IMAGE_SIZE[1] / 2],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        cameras = PerspectiveCameras(
            R=cam_R,
            T=cam_T,
            focal_length=focal_length,
            principal_point=principal_point,
            in_ndc=False,
            image_size=self.IMAGE_SIZE,
            device=self.device,
        )
        return cameras

    @classmethod
    def calc_scale_move(cls, pc: Pointclouds) -> tuple[float, float]:
        points = pc.get_cloud(0)[0]

        z_coords = points[:, 2]
        z_min = z_coords.min().item()
        z_max = z_coords.max().item()
        y_coords = points[:, 1]
        y_min = y_coords.min().item()

        total_extent = abs(z_min) + abs(z_max)
        scale = cls.CAR_SIZES[2] / total_extent

        move = (y_min < 0) * -y_min
        return (scale, move)

    def project_car(self, x, y, theta):
        # TODO: rotate
        car_world_points = self.car_pc.offset(
            torch.tensor(
                [x, 0, y],
                device=self.device,
                dtype=torch.float32,
            )
        )

        # Project to screen coordinates
        screen_points = self.camera.transform_points_screen(
            car_world_points.points_packed(), image_size=self.IMAGE_SIZE
        )

        return screen_points

    def display_points(self, screen_points):
        screen_points_batch = screen_points[0]  # Remove batch dim -> (N, 3)
        x_screen = screen_points_batch[:, 0]  # X coordinates (pixels)
        y_screen = screen_points_batch[:, 1]  # Y coordinates (pixels)
        z_depth = screen_points_batch[:, 2]  # Depth values

        # Keep only points in front of the camera (z > 0)
        valid = z_depth > 0
        x_screen = x_screen[valid]
        y_screen = y_screen[valid]

        # 3. Get image dimensions and clamp coordinates
        H, W = self.IMAGE_SIZE  # Image height and width
        x_screen = torch.clamp(x_screen, 0, W - 1)  # Clamp X to [0, W-1]
        y_screen = torch.clamp(y_screen, 0, H - 1)  # Clamp Y to [0, H-1]

        x_idx = torch.round(x_screen).to(torch.long)
        y_idx = torch.round(y_screen).to(torch.long)

        mask = torch.zeros((H, W), dtype=torch.float32)  # Black image (all zeros)
        mask[y_idx, x_idx] = 1.0  # Set projected pixels to white

        # 6. Visualize the silhouette
        plt.figure(figsize=(10, 10))
        plt.imshow(mask.cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.title("Projected Point Cloud Silhouette")
        plt.show()

    def render_mask(self, x, y, theta):
        screen_points = self.project_car(x, y, theta)
        self.display_points(screen_points)


if __name__ == "__main__":
    device = torch.device("cuda")
    projector = Projector("src/depth_net/utils/car.ply", device)
    t_ref = np.load("dataset/t_ref/0.npy")
    print(t_ref)
    # projector.render_mask(t_ref[0], t_ref[1], t_ref[2])
    projector.render_mask(t_ref[0], t_ref[1], t_ref[2])
