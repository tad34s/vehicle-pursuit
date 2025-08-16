import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch3d.io import IO
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix
from torchvision.io import read_image


def center_mesh(mesh: Meshes) -> Meshes:
    """
    Centers mesh in xz-plane and aligns lowest point to y=0.

    Steps:
    1. Compute bounding box center in x and z dimensions
    2. Translate mesh to center it in xz-plane
    3. Shift mesh so minimum y-coordinate becomes 0

    Args:
        mesh: Input PyTorch3D mesh (single or batched)

    Returns:
        Centered and floor-aligned mesh
    """
    verts_list = mesh.verts_list()
    new_verts_list = []

    for verts in verts_list:
        if len(verts) == 0:
            new_verts_list.append(verts)
            continue

        # Compute bounding box minima and maxima
        min_vals, _ = verts.min(dim=0)
        max_vals, _ = verts.max(dim=0)

        # Calculate center in xz-plane (ignore y)
        center_xz = torch.tensor(
            [
                (min_vals[0] + max_vals[0]) / 2.0,
                0,  # We'll handle y separately
                (min_vals[2] + max_vals[2]) / 2.0,
            ],
            device=verts.device,
            dtype=verts.dtype,
        )

        # Center mesh in xz-plane
        verts_centered = verts.clone()
        verts_centered[:, [0, 2]] -= center_xz[[0, 2]]

        # Align bottom to y=0
        min_y = verts_centered[:, 1].min()
        verts_centered[:, 1] -= min_y

        new_verts_list.append(verts_centered)

    return Meshes(verts=new_verts_list, faces=mesh.faces_list())


class Projector:
    CAMERA_POS = torch.tensor([0, 2, 2], dtype=torch.float32)
    CAMERA_ROT = torch.tensor([[25, 0, 0]], dtype=torch.float32)

    CAR_SIZES = [2.188, 1.273, 5.416]  # in x,y,z direction
    IMAGE_SIZE = (1080, 1080)

    FOCAL_LENGTH_MM = 50
    SENSOR_WIDTH_MM = 70

    def __init__(self, car_model_path: str, device=torch.device("cpu")) -> None:
        mesh = IO().load_mesh(car_model_path, device=device)
        scale, move_y = self.calc_scale_move(mesh)
        mesh.scale_verts_(scale)
        mesh = center_mesh(mesh)

        self.car_mesh = mesh
        self.device = device
        self.car_mesh.to(self.device)

        self.camera = self.create_camera()
        self.renderer = self.create_renderer()

    def create_camera(self) -> PerspectiveCameras:
        W, H = self.IMAGE_SIZE
        focal_len = (self.FOCAL_LENGTH_MM / self.SENSOR_WIDTH_MM) * W

        cam_R = axis_angle_to_matrix(math.pi * self.CAMERA_ROT / 180)
        T = -self.CAMERA_POS @ cam_R

        focal_length = torch.tensor(
            [focal_len, focal_len], device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        principal_point = torch.tensor(
            [self.IMAGE_SIZE[0] / 2, self.IMAGE_SIZE[1] / 2],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        camera = PerspectiveCameras(
            R=cam_R,
            T=T,
            focal_length=focal_length,
            principal_point=principal_point,
            in_ndc=False,
            image_size=[self.IMAGE_SIZE],
            device=self.device,
        )
        return camera

    def create_renderer(self):
        raster_settings = RasterizationSettings(
            image_size=self.IMAGE_SIZE,
            blur_radius=1e-6,
            faces_per_pixel=50,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.camera, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-6, gamma=1e-6)),
        )
        return renderer

    @classmethod
    def calc_scale_move(cls, pc: Meshes) -> tuple[float, float]:
        verts = pc.get_mesh_verts_faces(0)[0]

        z_coords = verts[:, 2]
        z_min = z_coords.min().item()
        z_max = z_coords.max().item()
        y_coords = verts[:, 1]
        y_min = y_coords.min().item()

        total_extent = abs(z_min) + abs(z_max)
        scale = cls.CAR_SIZES[2] / total_extent

        move = (y_min < 0) * -y_min
        return (scale, move)

    def move_car(self, x, y, theta):
        verts = self.car_mesh.verts_padded()

        # Create transformation matrix
        rot_matrix = axis_angle_to_matrix(math.pi * torch.tensor([0, -theta, 0]) / 180)
        # Apply transformation
        new_verts = verts @ rot_matrix.T + torch.tensor([x, 0, y], device=self.device)

        return self.car_mesh.update_padded(new_verts)

    def move_car_tensor(self, position: torch.Tensor):
        verts = self.car_mesh.verts_padded()

        # Extract components
        x = position[0]
        y = position[1]
        theta_deg = position[2]
        theta_rad = torch.deg2rad(theta_deg)
        zero = torch.zeros_like(theta_rad)

        rot_vec = torch.stack([zero, -theta_rad, zero]).unsqueeze(0)
        rot_matrix = axis_angle_to_matrix(rot_vec)  # (1, 3, 3)

        # Rotate vertices (differentiable)
        rotated_verts = torch.bmm(verts, rot_matrix.transpose(1, 2))

        # Create translation vector [x, 0, y] (differentiable)
        zero_trans = torch.zeros_like(x)
        translation = torch.stack([x, zero_trans, y]).unsqueeze(0).unsqueeze(0)  # (1, 1, 3)

        new_verts = rotated_verts + translation

        return self.car_mesh.update_padded(new_verts)

    def render_mask(self, x, y, theta):
        mesh = self.move_car(x, y, theta)
        silhouette = self.renderer(mesh, cameras=self.camera)

        silhouette_mask = silhouette[..., 3]
        mask = (silhouette_mask > 0.5).float()
        plt.imsave("out.png", mask[0].cpu().numpy(), cmap="gray")

    def calculate_mask(self, position: torch.Tensor) -> np.ndarray:
        mesh = self.move_car_tensor(position)
        silhouette = self.renderer(mesh, cameras=self.camera)
        return silhouette[..., 3]

    def loss(self, position: torch.Tensor, ref_image: torch.Tensor) -> torch.Tensor:
        ref_image = ref_image.to(self.device)
        if ref_image.dtype != torch.float32:
            ref_image = ref_image.float() / 255.0
        ref_image = (ref_image > 0.5).float()

        mask = projector.calculate_mask(position)
        ref_image = (ref_image > 0.5).float()
        loss = torch.sum((mask - ref_image) ** 2)
        return loss


if __name__ == "__main__":
    data_num = 107
    ref_image = read_image(f"dataset/output_masks/mask_{data_num}.png")
    t_ref: np.ndarray = np.load(f"dataset/t_ref/{data_num}.npy")
    t_ref_worse = t_ref.copy()
    t_ref_worse[1] = t_ref_worse[1] - 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projector = Projector("src/depth_net/utils/Prometheus.obj", device)
    position = torch.tensor(t_ref, device=device, dtype=torch.float32, requires_grad=True)
    position_worse = torch.tensor(
        t_ref_worse,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    print(position, position_worse)
    print("good estimate", projector.loss(position, ref_image))
    print("worse estimate", projector.loss(position_worse, ref_image))

    loss_value = projector.loss(position_worse, ref_image)

    # Compute gradients
    loss_value.backward()
    print("Loss:", loss_value.item())
    print("Gradient:", position.grad)
