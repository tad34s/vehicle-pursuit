import torch
from pytorch3d.io import IO
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds


def fbx_to_pytorch_pointcloud(
    fbx_path: str, num_points: int = 1000, device: str = "cpu"
) -> Pointclouds:
    """
    Reads an FBX file, converts its mesh data to a PyTorch3D Meshes object,
    samples points from the mesh, and returns a PyTorch3D Pointclouds object.

    Args:
        fbx_path (str): Path to the input.fbx file.
        num_points (int): Number of points to sample from the mesh.
                          Higher values result in denser point clouds.
        device (str): The device to store PyTorch tensors on ('cpu' or 'cuda').

    Returns:
        Pointclouds: A PyTorch3D Pointclouds object containing the sampled points,
                     normals, and colors (if available from the mesh).
    """
    if not torch.cuda.is_available():
        print(
            "CUDA not available, using CPU. For better performance, consider a CUDA-enabled device."
        )
        device = "cpu"
    else:
        print(f"Using device: {device}")

    mesh = IO().load_mesh(fbx_path, device=device)

    sampled_points = sample_points_from_meshes(mesh, num_samples=num_points)

    point_cloud = Pointclouds(points=sampled_points)

    return point_cloud


if __name__ == "__main__":
    cloud = fbx_to_pytorch_pointcloud("Prometheus.obj")
    IO().save_pointcloud(cloud, "car.ply")
