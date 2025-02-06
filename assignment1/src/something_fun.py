"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

import imageio

def render_fun_cow(
    cow_path="data/cow.obj", image_size=256, device=None,
):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = vertices.clone()

    # Normalize each coordinate axis (x, y, z)
    x_min, x_max = textures[:, :, 0].min(), textures[:, :, 0].max()
    y_min, y_max = textures[:, :, 1].min(), textures[:, :, 1].max()
    z_min, z_max = textures[:, :, 2].min(), textures[:, :, 2].max()

    textures[:, :, 0] = (textures[:, :, 0] - x_min) / (x_max - x_min)  # Normalize x
    textures[:, :, 1] = (textures[:, :, 1] - y_min) / (y_max - y_min)  # Normalize y
    textures[:, :, 2] = (textures[:, :, 2] - z_min) / (z_max - z_min)  # Normalize z

    # Define multiple colors
    color_x = torch.tensor([1, 0, 0])  # Red
    color_y = torch.tensor([0, 1, 0])  # Green
    color_z = torch.tensor([0, 0, 1])  # Blue

    # Blend colors based on the normalized coordinates
    textures = textures[:, :, 0:1] * color_x + textures[:, :, 1:2] * color_y + textures[:, :, 2:3] * color_z

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # #############################################
    my_images = []

    for i in range(0, 360, 10):
        # Add floating effect by modifying the T translation
        float_offset = 0.5 * torch.sin(torch.tensor(i * 3.14159 / 180))
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=i)
        T[:, 1] += float_offset  # Move cow up and down explicitly

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = (rend * 255).astype("uint8")

        my_images.append(rend)

    return my_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_fun.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    my_images = render_fun_cow(cow_path=args.cow_path, image_size=args.image_size)

    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(args.output_path, my_images, duration=duration, loop=0)