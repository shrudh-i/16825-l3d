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


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    # textures = torch.ones_like(vertices)  # (1, N_v, 3)
    # textures = textures * torch.tensor(color)  # (1, N_v, 3)
    textures = vertices.clone()

    # z_min & z_max
    z_min = textures[:,:,2].min()
    z_max = textures[:,:,2].max()

    # normalize the z coordinates (front & back)
    textures = (textures - z_min) / (z_max - z_min)
    
    # gray scale all vertices
    textures[:,:,0] = textures[:,:,2]
    textures[:,:,1] = textures[:,:,2]

    # color1 = torch.tensor([0, 0.5, 1])
    color2 = torch.tensor([0.5, 0, 0.5])
    color1 = torch.tensor([0, 1, 0])
    textures = textures * color2 + (1 - textures) * color1

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # #############################################3
    my_images = []

    for i in range(0,360,10):
        # print(f"this is i: {i}")
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=i)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = (rend * 255).astype("uint8")
        # The .cpu moves the tensor to GPU (if needed).
        # return rend

        my_images.append(rend)

    return my_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_retexture.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    my_images = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(args.output_path, my_images, duration=duration, loop=0)
