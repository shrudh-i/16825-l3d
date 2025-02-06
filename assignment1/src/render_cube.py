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


def render_cube(image_size=256, color=[0.7, 0.7, 1], device=None):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    # Define vertices and faces for a tetrahedron
    vertices = torch.tensor([
        [-1, -1, -1],  # Vertex 0
        [1, -1, -1],   # Vertex 1
        [1, 1, -1],    # Vertex 2
        [-1, 1, -1],   # Vertex 3
        [-1, -1, 1],   # Vertex 4
        [1, -1, 1],    # Vertex 5
        [1, 1, 1],     # Vertex 6
        [-1, 1, 1]     # Vertex 7
    ], dtype=torch.float32)
    
    faces = torch.tensor([
        [0, 1, 2], [2, 3, 0],  # Front face
        [1, 5, 6], [6, 2, 1],  # Right face
        [5, 4, 7], [7, 6, 5],  # Back face
        [4, 0, 3], [3, 7, 4],  # Left face
        [3, 2, 6], [6, 7, 3],  # Top face
        [4, 5, 1], [1, 0, 4]   # Bottom face
    ], dtype=torch.int64)

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    my_images = []

    for i in range(360):
        # print(f"this is i: {i}")
        R, T = pytorch3d.renderer.look_at_view_transform(dist=4, elev=0, azim=i)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -4]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = (rend * 255).astype("uint8")
        # The .cpu moves the tensor to GPU (if needed).
        # return rend

        my_images.append(rend)

    return my_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cube_360.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    my_images = render_cube(image_size=args.image_size)
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(args.output_path, my_images, duration=duration, loop=0)
