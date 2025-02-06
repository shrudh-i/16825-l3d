"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from starter import utils
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

import imageio
import mcubes

def render_torus(image_size=256, num_samples=200, device=None):
    if device is None:
        device = get_device()
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)
    r0 = 1
    r1 = 0.5
    x = (r0+r1*torch.cos(Theta))*torch.cos(Phi)
    y = (r0+r1*torch.cos(Theta))*torch.sin(Phi)
    z = r1*torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    my_images =[]
    
    for i in range(0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=i)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        renderer = utils.get_points_renderer(image_size=image_size, device=device)
        rend = renderer(torus_point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = (rend * 255).astype("uint8")
        my_images.append(rend)

    return my_images

def render_torus_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -3.1
    max_value = 3.1
    r0 = 1
    r1 = 0.5
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (X ** 2 + Y ** 2 + Z ** 2 + r0 ** 2 - r1 ** 2) ** 2 - 4 * r0 ** 2 * (X ** 2 + Y ** 2)
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    imgs =[]
    for i in range(-180,180,4):
        lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
        renderer = utils.get_mesh_renderer(image_size=image_size, device=device)
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=i)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        imgs.append(rend[0, ..., :3].detach().cpu().numpy().clip(0, 1))

    imgs = np.array(imgs)
    imgs = (imgs * 255).astype(np.uint8)
    
    return imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="images/torus_360.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    # my_images = render_torus(image_size=args.image_size)
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    # imageio.mimsave(args.output_path, my_images, duration=duration, loop=0)

    images_2 = render_torus_mesh(image_size=args.image_size)
    imageio.mimsave("images/torus_mesh.gif", images_2, duration=duration, loop=0)
