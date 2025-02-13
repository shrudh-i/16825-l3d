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

def render_cone(image_size=256, num_samples=200, device=None):
    if device is None:
        device = get_device()

    # Create theta and height (z) values
    theta = torch.linspace(0, 2 * np.pi, num_samples)  # Angle around the circle
    z = torch.linspace(0, 2, num_samples)  # Height of the cone

    # Generate the cone in cylindrical coordinates
    r0 = 1  # Base radius
    h = 2  # Height
    r = (r0 * (1 - z / h))  # Radius at each height level
    x = r.unsqueeze(1) * torch.cos(theta).unsqueeze(0)  # Calculate x coordinates for each height level
    y = r.unsqueeze(1) * torch.sin(theta).unsqueeze(0)  # Calculate y coordinates for each height level
    z = z.unsqueeze(1).expand_as(x)  # Broadcast z across the x and y coordinates

    # Flatten the coordinates and stack them into a point cloud
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())  # Normalize colors based on point positions

    # Create point cloud structure
    cone_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    # List to store rendered images
    my_images = []

    # Render the cone from different camera angles
    for i in range(0, 360, 10):  # Loop to create images from different azimuth angles
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=i)

        # Set up camera and lighting
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        # Render the point cloud
        renderer = utils.get_points_renderer(image_size=image_size, device=device)
        rend = renderer(cone_point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]  # Get RGB values only
        rend = (rend * 255).astype("uint8")
        my_images.append(rend)

    return my_images

def render_cone_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()

    min_value = -2
    max_value = 2
    r0 = 1  # Radius of the base of the cone
    r1 = 0  # Top radius (0 for a pointed cone)
    height = 1  # Shorter height of the cone (adjust this value)

    # Create a grid of points
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

    # Define the equation for the cone with the new shorter height
    r = ((r0 - r1) * (Z - min_value) / (max_value - min_value)) + r1
    voxels = (X ** 2 + Y ** 2) - r ** 2  # Condition for points inside the cone

    # Apply marching cubes algorithm to generate mesh
    vertices, faces = mcubes.marching_cubes(voxels.cpu().numpy(), isovalue=0)

    # Convert vertices and faces to tensors
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Normalize vertex coordinates to fit the grid
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value

    # Texture the vertices based on their position
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    # Create the mesh from the vertices, faces, and textures
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)

    # List to store rendered images
    imgs = []

    # Render the cone from different camera angles
    for i in range(-180, 180, 4):  # Loop through angles for full 360-degree rotation
        lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)
        
        # Get the mesh renderer from utils
        renderer = utils.get_mesh_renderer(image_size=image_size, device=device)

        # Set up camera transformations
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=i)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

        # Render the scene with lighting
        rend = renderer(mesh, cameras=cameras, lights=lights)

        # Append the rendered image to the list, cropping to [0, 1] range
        imgs.append(rend[0, ..., :3].detach().cpu().numpy().clip(0, 1))

    imgs = np.array(imgs)  # Convert to a numpy array
    imgs = (imgs * 255).astype(np.uint8)  # Convert to uint8 for image saving

    return imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="images/torus_360.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    # my_images = render_torus(image_size=args.image_size)
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    # imageio.mimsave(args.output_path, my_images, duration=duration, loop=0)

    images_2 = render_cone_mesh(image_size=args.image_size)
    imageio.mimsave("temp/cone_mesh.gif", images_2, duration=duration, loop=0)
