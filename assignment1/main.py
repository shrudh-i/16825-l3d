import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
import imageio
import mcubes

from starter import utils
from src import render_mesh_360, dolly_zoom, render_tetra, render_cube, retexture_mesh, camera_transforms, render_generic, something_fun, render_torus

# This should print True if you are using your GPU
print("Using GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

############### 1. Practicing with Cameras ###############

# 1.1. 360-degree Renders (5 points)
parser = argparse.ArgumentParser()
parser.add_argument("--cow_path", type=str, default="data/cow.obj")
parser.add_argument("--output_path", type=str, default="submission/cow_360.gif")
parser.add_argument("--image_size", type=int, default=256)
args = parser.parse_args()
my_images = render_mesh_360.render_cow(cow_path=args.cow_path, image_size=args.image_size)
# Convert FPS (frames per second) to duration (ms per frame)
duration = 1000 // 15  
imageio.mimsave(args.output_path, my_images, duration=duration, loop=0)
print("Completed 1.1")

# 1.2 Re-creating the Dolly Zoom (10 points)
parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=10)
parser.add_argument("--duration", type=float, default=3)
parser.add_argument("--output_file", type=str, default="submission/dolly.gif")
parser.add_argument("--image_size", type=int, default=256)
args = parser.parse_args()
dolly_zoom.dolly_zoom(
    image_size=args.image_size,
    num_frames=args.num_frames,
    duration=args.duration,
    output_file=args.output_file,
)
print("Completed 1.2")

############### 2. Practicing with Meshes ###############

# 2.1 Constructing a Tetrahedron (5 points)
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="submission/tetra_360.gif")
parser.add_argument("--image_size", type=int, default=256)
args = parser.parse_args()
my_images = render_tetra.render_tetra(image_size=args.image_size)
duration = 1000 // 15  
imageio.mimsave(args.output_path, my_images, duration=duration, loop=0)
print("Completed 2.1")

# 2.2 Constructing a Cube (5 points)
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="submission/cube_360.gif")
parser.add_argument("--image_size", type=int, default=256)
args = parser.parse_args()
my_images = render_cube.render_cube(image_size=args.image_size)
duration = 1000 // 15  
imageio.mimsave(args.output_path, my_images, duration=duration, loop=0)
print("Completed 2.2")

############### 3. Re-texturing a mesh ###############
parser = argparse.ArgumentParser()
parser.add_argument("--cow_path", type=str, default="data/cow.obj")
parser.add_argument("--output_path", type=str, default="submission/cow_retexture.gif")
parser.add_argument("--image_size", type=int, default=256)
args = parser.parse_args()
my_images = retexture_mesh.render_cow(cow_path=args.cow_path, image_size=args.image_size)
duration = 1000 // 15  
imageio.mimsave(args.output_path, my_images, duration=duration, loop=0)
print("Completed 3")

############### 4. Camera Transformations ###############
## image 1
R_relative=[[np.cos(np.pi/2), np.sin(np.pi/2), 0], [-np.sin(np.pi/2), np.cos(np.pi/2), 0], [0, 0, 1]]
T_relative=[0, 0, 0]
image1 = camera_transforms.render_textured_cow(cow_path="data/cow_with_axis.obj", R_relative = R_relative, T_relative=T_relative)
plt.imsave("submission/cow_trans1.jpg", image1)

## image 2
R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
T_relative=[0, 0, 2]
image2 = camera_transforms.render_textured_cow(cow_path="data/cow_with_axis.obj",R_relative = R_relative, T_relative=T_relative)
plt.imsave("submission/cow_trans2.jpg", image2)

## image 3
R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
T_relative=[0.5, -0.5, 0]
image3 = camera_transforms.render_textured_cow(cow_path="data/cow_with_axis.obj",R_relative = R_relative, T_relative=T_relative)
plt.imsave("submission/cow_trans3.jpg", image3)

## image 4
R_relative=[[np.cos(np.pi/2), 0, np.sin(np.pi/2)], [0, 1, 0], [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]]
T_relative=[-3, 0, 3]
image4 = camera_transforms.render_textured_cow(cow_path="data/cow_with_axis.obj",R_relative = R_relative, T_relative=T_relative)
plt.imsave("submission/cow_trans4.jpg", image4)

print("Completed 4")

############### 5. Rendering Generic 3D Representations ###############

# 5.1 Rendering Point Clouds from RGB-D Images (10 points)
num_views = 24
lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
data_dict = render_generic.load_rgbd_data()

points1, rgb1 = utils.unproject_depth_image(torch.Tensor(data_dict["rgb1"]), 
                                torch.Tensor(data_dict["mask1"]), 
                                torch.Tensor(data_dict["depth1"]), 
                                data_dict["cameras1"])
pc1 = pytorch3d.structures.Pointclouds(
    points=points1.unsqueeze(0),
    features=rgb1.unsqueeze(0),
).to(device)

points2, rgb2 = utils.unproject_depth_image(torch.Tensor(data_dict["rgb2"]), 
                                torch.Tensor(data_dict["mask2"]), 
                                torch.Tensor(data_dict["depth2"]), 
                                data_dict["cameras2"])
pc2 = pytorch3d.structures.Pointclouds(
    points=points2.unsqueeze(0),
    features=rgb2.unsqueeze(0),
).to(device)

pc3 = pytorch3d.structures.Pointclouds(
    points=torch.cat((points1,points2), 0).unsqueeze(0),
    features=torch.cat((rgb1,rgb2), 0).unsqueeze(0),
).to(device)

R0 = torch.tensor([[float(np.cos(np.pi)), float(-np.sin(np.pi)), 0.], [float(np.sin(np.pi)), float(np.cos(np.pi)), 0.], [0., 0., 1.]])
pc_R, pc_T = pytorch3d.renderer.look_at_view_transform(
    dist=6,
    elev=0,
    azim=np.linspace(-180, 180, num_views, endpoint=False),
)
pc_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R=pc_R@R0,
    T=pc_T,
    device=device
)

pc_renderer = utils.get_points_renderer(device=device, radius=0.03)

my_images = pc_renderer(pc1.extend(num_views), cameras=pc_cameras, lights=lights)
imageio.mimsave("submission/pc1.gif", (my_images.cpu().numpy() * 255).astype(np.uint8), fps=4)

my_images = pc_renderer(pc2.extend(num_views), cameras=pc_cameras, lights=lights)
imageio.mimsave("submission/pc2.gif", (my_images.cpu().numpy() * 255).astype(np.uint8), fps=4)


my_images = pc_renderer(pc3.extend(num_views), cameras=pc_cameras, lights=lights)
imageio.mimsave("submission/pc3.gif", (my_images.cpu().numpy() * 255).astype(np.uint8), fps=4)

print("Completed 5.1")

# 5.2 Parametric Functions (10 points)
duration = 1000 // 15
images_1 = render_torus.render_torus(image_size=args.image_size)
imageio.mimsave("submission/torus_360.gif", images_1, duration=duration, loop=0)
print("Completed 5.2")

# 5.3 Implicit Surfaces (15 points)
duration = 1000 // 15
images_2 = render_torus.render_torus_mesh(image_size=args.image_size)
imageio.mimsave("submission/torus_mesh.gif", images_2, duration=duration, loop=0)
print("Completed 5.3")

############### 6. Do Something Fun ###############
parser = argparse.ArgumentParser()
parser.add_argument("--cow_path", type=str, default="data/cow.obj")
parser.add_argument("--output_path", type=str, default="submission/cow_fun.gif")
parser.add_argument("--image_size", type=int, default=256)
args = parser.parse_args()

my_images = something_fun.render_fun_cow(cow_path=args.cow_path, image_size=args.image_size)

duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
imageio.mimsave(args.output_path, my_images, duration=duration, loop=0)

print("Completed 6")
