import pytorch3d.renderer
import torch
import numpy as np
import imageio
import pytorch3d
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, TexturesVertex, RasterizationSettings,
    MeshRenderer, MeshRasterizer, HardPhongShader, FoVPerspectiveCameras, TexturesVertex,
    look_at_view_transform, PointsRasterizationSettings, PointsRenderer, AlphaCompositor,
    PointsRasterizer
)

def get_mesh_renderer(image_size=512, lights=None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

# def render_vox(voxels_src, voxels_tgt = None, src_path = "submission/source_vox.gif", tgt_path = "submission/target_vox.gif", num_views = 24):
def render_vox(voxels_src, src_path = "submission/source_vox.gif"):

    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    mesh = pytorch3d.ops.cubify(voxels_src, thresh=0.5)
    src_verts = mesh.verts_list()[0]
    src_faces = mesh.faces_list()[0]
    
    textures = (src_verts - src_verts.min()) / (src_verts.max() - src_verts.min())
    textures = pytorch3d.renderer.TexturesVertex(src_verts.unsqueeze(0))
    
    src_mesh = pytorch3d.structures.Meshes(
        verts=[src_verts], 
        faces=[src_faces], 
        textures = textures
    ).to(device)
    
    # lights = pytorch3d.renderer.PointLights(device=device, location=[[2.0, 2.0, -2.0]])
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    
    renderer = get_mesh_renderer(image_size=256, device=device)

    my_images = []
    
    for i in range (0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=2, elev=0, azim=i)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(src_mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3].clip(0,1)
        rend = (rend * 255).astype(np.uint8)
        my_images.append(rend)

    imageio.mimsave(src_path, my_images, loop=0, fps=12)
    
    return

def render_pointcloud(pointclouds_src, src_path = "submission/source_point.gif"):
    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    pointclouds_src = pointclouds_src.detach()[0]
    
    color = (pointclouds_src - pointclouds_src.min()) / (pointclouds_src.max() - pointclouds_src.min())

    pcd = pytorch3d.structures.Pointclouds(
        points=[pointclouds_src], features=[color],
    ).to(device)
        
    # initialize the renderer
    raster_settings = PointsRasterizationSettings(image_size=256, radius=0.01)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=(1,1,1)),
    )

    angle = torch.Tensor([0, 0, 0])
    r = pytorch3d.transforms.euler_angles_to_matrix(angle, "XYZ")
    my_images = []

    for i in range (0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=1.8, elev=15, azim=i)
        # R, T = pytorch3d.renderer.look_at_view_transform(dist=2, elev=30, azim=i)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R @ r, T=T, device=device)

        # lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -4]], device=device)

        rend = renderer(pcd, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

        rend = (rend*255).astype(np.uint8)
        my_images.append(rend)
    
    imageio.mimsave(src_path, my_images, fps=12, loop=0)

    return

def render_mesh(mesh_src, src_path = "submission/source_mesh.gif"):
    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    src_verts = mesh_src.verts_list()[0]
    src_faces = mesh_src.faces_list()[0]
    
    textures = (src_verts - src_verts.min()) / (src_verts.max() - src_verts.min())
    textures = pytorch3d.renderer.TexturesVertex(src_verts.unsqueeze(0))
    
    src_mesh = pytorch3d.structures.Meshes(
        verts=[src_verts], 
        faces=[src_faces], 
        textures = textures
    ).to(device)
    
    # lights = pytorch3d.renderer.PointLights(device=device, location=[[2.0, 2.0, -2.0]])
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    
    renderer = get_mesh_renderer(image_size=256, device=device)

    my_images = []
    
    for i in range (0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=1, elev=30, azim=i)
        # R, T = pytorch3d.renderer.look_at_view_transform(dist=2, elev=0, azim=i)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        # lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        rend = renderer(src_mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().detach().numpy()[0, ..., :3].clip(0,1)
        rend = (rend * 255).astype(np.uint8)
        my_images.append(rend)

    imageio.mimsave(src_path, my_images, loop=0, fps=12)
    
    return

def render_pointcloud_2_6(pointclouds_src, src_path = "submission/source_point.gif"):
    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # pointclouds_src = pointclouds_src.detach()[0]
        
    # initialize the renderer
    raster_settings = PointsRasterizationSettings(image_size=256, radius=0.01)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=(1,1,1)),
    )

    angle = torch.Tensor([0, 0, 0])
    r = pytorch3d.transforms.euler_angles_to_matrix(angle, "XYZ")
    my_images = []

    for i in range (0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=1.8, elev=15, azim=i)
        # R, T = pytorch3d.renderer.look_at_view_transform(dist=2, elev=30, azim=i)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R @ r, T=T, device=device)

        # lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -4]], device=device)

        rend = renderer(pointclouds_src, cameras=cameras, lights=lights)
        rend = rend.cpu().detach().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

        rend = (rend*255).astype(np.uint8)
        my_images.append(rend)
    
    imageio.mimsave(src_path, my_images, fps=12, loop=0)

    return
