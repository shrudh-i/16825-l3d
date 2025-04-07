import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
import numpy as np  

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_seg (verts, labels, path, device):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1,10000,3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15)

# def viz_cls (verts, path, device, title=None):
#     """
#     Visualize a point cloud for classification
#     output: a 360-degree gif
    
#     Args:
#         verts: point cloud vertices tensor of shape (N, 3)
#         path: output path for the gif
#         device: torch device
#         title: optional title for the visualization
#     """
#     import torch
#     import pytorch3d
#     import numpy as np
#     import imageio
    
#     image_size = 256
#     background_color = (1, 1, 1)
    
#     # Construct various camera viewpoints
#     dist = 3
#     elev = 20
#     azim = [180 - 12*i for i in range(30)]
#     R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
#     c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
    
#     # Prepare point cloud data
#     sample_verts = verts.unsqueeze(0).repeat(30, 1, 1).to(torch.float)
    
#     # Create colors based on z-coordinate for better visualization
#     # This creates a viridis-like colormap effect
#     z_vals = verts[:, 2]
#     min_z, max_z = z_vals.min(), z_vals.max()
#     normalized_z = (z_vals - min_z) / (max_z - min_z + 1e-8)
    
#     # Create RGB colors: blue to yellow gradient based on height
#     colors = torch.zeros((verts.shape[0], 3), device=device)
#     colors[:, 0] = normalized_z  # R channel increases with height
#     colors[:, 1] = normalized_z  # G channel increases with height
#     colors[:, 2] = 1 - normalized_z  # B channel decreases with height
    
#     sample_colors = colors.unsqueeze(0).repeat(30, 1, 1).to(torch.float)
    
#     # Create point cloud structure
#     point_cloud = pytorch3d.structures.Pointclouds(
#         points=sample_verts, 
#         features=sample_colors
#     ).to(device)
    
#     # Create renderer
#     raster_settings = pytorch3d.renderer.PointsRasterizationSettings(
#         image_size=image_size,
#         radius=0.01,
#     )
#     renderer = pytorch3d.renderer.PointsRenderer(
#         rasterizer=pytorch3d.renderer.PointsRasterizer(raster_settings=raster_settings),
#         compositor=pytorch3d.renderer.AlphaCompositor(background_color=background_color)
#     )
    
#     # Render point cloud from different viewpoints
#     rend = renderer(point_cloud, cameras=c).cpu().numpy()  # (30, 256, 256, 3)
    
#     # Add title if provided
#     if title:
#         import cv2
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.7
#         thickness = 2
#         color = (0, 0, 0)  # Black color for text
        
#         for i in range(rend.shape[0]):
#             img = (rend[i] * 255).astype(np.uint8)
#             # Get text size
#             text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
#             # Position text at top center
#             text_x = (img.shape[1] - text_size[0]) // 2
#             text_y = 30  # 30 pixels from the top
#             # Add text to image
#             cv2.putText(img, title, (text_x, text_y), font, font_scale, color, thickness)
#             rend[i] = img / 255.0
    
#     # Convert to uint8 for saving
#     rend = (rend * 255).astype(np.uint8)
    
#     # Save as gif
#     imageio.mimsave(path, rend, fps=15)

def viz_cls(verts, path, device, title=None, num_frames=50, fps=10):
    """
    Visualize a point cloud for classification using matplotlib
    output: a 360-degree gif showing the point cloud with proper orientation
    
    Args:
        verts: point cloud vertices tensor of shape (N, 3)
        path: output path for the gif
        device: torch device (not used in this implementation but kept for interface consistency)
        title: optional title for the visualization
        num_frames: number of frames in the animation (default: 30)
        fps: frames per second for the output gif (default: 15)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio
    from mpl_toolkits.mplot3d import Axes3D
    
    # Convert vertices to numpy if they're torch tensors
    if hasattr(verts, 'cpu'):
        points = verts.detach().cpu().numpy()
    else:
        points = verts
    
    # Normalize points for better visualization
    max_range = np.max(np.abs(points))
    
    # Store frames in memory
    frames = []
    
    # Generate frames with different view angles
    for i in range(num_frames):
        # Rotate around the y-axis
        azim_angle = i * (360 / num_frames)
        
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Use the z-coordinate for coloring with viridis colormap
        scatter = ax.scatter(
            points[:, 0], 
            points[:, 2],  # Swap y and z for better orientation
            points[:, 1], 
            s=5,  # Point size
            c=points[:, 2],  # Color by height (z-coordinate)
            cmap='viridis'
        )
        
        # Set consistent limits
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        # Add title if provided
        if title:
            ax.set_title(title)
        
        # Turn off axis for cleaner visualization
        ax.set_axis_off()
        
        # Set viewing angle to keep objects upright
        ax.view_init(elev=20, azim=azim_angle)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save the frame to memory
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(frame)
        plt.close(fig)
    
    # Create the GIF directly from memory
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"Saved GIF to {path}")

