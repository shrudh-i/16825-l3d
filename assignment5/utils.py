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

#NOTE: original visualization code
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

    # Get the number of points from the input tensor
    num_points = verts.shape[0]

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1,num_points,3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15)
'''

#NOTE: my additional visualization code
def viz_seg(verts, labels, path, device, title=None, num_frames=30, fps=15):
    """
    Visualize segmentation result using matplotlib
    output: a 360-degree gif
    
    Args:
        verts: point cloud vertices tensor of shape (N, 3)
        labels: segmentation labels tensor of shape (N,)
        path: output path for the gif
        device: torch device (kept for interface consistency)
        title: optional title for the visualization
        num_frames: number of frames in the animation (default: 30)
        fps: frames per second for the output gif (default: 15)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio
    from mpl_toolkits.mplot3d import Axes3D
    
    # Convert vertices and labels to numpy if they're torch tensors
    if hasattr(verts, 'cpu'):
        points = verts.detach().cpu().numpy()
    else:
        points = verts
        
    if hasattr(labels, 'cpu'):
        labels = labels.detach().cpu().numpy()
    
    # Define colors for each segment (enhanced for better aesthetics while keeping the same approach)
    colors = [
        [1.0, 1.0, 1.0],  # White
        [1.0, 0.0, 1.0],  # Magenta/Purple
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 1.0, 0.0],  # Yellow
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 0.0, 0.0]   # Red
    ]
    
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
        
        # Plot each segment with its color
        for segment_id in range(6):
            mask = labels == segment_id
            if np.any(mask):  # Only plot if there are points with this label
                segment_points = points[mask]
                color = colors[segment_id]
                ax.scatter(
                    segment_points[:, 0],
                    segment_points[:, 2],  # Swap y and z for better orientation
                    segment_points[:, 1],
                    s=5,  # Point size
                    color=color,
                    label=f"Segment {segment_id}"
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
    print(f"Saved segmentation GIF to {path}")
'''

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

def rotate_point_cloud(point_cloud, angle_deg):
    """
    Rotate the point cloud around X axis by angle_deg degrees
    
    Args:
        point_cloud: numpy array of shape (N, 3)
        angle_deg: rotation angle in degrees
    
    Returns:
        rotated point cloud of shape (N, 3)
    """
    rotation_angle = np.radians(angle_deg)
    
    # Rotation matrix around x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
        [0, np.sin(rotation_angle), np.cos(rotation_angle)]
    ])
    
    # Apply rotation
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
    
    return rotated_point_cloud
