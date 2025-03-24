import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import TruckDataset, visualize_renders
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn.functional 

#NOTE: My addition to the implementation
def save_checkpoint(gaussians, optimizer, itr, loss, args):
    """Save a checkpoint of the model"""
    checkpoint = {
        'iteration': itr,
        'means': gaussians.means.detach().cpu(),
        'colours': gaussians.colours.detach().cpu(),
        'pre_act_scales': gaussians.pre_act_scales.detach().cpu(),
        'pre_act_opacities': gaussians.pre_act_opacities.detach().cpu(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss.item()
    }

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = os.path.join(args.out_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{itr:07d}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save the latest checkpoint path to a small file for easy retrieval
    with open(os.path.join(checkpoint_dir, "latest_checkpoint.txt"), "w") as f:
        f.write(checkpoint_path)
    
    print(f"[*] Checkpoint saved at iteration {itr}")

#NOTE: My addition to the implementation
def load_checkpoint(gaussians, optimizer, args):
    """Load the latest checkpoint if available"""
    checkpoint_dir = os.path.join(args.out_path, "checkpoints")
    latest_file = os.path.join(checkpoint_dir, "latest_checkpoint.txt")
    if not os.path.exists(checkpoint_dir) or not os.path.exists(latest_file):
        print("[*] No checkpoint found, starting from scratch")
        return 0, 0.0
    
    with open(latest_file, "r") as f:
        checkpoint_path = f.read().strip()
    
    if not os.path.exists(checkpoint_path):
        print(f"[*] Checkpoint file {checkpoint_path} not found, starting from scratch")
        return 0, 0.0
        
    print(f"[*] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Load model parameters
    gaussians.means = checkpoint['means'].to(args.device).requires_grad_(True)
    gaussians.colours = checkpoint['colours'].to(args.device).requires_grad_(True)
    gaussians.pre_act_scales = checkpoint['pre_act_scales'].to(args.device).requires_grad_(True)
    gaussians.pre_act_opacities = checkpoint['pre_act_opacities'].to(args.device).requires_grad_(True)
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return checkpoint['iteration'], checkpoint['loss']

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    # Making all parameters trainable by setting requires_grad=True
    gaussians.means.requires_grad = True 
    gaussians.pre_act_scales.requires_grad = True
    gaussians.pre_act_opacities.requires_grad = True
    gaussians.colours.requires_grad = True

    '''
    note:
        * means -> centre positions of the gaussians in space
        * pre_act_scales -> controls the size of the gaussians
        * pre_act_opacities -> controls the opacity of the gaussians
        * colours -> controls the colour (RGB) of the gaussians
    '''

def setup_optimizer(gaussians):

    gaussians.check_if_trainable()

    ### YOUR CODE HERE ###
    # HINT: Modify the learning rates to reasonable values. We have intentionally
    # set very high learning rates for all parameters.
    # HINT: Consider reducing the learning rates for parameters that seem to vary too
    # fast with the default settings.
    # HINT: Consider setting different learning rates for different sets of parameters.
    '''
    DEFAULT:
    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': 0.05, "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': 0.05, "name": "scales"},
        {'params': [gaussians.colours], 'lr': 0.05, "name": "colours"},
        {'params': [gaussians.means], 'lr': 0.05, "name": "means"},
    ]
    optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)
    optimizer = None
    '''
    # parameters = [
    #     {'params': [gaussians.pre_act_opacities], 'lr': 0.05, "name": "opacities"},
    #     {'params': [gaussians.pre_act_scales], 'lr': 0.02, "name": "scales"},
    #     {'params': [gaussians.colours], 'lr': 0.05, "name": "colours"},
    #     {'params': [gaussians.means], 'lr': 0.005, "name": "means"},
    # ]

    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': 0.001, "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': 0.002, "name": "scales"},
        {'params': [gaussians.colours], 'lr': 0.01, "name": "colours"},
        {'params': [gaussians.means], 'lr': 0.00035, "name": "means"},
    ]


    optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)
    # optimizer = None

    return optimizer

'''
#NOTE: My addition to the implementation
def setup_scheduler(optimizer):
    # The ReduceLROnPlateau scheduler will reduce learning rates when the loss plateaus
    # - 'mode': 'min' because we want to minimize the loss
    # - 'factor': 0.5 means the learning rate will be halved when triggered
    # - 'patience': 50 means it will wait 50 iterations before reducing LR if no improvement
    # - 'threshold': 0.01 is the minimum improvement needed to be considered progress
    # - 'verbose': True means it will print a message when LR is reduced
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, 
        threshold=0.01, verbose=True
    )

    return scheduler
'''

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    # Setting up dataset
    train_dataset = TruckDataset(root=args.data_path, split="train")
    test_dataset = TruckDataset(root=args.data_path, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=TruckDataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=TruckDataset.collate_fn
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    viz_gif_path_1 = os.path.join(args.out_path, "q1_training_progress.gif")
    viz_gif_path_2 = os.path.join(args.out_path, "q1_training_final_renders.gif")
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [(train_dataset[i][0]*255.0).numpy().astype(np.uint8) for i in viz_idxs]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [train_dataset[i][1].cuda() for i in viz_idxs]

    # Init gaussians and scene. Do note that we are setting isotropic to True
    gaussians = Gaussians(
        load_path=train_dataset.points_path, init_type="points",
        device=args.device, isotropic=True
    )
    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    optimizer = setup_optimizer(gaussians)
    '''
    #NOTE: My addition to the implementation
    # scheduler = setup_scheduler(optimizer)
    # # For tracking loss history (useful for debugging)
    # loss_history = []
    '''
    
    #NOTE: My addition to the implementation
    # Try to load the latest checkpoint
    start_itr, last_loss = load_checkpoint(gaussians, optimizer, args)

    print("[*] Starting Training")

    # Training loop
    viz_frames = []
    # for itr in range(args.num_itrs):
    for itr in range(start_itr, args.num_itrs):

        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img, camera, gt_mask = data

        gt_img = gt_img[0].cuda()
        camera = camera[0].cuda()
        if gt_mask is not None:
            gt_mask = gt_mask[0].cuda()

        # Rendering scene using gaussian splatting
        ### YOUR CODE HERE ###
        # HINT: Can any function from the Scene class help?
        # HINT: Set bg_colour to (0.0, 0.0, 0.0)
        # HINT: Get img_size from train_dataset
        # HINT: Get per_splat from args.gaussians_per_splat
        # HINT: camera is available above
        pred_img, _, _ = scene.render(
            camera = camera,
            img_size = train_dataset.img_size,
            per_splat = args.gaussians_per_splat,
            bg_colour = (0.0, 0.0, 0.0)
        )

        # Compute loss
        ### YOUR CODE HERE ###
        # HINT: A simple standard loss function should work.
        loss = torch.nn.functional.l1_loss(pred_img, gt_img)
        
        '''
        #NOTE: My addition to the implementation
        # loss_history.append(loss.item())
        # # Check for NaN values (debugging)
        # if torch.isnan(loss):
        #     print("Warning: NaN loss detected! Skipping this iteration.")
        #     optimizer.zero_grad()
        #     continue
        '''
        
        '''
        #NOTE: My addition to the implementation
        if itr % 20 == 0:  # Periodically check parameters for NaN
            if torch.isnan(gaussians.means).any():
                print("Warning: NaN values detected in Gaussian means")
            if torch.isnan(gaussians.pre_act_scales).any():
                print("Warning: NaN values detected in Gaussian scales")
            if torch.isnan(gaussians.pre_act_opacities).any():
                print("Warning: NaN values detected in Gaussian opacities")
            if torch.isnan(gaussians.colours).any():
                print("Warning: NaN values detected in Gaussian colours")
        '''


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #NOTE: My addition to the implementation
        # Save checkpoint periodically
        if (itr+1) % args.checkpoint_freq == 0:
            save_checkpoint(gaussians, optimizer, itr, loss, args)

        '''
        #NOTE: My addition to the implementation
        # scheduler.step(loss)

        ///////////////////
        # Calculate and report PSNR during training for better monitoring
        if itr % 20 == 0:
            with torch.no_grad():
                mse = ((pred_img - gt_img) ** 2).mean()
                psnr = -10 * torch.log10(mse)
                print(f"[*] Itr: {itr:07d} | Loss: {loss.item():0.5f} | PSNR: {psnr.item():0.3f}")
        else:
            print(f"[*] Itr: {itr:07d} | Loss: {loss.item():0.5f}")
        '''

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

        if itr % args.viz_freq == 0:
            viz_frame = visualize_renders(
                scene, gt_viz_img,
                viz_cameras, train_dataset.img_size
            )
            viz_frames.append(viz_frame)

    print("[*] Training Completed.")

    '''
    #NOTE: My addition to the implementation
    # Save loss history for analysis (optional)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(args.out_path, training_loss.png"))
    '''

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=TruckDataset.collate_fn
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img, camera, gt_mask = viz_data
        gt_img = gt_img[0].cuda()
        camera = camera[0].cuda()
        if gt_mask is not None:
            gt_mask = gt_mask[0].cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Get img_size from train_dataset
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, _, _ = scene.render(
                camera = camera,
                img_size = train_dataset.img_size,
                per_splat = args.gaussians_per_splat,
                bg_colour = (0.0, 0.0, 0.0)
            )

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1/10.0)*1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for test_data in tqdm(test_loader, desc="Running Evaluation"):

        gt_img, camera, gt_mask = test_data
        gt_img = gt_img[0].cuda()
        camera = camera[0].cuda()
        if gt_mask is not None:
            gt_mask = gt_mask[0].cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Get img_size from test_dataset
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, _, _ = scene.render(
                camera = camera,
                img_size = test_dataset.img_size, #NOTE: using test dataset image size jere
                per_splat = args.gaussians_per_splat,
                bg_colour = (0.0, 0.0, 0.0)
            )

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/truck", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument(
        "--num_itrs", default=1000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])

    #NOTE: My addition to the implementation
    parser.add_argument(
        "--checkpoint_freq", default=100, type=int,
        help="Frequency with which checkpoints should be saved."
    )

    #NOTE: My addition to the implementation
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint if available."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)
