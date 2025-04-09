import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import imageio
from models import cls_model
from utils import create_dir, viz_cls, rotate_point_cloud

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/cls_output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--num_vis', type=int, default=3, help='Number of examples to visualize per class')
    parser.add_argument('--frames', type=int, default=36, help='Number of frames in GIF animation')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second in the GIF')

    # Argument for rotation
    parser.add_argument('--rotation', type=bool, default=False, help='Toggle between Rotation analysis')
    parser.add_argument('--rotation_angles', type=str, default='0,45,90,120,160,190,220,270,300,360', 
                        help='Comma-separated list of rotation angles to test')
    
    return parser

def find_failures_by_class(pred_labels, true_labels, data, class_idx):
    """
    Find examples where model predicted incorrectly for a specific class
    """
    failure_indices = []
    
    # Find indices where true label is class_idx but prediction is wrong
    for i in range(len(true_labels)):
        if true_labels[i] == class_idx and pred_labels[i] != class_idx:
            failure_indices.append(i)
            
    return failure_indices

def find_success_by_class(pred_labels, true_labels, data, class_idx):
    """
    Find examples where model predicted correctly for a specific class
    """
    success_indices = []
    
    # Find indices where true label is class_idx and prediction is correct
    for i in range(len(true_labels)):
        if true_labels[i] == class_idx and pred_labels[i] == class_idx:
            success_indices.append(i)
            
    return success_indices

def evaluate_model(model, data, labels, device):
    """
    Evaluate model performance on the given data and labels
    
    Args:
        model: the classification model
        data: input point cloud data (numpy array)
        labels: ground truth labels (numpy array)
        device: torch device
    
    Returns:
        pred_labels: predicted labels
        accuracy: overall accuracy
        class_accuracies: dictionary of per-class accuracies
    """

    model.eval()
    pred_labels = []
    
    # Convert to torch tensors for prediction
    data_tensor = torch.from_numpy(data).float()
    
    with torch.no_grad():
        for i in range(len(data_tensor)):
            # Add batch dimension for single example
            input_data = data_tensor[i:i+1].to(device)
            output = model(input_data)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            pred_labels.append(pred)
    
    pred_labels = np.array(pred_labels, dtype=int)
    
    # Compute overall accuracy
    accuracy = np.mean(pred_labels == labels)
    
    # Compute per-class accuracy
    class_accuracies = {}
    for class_idx in range(len(np.unique(labels))):
        class_mask = labels == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(pred_labels[class_mask] == labels[class_mask])
            class_accuracies[class_idx] = class_acc
    
    return pred_labels, accuracy, class_accuracies

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)
    
    # Class names for better visualization
    class_names = ['chair', 'vase', 'lamp']

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(num_classes=args.num_cls_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("Successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = np.load(args.test_data)[:, ind, :]
    test_label = np.load(args.test_label).astype(int)
    
    # ------ TO DO: Make Prediction ------
    print("Evaluating model on original data:")
    pred_labels, accuracy, class_accuracies = evaluate_model(model, test_data,
                                                             test_label, args.device)

    print(f"Overall test accuracy: {accuracy:.4f}")
    for class_idx in range(args.num_cls_class):
        print(f"Class {class_names[class_idx]} accuracy: {class_accuracies.get(class_idx, 0):.4f}")
    
    # Find failures and successes for each class
    results = {}
    for class_idx in range(args.num_cls_class):
        failures = find_failures_by_class(pred_labels, test_label, test_data, class_idx)
        successes = find_success_by_class(pred_labels, test_label, test_data, class_idx)
        
        results[class_idx] = {
            'failures': failures,
            'successes': successes
        }
        
        print(f"Class {class_names[class_idx]}: {len(successes)} correct predictions, {len(failures)} failures")
    
    if not args.rotation:
    # Visualize random successful cases for each class
        print("\nCreating GIFs for successful predictions...")
        for class_idx in range(args.num_cls_class):
            if len(results[class_idx]['successes']) > 0:
                # Select random successful cases
                num_vis = min(args.num_vis, len(results[class_idx]['successes']))
                if num_vis > 0:
                    indices = random.sample(results[class_idx]['successes'], num_vis)
                    
                    for i, idx in enumerate(indices):
                        title = f"True: {class_names[test_label[idx]]}, Pred: {class_names[pred_labels[idx]]}"
                        out_file = f"{args.output_dir}/success_{class_names[test_label[idx]]}_{i}.gif"
                        viz_cls(torch.from_numpy(test_data[idx]).to(args.device), out_file, args.device, title)

        # Visualize failure cases for each class
        print("\nCreating GIFs for failure cases...")
        for class_idx in range(args.num_cls_class):
            if len(results[class_idx]['failures']) > 0:
                # Select at least one failure case per class if available
                num_vis = min(args.num_vis, len(results[class_idx]['failures']))
                if num_vis > 0:
                    indices = random.sample(results[class_idx]['failures'], num_vis)
                    
                    for i, idx in enumerate(indices):
                        title = f"True: {class_names[test_label[idx]]}, Pred: {class_names[pred_labels[idx]]}"
                        out_file = f"{args.output_dir}/failure_{class_names[test_label[idx]]}_as_{class_names[pred_labels[idx]]}_{i}.gif"
                        viz_cls(torch.from_numpy(test_data[idx]).to(args.device), out_file, args.device, title)
    
    # Rotation analysis
    if args.rotation:
        rotation_output_dir = os.path.join(args.output_dir, 'rotation_analysis')
        create_dir(rotation_output_dir)

        print("\n========== Starting Rotation Analysis ==========")
        
        # Parse rotation angles
        rotation_angles = [float(angle) for angle in args.rotation_angles.split(',')]

        # Store results for each angle
        rotation_results = {
            'angles': rotation_angles,
            'overall_accuracy': [],
            'class_accuracies': {}
        }
        
        for class_idx in range(args.num_cls_class):
            rotation_results['class_accuracies'][class_idx] = []

        # For visualization: Pick two successful and two failed samples from each class
        vis_samples = {'successes': {}, 'failures': {}}
        for class_idx in range(args.num_cls_class):
            # Get successful examples
            if len(results[class_idx]['successes']) >= 2:
                vis_samples['successes'][class_idx] = random.sample(results[class_idx]['successes'], 2)
            elif len(results[class_idx]['successes']) == 1:
                vis_samples['successes'][class_idx] = results[class_idx]['successes']
            
            # Get failed examples
            if len(results[class_idx]['failures']) >= 2:
                vis_samples['failures'][class_idx] = random.sample(results[class_idx]['failures'], 2)
            elif len(results[class_idx]['failures']) == 1:
                vis_samples['failures'][class_idx] = results[class_idx]['failures']

        # Initialize per-object accuracy tracking
        per_object_correct = np.zeros(len(test_data))
        per_object_total = np.zeros(len(test_data))

        # Test for each rotation angle
        for angle in rotation_angles:
            print(f"\nEvaluating at rotation angle: {angle} degrees")
            
            # Rotate each point cloud
            rotated_data = np.zeros_like(test_data)
            for i in range(len(test_data)):
                rotated_data[i] = rotate_point_cloud(test_data[i], angle)
            
            # Evaluate model on rotated data
            rot_pred_labels, rot_accuracy, rot_class_accuracies = evaluate_model(
                model, rotated_data, test_label, args.device
            )
            
            # Store results
            rotation_results['overall_accuracy'].append(rot_accuracy)
            for class_idx in range(args.num_cls_class):
                rotation_results['class_accuracies'][class_idx].append(
                    rot_class_accuracies.get(class_idx, 0)
                )
            
            # Report results
            print(f"Overall accuracy at {angle}°: {rot_accuracy:.4f}")
            for class_idx in range(args.num_cls_class):
                print(f"Class {class_names[class_idx]} accuracy at {angle}°: {rot_class_accuracies.get(class_idx, 0):.4f}")

            # Visualize rotated samples
            for sample_type in ['successes', 'failures']:
                for class_idx, indices in vis_samples[sample_type].items():
                    for i, idx in enumerate(indices):
                        current_pred = rot_pred_labels[idx]
                        title = f"GT: {class_names[test_label[idx]]}; Pred: {class_names[current_pred]}; Rotation: {angle}°"
                        out_file = f"{rotation_output_dir}/rot_{angle}_class_{class_names[class_idx]}_{sample_type[:-1]}_{i}.gif"
                        viz_cls(torch.from_numpy(rotated_data[idx]).to(args.device), out_file, args.device, title)
            
            '''
            # Plot accuracy vs rotation angle
            plt.figure(figsize=(12, 6))
            
            # Plot overall accuracy
            plt.plot(rotation_angles, rotation_results['overall_accuracy'], 'o-', label='Overall', linewidth=2)
            
            # Plot per-class accuracy
            for class_idx in range(args.num_cls_class):
                plt.plot(rotation_angles, rotation_results['class_accuracies'][class_idx], 'o-', 
                        label=f'Class {class_names[class_idx]}', linewidth=2)
            
            plt.xlabel('Rotation Angle (degrees)')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy vs. Rotation Angle (X-axis Rotation)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f"{rotation_output_dir}/accuracy_vs_rotation.png", dpi=300)
            print(f"\nAccuracy plot saved to {rotation_output_dir}/accuracy_vs_rotation.png")
            '''

            print("\n========== Rotation Analysis Completed ==========")

        
    
    