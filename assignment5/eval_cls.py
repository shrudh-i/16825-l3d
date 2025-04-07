import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import imageio
from models import cls_model
from utils import create_dir, viz_cls

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
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--num_vis', type=int, default=3, help='Number of examples to visualize per class')
    parser.add_argument('--frames', type=int, default=36, help='Number of frames in GIF animation')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second in the GIF')

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
    test_label = np.load(args.test_label)
    
    # ------ TO DO: Make Prediction ------
    model.eval()
    pred_labels = []
    
    # Convert to torch tensors for prediction
    test_data_tensor = torch.from_numpy(test_data).float()
    
    with torch.no_grad():
        for i in range(len(test_data_tensor)):
            # Add batch dimension for single example
            input_data = test_data_tensor[i:i+1].to(args.device)
            output = model(input_data)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            pred_labels.append(pred)
    
    pred_labels = np.array(pred_labels)
    
    # Convert labels to integers if they are floats
    test_label = test_label.astype(int)
    pred_labels = pred_labels.astype(int)
    
    # Compute overall accuracy
    accuracy = np.mean(pred_labels == test_label)
    print(f"Overall test accuracy: {accuracy:.4f}")
    
    # Compute per-class accuracy
    for class_idx in range(args.num_cls_class):
        class_mask = test_label == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(pred_labels[class_mask] == test_label[class_mask])
            print(f"Class {class_names[class_idx]} accuracy: {class_acc:.4f}")
    
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
                    # Use the new viz_cls function instead of create_rotating_gif
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
                    # Use the new viz_cls function instead of create_rotating_gif
                    viz_cls(torch.from_numpy(test_data[idx]).to(args.device), out_file, args.device, title)
    
    # Compute and display confusion matrix
    confusion_matrix = np.zeros((args.num_cls_class, args.num_cls_class), dtype=int)
    for i in range(len(test_label)):
        confusion_matrix[test_label[i], pred_labels[i]] += 1
    
    print("\nConfusion Matrix:")
    print("Rows: True class, Columns: Predicted class")
    print(confusion_matrix)
    
    # Print interpretation of failure cases
    print("\nInterpretation of Failure Cases:")
    for class_idx in range(args.num_cls_class):
        if len(results[class_idx]['failures']) > 0:
            misclassified_as = {}
            for idx in results[class_idx]['failures']:
                pred = pred_labels[idx]
                if pred not in misclassified_as:
                    misclassified_as[pred] = 0
                misclassified_as[pred] += 1
            
            print(f"\n{class_names[class_idx].capitalize()} misclassification analysis:")
            for pred_class, count in misclassified_as.items():
                print(f"  • Misclassified as {class_names[pred_class]}: {count} instances")
            
            # Detailed analysis based on confusion patterns
            if class_idx == 0:  # chair
                chair_as_vase = misclassified_as.get(1, 0)
                chair_as_lamp = misclassified_as.get(2, 0)
                print(f"  • Analysis of chair misclassifications:")
                if chair_as_vase > 0:
                    print(f"    - {chair_as_vase} chairs misclassified as vases - likely chairs with thin vertical structures")
                if chair_as_lamp > 0:
                    print(f"    - {chair_as_lamp} chairs misclassified as lamps - likely chairs with unusual shapes or arm structures")
                
            elif class_idx == 1:  # vase
                vase_as_chair = misclassified_as.get(0, 0)
                vase_as_lamp = misclassified_as.get(2, 0)
                print(f"  • Analysis of vase misclassifications:")
                if vase_as_chair > 0:
                    print(f"    - {vase_as_chair} vases misclassified as chairs - possibly vases with wider bases or unusual shapes")
                if vase_as_lamp > 0:
                    print(f"    - {vase_as_lamp} vases misclassified as lamps - likely vases with flared openings resembling lampshades")
                
            elif class_idx == 2:  # lamp
                lamp_as_chair = misclassified_as.get(0, 0)
                lamp_as_vase = misclassified_as.get(1, 0)
                print(f"  • Analysis of lamp misclassifications:")
                if lamp_as_chair > 0:
                    print(f"    - {lamp_as_chair} lamps misclassified as chairs - possibly desk lamps with arm structures")
                if lamp_as_vase > 0:
                    print(f"    - {lamp_as_vase} lamps misclassified as vases - likely floor lamps with straight vertical structures")