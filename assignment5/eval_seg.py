import numpy as np
import argparse
import os

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg, rotate_point_cloud


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/seg_output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    parser.add_argument('--num_good', type=int, default=5, help='number of success visualizations')
    parser.add_argument('--num_bad', type=int, default=2, help='number of failure visualizations')

    # Argument for rotation
    parser.add_argument('--rotation', type=bool, default=False, help='Toggle between Rotation analysis')
    parser.add_argument('--rotation_angles', type=str, default='0,45,90,120,160,190,220,270,300,360', 
                        help='Comma-separated list of rotation angles to test')

    return parser

def evaluate_segmentation(model, data, labels, device):
    """
    Evaluate model performance on segmentation data
    
    Args:
        model: segmentation model
        data: input point cloud data tensor
        labels: ground truth labels tensor
        device: torch device
        
    Returns:
        pred_labels: predicted labels
        accuracy: overall point accuracy
        per_obj_accuracy: accuracy for each object
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(data.to(device))
        pred_labels = torch.argmax(outputs, dim=2)
    
    # Calculate overall accuracy (across all points)
    accuracy = pred_labels.eq(labels.to(device)).cpu().sum().item() / (labels.numel())
    
    # Calculate per-object accuracy
    per_obj_accuracy = []
    for i in range(len(data)):
        obj_acc = pred_labels[i].eq(labels[i].to(device)).cpu().sum().item() / labels[i].numel()
        per_obj_accuracy.append(obj_acc)
    
    return pred_labels, accuracy, per_obj_accuracy

def find_best_worst_objects(per_obj_accuracy, num_good=5, num_bad=2):
    """
    Find indices of the best and worst performing objects
    
    Args:
        per_obj_accuracy: list of accuracy values for each object
        num_good: number of best objects to return
        num_bad: number of worst objects to return
        
    Returns:
        good_indices: indices of best performing objects
        bad_indices: indices of worst performing objects
    """
    # Convert to numpy array for easier manipulation
    accuracies = np.array(per_obj_accuracy)
    
    # Get indices sorted by accuracy (ascending)
    sorted_indices = np.argsort(accuracies)
    
    # Get worst and best indices
    bad_indices = sorted_indices[:num_bad]
    good_indices = sorted_indices[-num_good:]
    
    return good_indices, bad_indices

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TO DO: Make Prediction ------
    pred_label, test_accuracy, per_obj_accuracy = evaluate_segmentation(
        model, test_data, test_label, args.device
    )

    print ("Overall test accuracy: {}".format(test_accuracy))

    # # Visualize Segmentation Result (Pred VS Ground Truth)
    # viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    # viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)

    # Find best and worst performing objects
    good_indices, bad_indices = find_best_worst_objects(
        per_obj_accuracy, args.num_good, args.num_bad
    )
    
    '''
    # Create class-specific output directories
    class_dirs = []
    for i in range(args.num_seg_class):
        class_dir = os.path.join(args.output_dir, f"class_{i}")
        create_dir(class_dir)
        class_dirs.append(class_dir)
    '''

    if not args.rotation:
        # Visualize good predictions
        print("\nCreating visualizations for good predictions...")
        for i, idx in enumerate(good_indices):
            acc = per_obj_accuracy[idx]
            title = f"Good Prediction: Accuracy = {acc:.4f}"
            
            # Save ground truth visualization
            gt_path = os.path.join(args.output_dir, f"good_{i}_gt_{args.exp_name}.gif")
            viz_seg(test_data[idx], test_label[idx], gt_path, args.device)
            
            # Save prediction visualization
            pred_path = os.path.join(args.output_dir, f"good_{i}_pred_{args.exp_name}.gif")
            viz_seg(test_data[idx], pred_label[idx], pred_path, args.device)
            print(f"  Object {idx}: Accuracy = {acc:.4f}")
            
            '''
            # Save to class-specific folder based on majority class
            majority_class = torch.mode(test_label[idx])[0].item()
            # class_gt_path = os.path.join(class_dirs[majority_class], f"good_{i}_gt.gif")
            # class_pred_path = os.path.join(class_dirs[majority_class], f"good_{i}_pred.gif")
            
            # viz_seg(test_data[idx], test_label[idx], class_gt_path, args.device)
            # viz_seg(test_data[idx], pred_label[idx], class_pred_path, args.device)
            
            print(f"  Object {idx}: Accuracy = {acc:.4f}, Majority Class = {majority_class}")
            '''
        
        # Visualize bad predictions
        print("\nCreating visualizations for bad predictions...")
        for i, idx in enumerate(bad_indices):
            acc = per_obj_accuracy[idx]
            title = f"Bad Prediction: Accuracy = {acc:.4f}"
            
            # Save ground truth visualization
            gt_path = os.path.join(args.output_dir, f"bad_{i}_gt_{args.exp_name}.gif")
            viz_seg(test_data[idx], test_label[idx], gt_path, args.device)
            
            # Save prediction visualization
            pred_path = os.path.join(args.output_dir, f"bad_{i}_pred_{args.exp_name}.gif")
            viz_seg(test_data[idx], pred_label[idx], pred_path, args.device)
            print(f"  Object {idx}: Accuracy = {acc:.4f}")
            
            '''
            # Save to class-specific folder based on majority class
            majority_class = torch.mode(test_label[idx])[0].item()
            # class_gt_path = os.path.join(class_dirs[majority_class], f"bad_{i}_gt.gif")
            # class_pred_path = os.path.join(class_dirs[majority_class], f"bad_{i}_pred.gif")
            
            # viz_seg(test_data[idx], test_label[idx], class_gt_path, args.device)
            # viz_seg(test_data[idx], pred_label[idx], class_pred_path, args.device)
            
            print(f"  Object {idx}: Accuracy = {acc:.4f}, Majority Class = {majority_class}")
            '''

    if args.rotation:
        rotation_output_dir = os.path.join(args.output_dir, 'rotation_analysis')
        create_dir(rotation_output_dir)

        print("\n========== Starting Rotation Analysis ==========")

         # Parse rotation angles
        rotation_angles = [float(angle) for angle in args.rotation_angles.split(',')]
        
        # # Create class-specific rotation folders
        # rotation_class_dirs = []
        # for i in range(args.num_seg_class):
        #     class_rot_dir = os.path.join(rotation_output_dir, f"class_{i}")
        #     create_dir(class_rot_dir)
        #     rotation_class_dirs.append(class_rot_dir)
        
        # Select a few objects for rotation visualization (1 good and 1 bad per class if available)
        vis_indices = []
        
        # Add num_good and num_bad example for visualization
        if len(good_indices) > 0:
            vis_indices.append(('good', good_indices[0]))
        
        if len(bad_indices) > 0:
            vis_indices.append(('bad', bad_indices[0]))

        # Test for each rotation angle
        for angle in rotation_angles:
            print(f"\nEvaluating at rotation angle: {angle} degrees")

            # Rotate the data
            rotated_data = torch.zeros_like(test_data)
            for i in range(len(test_data)):
                rotated_points = rotate_point_cloud(test_data[i].cpu().numpy(), angle)
                rotated_data[i] = torch.from_numpy(rotated_points)

            # Evaluate model on rotated data
            rot_pred_label, rot_accuracy, rot_per_obj_accuracy = evaluate_segmentation(
                model, rotated_data, test_label, args.device
            )
            
            print(f"Overall accuracy at {angle}°: {rot_accuracy:.4f}")

            # Visualize selected objects at this rotation angle
            for sample_type, idx in vis_indices:
                acc = rot_per_obj_accuracy[idx]
                
                # Get majority class
                majority_class = torch.mode(test_label[idx])[0].item()
                
                # Create paths for rotation visualizations
                gt_path = os.path.join(rotation_output_dir, 
                                      f"rot_{angle}_{sample_type}_gt_{idx}.gif")
                pred_path = os.path.join(rotation_output_dir, 
                                        f"rot_{angle}_{sample_type}_pred_{idx}.gif")
                
                # # Class-specific paths
                # class_gt_path = os.path.join(rotation_class_dirs[majority_class], 
                #                            f"rot_{angle}_{sample_type}_gt_{idx}.gif")
                # class_pred_path = os.path.join(rotation_class_dirs[majority_class], 
                #                              f"rot_{angle}_{sample_type}_pred_{idx}.gif")
                
                # Visualize
                viz_seg(rotated_data[idx], test_label[idx], gt_path, args.device)
                viz_seg(rotated_data[idx], rot_pred_label[idx], pred_path, args.device)
                
                # # Class-specific visualizations
                # viz_seg(rotated_data[idx], test_label[idx], class_gt_path, args.device)
                # viz_seg(rotated_data[idx], rot_pred_label[idx], class_pred_path, args.device)
                
                print(f"  Rotated Object {idx}: Accuracy = {acc:.4f}, Majority Class = {majority_class}")
        
        print("\n========== Rotation Analysis Completed ==========")