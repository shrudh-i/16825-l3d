import torch
import pytorch3d
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	loss = torch.nn.functional.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
	# implement some loss for binary voxel grids
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  

	# KNN from source to target 
	knn_src2trgt = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt)
	knn_trgt2src = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src)
	
	# Taking the mean of these distances ensures that the loss scales properly
	loss_src2trgt = torch.mean(knn_src2trgt.dists)  # (B, N, k)
	loss_trgt2src = torch.mean(knn_trgt2src.dists)  # (B, M, k)
	
	# Sum both directions
	loss_chamfer = loss_src2trgt + loss_trgt2src

	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian