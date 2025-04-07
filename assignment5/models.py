import torch
import torch.nn as nn
import torch.nn.functional as F

#NOTE: my implementation - a feature extraction module that will be chared by cls_model & seg_model
class PointNetFeatures(nn.Module):
    '''
    Shared MLP Layers for feature extraction
    '''
    def __init__(self):
        super(PointNetFeatures, self).__init__()
        # MLP 1: 3 -> 64 -> 64
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # MLP 2: 64 -> 64 -> 128 -> 1024
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B - batch size, N - number of points
        output: tensor of size (B, 1024, N)
        '''
        # Reshape to (B, 3, N) for Cov1d operations 
        x = points.transpose(2,1)

        # Apply MLP1
        x = self.mlp1(x) # (B, 64, N)

        # Apply MLP2
        x = self.mlp2(x) # (B, 1024, N)
        return x

# ------ TO DO ------
# Classification model
class cls_model(nn.Module): 
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.num_classes = num_classes

        # Feature extraction
        self.features = PointNetFeatures()

        # MLP for classification: 1024 -> 512 -> 256 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # Extract point features
        x = self.features(points)  # (B, 1024, N)
        
        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, 1024)
        
        # Classification
        x = self.classifier(x)  # (B, num_classes)
        return x



# ------ TO DO ------
# class seg_model(nn.Module):
#     def __init__(self, num_seg_classes = 6):
#         super(seg_model, self).__init__()
        
#         # Feature extraction
#         self.features = PointNetFeatures()

#         # MLP for segmentation
#         self.segmenter = nn.Sequential(
#             nn.Conv1d(1088, 512, 1),  # 1088 = 1024 (global) + 64 (local)
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Conv1d(512, 256, 1),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Conv1d(256, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Conv1d(128, num_seg_classes, 1)
#         )

#     def forward(self, points):
#         '''
#         points: tensor of size (B, N, 3)
#                 , where B is batch size and N is the number of points per object (N=10000 by default)
#         output: tensor of size (B, N, num_seg_classes)
#         '''
#         batch_size = points.size(0)
#         num_points = points.size(1)
        
#         # Extract point features (B, 1024, N)
#         x_local = self.features.mlp1(points.transpose(2, 1))  # (B, 64, N)
#         x = self.features.mlp2(x_local)  # (B, 1024, N)
        
#         # Global feature (B, 1024, 1)
#         global_feat = torch.max(x, dim=2, keepdim=True)[0]  # max pooling
        
#         # Expand global feature to all points
#         global_feat_expanded = global_feat.repeat(1, 1, num_points)  # (B, 1024, N)
        
#         # Concatenate global and local features
#         x = torch.cat([global_feat_expanded, x_local], dim=1)  # (B, 1024+64, N)
        
#         # Segmentation
#         x = self.segmenter(x)  # (B, num_seg_classes, N)
        
#         # Reshape to (B, N, num_seg_classes)
#         x = x.transpose(2, 1)
        
#         return x

class seg_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super(seg_model, self).__init__()

        self.features = PointNetFeatures()

        self.segmenter = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, num_seg_classes, 1)
        )

    def forward(self, points):
        batch_size = points.size(0)
        num_points = points.size(1)

        x_local = self.features.mlp1(points.transpose(2, 1))  # (B, 64, N)
        x_global = self.features.mlp2(x_local)  # (B, 1024, N)
        global_feat = torch.max(x_global, dim=2, keepdim=True)[0]  # (B, 1024, 1)
        global_feat_expanded = global_feat.repeat(1, 1, num_points)  # (B, 1024, N)
        x = torch.cat([global_feat_expanded, x_local], dim=1)  # (B, 1088, N)

        x = self.segmenter(x)  # (B, num_seg_classes, N)
        return x.transpose(2, 1)  # (B, N, num_seg_classes)
