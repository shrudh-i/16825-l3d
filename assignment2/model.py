from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # pass
            # TODO:
            self.layer0 = torch.nn.Sequential(
                torch.nn.Linear(512, 2048)
            )
            # self.layer0 = torch.nn.Sequential(
            #     torch.nn.Linear(512, 1024),
            #     # torch.nn.ReLU(),
            #     torch.nn.Linear(1024, 2048),
            #     # torch.nn.ReLU()
            # )
            self.layer1 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
                torch.nn.BatchNorm3d(128),
                torch.nn.ReLU()
            )
            self.layer2 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
                torch.nn.BatchNorm3d(64),
                torch.nn.ReLU()
            )
            self.layer3 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
                torch.nn.BatchNorm3d(32),
                torch.nn.ReLU()
            )
            self.layer4 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=False, padding=1),
                torch.nn.BatchNorm3d(8),
                torch.nn.ReLU()
            )
            self.layer5 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
                # torch.nn.Sigmoid()
            )
            # self.decoder =             
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            # self.decoder =  torch.nn.Sequential(
            #     torch.nn.Linear(512, 1024),
            #     torch.nn.ReLU(),
            #     # torch.nn.Linear(1024, self.n_point),
            #     torch.nn.Linear(1024, 1000),
            #     torch.nn.ReLU(),
            #     # torch.nn.Linear(self.n_point, self.n_point*3),
            #     torch.nn.Linear(1000, 1000*3),
            #     torch.nn.Tanh()
            # )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(512, self.n_point),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.n_point, self.n_point*3),
                torch.nn.Tanh()
            )
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            self.layer0 =  nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 3*mesh_pred.verts_packed().shape[0])
            )   

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            gen_volume = self.layer0(encoded_feat)
            gen_volume = gen_volume.view(-1, 256, 2, 2, 2)
            gen_volume = self.layer1(gen_volume)
            gen_volume = self.layer2(gen_volume)
            gen_volume = self.layer3(gen_volume)
            gen_volume = self.layer4(gen_volume)
            voxels_pred = self.layer5(gen_volume)      
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)      
            # pointclouds_pred = pointclouds_pred.view(-1, args.n_points, 3)
            pointclouds_pred = pointclouds_pred.view(-1, 1000, 3)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.layer0(encoded_feat)         
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

