import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np


batch_size = 1
# Select number of principal components for pose space
ncomps = 45

taxonomy_posess = np.load('average_hand_pose_taxonomy.npy')  # [33, 1, 45]
taxonomy_tensors = torch.FloatTensor(taxonomy_posess[:, 0])
taxonomy_tensors = taxonomy_tensors.unsqueeze(1).repeat(1, batch_size, 1)
 
for i in range(taxonomy_tensors.shape[0]):

    pose = taxonomy_tensors[i]
    poses = torch.cat((torch.zeros(1,3),pose),dim=1)
    
    # Initialize MANO layer
    mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=ncomps)

    # Generate random shape parameters
    random_shape = torch.rand(batch_size, 10)
    # Generate random pose parameters, including 3 values for global axis-angle rotation
    random_pose = torch.rand(batch_size, ncomps + 3)

    # Forward pass through MANO layer
    hand_verts, hand_joints = mano_layer(poses, random_shape)
    demo.display_hand({'verts': hand_verts, 'joints': hand_joints}, mano_faces=mano_layer.th_faces)
    