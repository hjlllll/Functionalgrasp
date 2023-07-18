#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

from email.mime import base
from re import S

import torch

from models.blocks import *
import numpy as np
from os.path import exists, join
import pickle
import os, sys
from utils.FK_model import Shadowhand_FK#, show_data_fast
from utils.pointnet2_utils import PointNetSetAbstraction,PointNetSetAbstractionMsg
import  torch.nn.functional as F

os.chdir(sys.path[0])

def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)

def FK1(outputs_r,outputs_t,outputs_a):
    # # 输入正向运动学层
                # 正则化
                outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)
                # 3 + 4
                outputs_base = torch.cat((outputs_t / 5.0 * 1000, outputs_r), 1)
                # 17(18) -> 27(J)
                outputs_rotation = torch.zeros([outputs_a.shape[0], 27]).type_as(outputs_a)  # .cuda()
                # 20210706:因为graspit中shadowhand模型和运动学与真实手不一致，因此与预训练fk_cpu.py用的模型存在不同，
                #          目前有两种策略（详见onenote）：①网络预测指尖两关节和，两处模型不同，让他猜；②网络分别预测两关节，用loss进行约束。

    ###################################################
                outputs_rotation[:, 0:3] = outputs_a[:, 0:3]
                angle_2_pair = torch.ones([2, outputs_a.shape[0]]).cuda()
                angle_1_pair = torch.zeros([2, outputs_a.shape[0]]).cuda()
                angle_2_pair[0] = outputs_a[:, 3]
                angle_1_pair[0] = outputs_a[:, 3] - 1
                outputs_rotation[:, 3] = torch.min(angle_2_pair, 0)[0]
                outputs_rotation[:, 4] = torch.max(angle_1_pair, 0)[0]
                outputs_rotation[:, 6:8] = outputs_a[:, 4:6]
                angle_2_pair[0] = outputs_a[:, 6]
                angle_1_pair[0] = outputs_a[:, 6] - 1
                outputs_rotation[:, 8] = torch.min(angle_2_pair, 0)[0]
                outputs_rotation[:, 9] = torch.max(angle_1_pair, 0)[0]
                outputs_rotation[:, 11:13] = outputs_a[:, 7:9]
                angle_2_pair[0] = outputs_a[:, 9]
                angle_1_pair[0] = outputs_a[:, 9] - 1
                outputs_rotation[:, 13] = torch.min(angle_2_pair, 0)[0]
                outputs_rotation[:, 14] = torch.max(angle_1_pair, 0)[0]
                outputs_rotation[:, 16:18] = outputs_a[:, 10:12]
                angle_2_pair[0] = outputs_a[:, 12]
                angle_1_pair[0] = outputs_a[:, 12] - 1
                outputs_rotation[:, 18] = torch.min(angle_2_pair, 0)[0]
                outputs_rotation[:, 19] = torch.max(angle_1_pair, 0)[0]
                outputs_rotation[:, 21:26] = outputs_a[:, 13:]  # all


    ##########################################
               
                fk = Shadowhand_FK()
                outputs_FK = fk.run(outputs_base, outputs_rotation * 1.5708)
                return outputs_FK

class Finger_feature(nn.Module):  # 调用需要输入 输入、输出参数，部位总个数、normal_channel
    def __init__(self, normal_channel=False):  # numberclasses 物体种类/手指的数量
        super(Finger_feature, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = 0
        self.sa_finger = PointNetSetAbstraction(npoint=None, radius=None,
                                                nsample=None, in_channel=3,
                                                mlp=[256, 512, 1024], group_all=True)  # 全局


        # 128的维度还需要斟酌
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(512, 256, kernel_size=1)  # 测试修改维度用的

        # self.fc = nn.Linear(3,1024)

    def forward(self, xyz):
        # SA Layers  6个SA层
        # l0_points 是特征，l0_xyz是坐标
        l0_points =None
        l0_xyz = xyz.permute(0, 2, 1)
        _, global_feature = self.sa_finger(l0_xyz, l0_points)

        # global_features = self.fc(xyz)
        # global_features = torch.max(global_features, 1)[0]
        # global_feature = global_features.reshape(global_features.shape[0],global_features.shape[1],1)
        return global_feature

class SA_Layer(nn.Module):  # 改进后的offset-attention
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)  # 归一化
        # self.act = nn.ReLU()
        self.act = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)  # 两个矩阵维度必须为3 (b,c,n) (b,n,c)矩阵乘法

        attention = self.softmax(energy)  # softmax
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))  # normalization
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))  # off-set attention
        x = x + x_r
        return x, attention

class SA_Layer_1(nn.Module):
    def __init__(self, channels):
        super(SA_Layer_1, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        # self.act = nn.ReLU()
        self.act = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, xh, xo):
        xh_q = self.q_conv(xh).permute(0, 2, 1)
        xo_k = self.k_conv(xo)
        xo_v = self.v_conv(xo).permute(0, 2, 1)
        # 一个空的三维数组，存储每个手指和各个部位的相关性
        alpha = torch.bmm(xh_q, xo_k)
        attention = alpha
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        x_r = torch.bmm(attention, xo_v)
        x_r = x_r.permute(0, 2, 1)
        return x_r, attention


class Grasp_Transformer_Last(nn.Module):
    def __init__(self, channels):
        super(Grasp_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer_1(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x_h, x_o):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x_h.size()

        # B, D, N
        # x_h = F.relu(self.bn1(self.conv1(x_h)))
        # x_h = F.relu(self.bn2(self.conv2(x_h)))
        # x_o = F.relu(self.bn1(self.conv1(x_o)))
        # x_o = F.relu(self.bn2(self.conv2(x_o)))
        x1, attention = self.sa1(x_h, x_o)  # 四个self-attention层

        # ----------去掉两个salayer
        # x2,_ = self.sa2(x1)
        # x3,attention = self.sa3(x2)
        # x, attention = self.sa4(x3)
        x = x1  # 拼接

        return x, attention


class FingerEncoder(nn.Module):
    def __init__(self, finger_channels):
        super(FingerEncoder, self).__init__()
        self.grasp_last = Grasp_Transformer_Last(finger_channels)

    def forward(self, x1, x2):
        xyz_hand = x1
        xyz_object = x2
        x, attention = self.grasp_last(xyz_hand, xyz_object)
        return x, attention  # x是qk相乘之后和v相乘  attention是q和k相乘

class KPCNN_G(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super(KPCNN_G, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                layer,
                                                config))


            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim


            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim, 1024, use_bn=True, bn_momentum=0.1, no_relu=False)
        self.head_r = UnaryBlock(1024, 512, use_bn=True, bn_momentum=0.1, no_relu=False)
        self.head_t = UnaryBlock(1024, 512, use_bn=True, bn_momentum=0.1, no_relu=False)
        self.head_a = UnaryBlock(1024, 512, use_bn=True, bn_momentum=0.1, no_relu=False)
        self.output_r = UnaryBlock(512, 4, use_bn=False, bn_momentum=0, no_relu=True)
        self.output_t = UnaryBlock(512, 3, use_bn=False, bn_momentum=0, no_relu=True)
        self.output_a = UnaryBlock(512, 18, use_bn=False, bn_momentum=0, no_relu=True)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.MSELoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.chamfer_loss = 0
        self.rectangle_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()
        
        ##########
        #new network
        ##########
        self.linear = nn.Linear(512,1024)
        self.finger_feature = Finger_feature(normal_channel=False)
        self.thumb_encoder = FingerEncoder(1024)
        self.index_encoder = FingerEncoder(1024)
        self.other_encoder = FingerEncoder(1024)
        self.head_th = UnaryBlock(1024, 256, use_bn=True, bn_momentum=0.1, no_relu=False)
        self.th_256t64 = UnaryBlock(256, 64, use_bn=True, bn_momentum=0.1, no_relu=False)
        self.output_th = UnaryBlock(64, 5, use_bn=False, bn_momentum=0, no_relu=True)

        self.head_ind = UnaryBlock(1024, 256, use_bn=True, bn_momentum=0.1, no_relu=False)
        self.ind_256t64 = UnaryBlock(256, 64, use_bn=True, bn_momentum=0.1, no_relu=False)
        self.output_ind = UnaryBlock(64, 3, use_bn=False, bn_momentum=0, no_relu=True)

        self.head_other = UnaryBlock(1024, 512, use_bn=True, bn_momentum=0.1, no_relu=False)
        # self.other_256t64 = UnaryBlock(256, 64, use_bn=False, bn_momentum=0, no_relu=False)
        self.output_other = UnaryBlock(512, 10, use_bn=False, bn_momentum=0, no_relu=True)


        #############
        ##pointnet++ssg
        #############

        in_channel = 3 +16
        self.sa1 = PointNetSetAbstraction(npoint=512,radius=0.02,nsample=32,in_channel=in_channel,mlp=[64,64,128],group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128,radius=0.04,nsample=64,in_channel=128+3,mlp=[128,128,256],group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.4)
        # self.fc_r = nn.Linear(256, 4)
        # self.fc_t = nn.Linear(256, 3)

        return

    def forward(self, batch, config):
        xyz = batch.down_points.clone().detach()
        # grasp_types = batch.label_news.clone().detach()
        # pre_shapes = grasp_types.squeeze(1)
        B, _, _ = xyz.shape
        xyz = xyz.transpose(2, 1)
        norm = batch.down_point_2048s.clone().detach().transpose(2, 1) #16 bit code
        xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        xyz = l3_points.view(B, 1024)

        x_r = self.head_r(xyz, batch)
        r = self.output_r(x_r, batch)
        x_t = self.head_t(xyz, batch)
        t = self.output_t(x_t, batch)
        x_a = self.head_a(xyz,batch)
        a = self.output_a(x_a,batch)
        # r = r / (r.pow(2).sum(-1).sqrt()).reshape(-1, 1)
        grasp_types = batch.label_news.clone().detach()
        pre_shapes = grasp_types.squeeze(1)#/1.57
        index = torch.tensor([1,4,7,10])
        pre_shapes[:,index] = 0
        pre_shapes *= 0.0
        xo_global = xyz
        deta_angles = torch.zeros([xyz.shape[0],18], dtype=torch.float32).cuda()

        output_fk1 = FK1(r, t, pre_shapes)
        thumb_pose = output_fk1#[:, 22:28,:] #[b,6,3]
        thumb_feature = self.finger_feature(thumb_pose)
        xo_global = xo_global.unsqueeze(0)
        xo_global = xo_global.permute(1,2,0)
        th_object, _ = self.thumb_encoder(thumb_feature, xo_global)
        th_object = th_object.permute(0, 2, 1)#[B,1,1024]

        th_object = th_object.squeeze(1)
        th_angles = self.head_th(th_object, th_object.shape[0])
        th_angles = self.th_256t64(th_angles)
        th_angles = self.output_th(th_angles, th_angles.shape[0])
        deta_angles[:, 13:18] = th_angles

        pre_shapes2 = pre_shapes + deta_angles
        output_fk2 = FK1(r, t, pre_shapes2)
        index_pose = output_fk2#[:, 18:22, :]
        index_feature = self.finger_feature(index_pose)
        ind_object, _ = self.index_encoder(index_feature, xo_global)
        ind_object = ind_object.permute(0, 2, 1)
        ind_object = ind_object.squeeze(1)
        ind_angles = self.head_ind(ind_object, ind_object.shape[0])
        ind_angles = self.ind_256t64(ind_angles)
        ind_angles = self.output_ind(ind_angles, ind_angles.shape[0])

        deta_angles[:, 10:13] = ind_angles

        pre_shapes3 = pre_shapes + deta_angles
        output_fk3 = FK1(r, t, pre_shapes3)
        other_pose = output_fk3#[:, 1:17, :]
        other_feature = self.finger_feature(other_pose)
        other_object, _ = self.other_encoder(other_feature, xo_global)
        other_object = other_object.permute(0, 2, 1)
        other_object = other_object.squeeze(1)
        other_angles = self.head_other(other_object)
        other_angles = self.output_other(other_angles)
        deta_angles[::, :10] = other_angles[:][0]
        # final_angles = pre_shapes + 0.00000000000000001 * deta_angles #* 0.2 + a
        final_angles = pre_shapes + deta_angles #* 0.2 + a
        # final_angles = pre_shapes + deta_angles * 0.1
        # final_angles = deta_angles


        return r, t, final_angles

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # # Regularization of deformable offsets
        # if self.deform_fitting_mode == 'point2point':
        #     self.reg_loss = p2p_fitting_regularizer(self)
        # elif self.deform_fitting_mode == 'point2plane':
        #     raise ValueError('point2plane fitting mode not implemented yet.')
        # else:
        #     raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls, is_train=True):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.lbl_values = lbl_values
        self.ign_lbls = ign_lbls
        if is_train:
            self.C = len(self.lbl_values) - len(self.ign_lbls)
            if exists('./utils/seg_label_parameter.data'):
                with open('./utils/seg_label_parameter.data', 'rb') as f:
                    self_C, old_lbl_values, old_ign_lbls = pickle.load(f)
                    if self_C != self.C or old_lbl_values != lbl_values or old_ign_lbls != ign_lbls:
                        print('旧训练集中分类种类和新数据中分类种类不一致')
                        print('old is:{}; {}; {}'.format(self_C, old_lbl_values, old_ign_lbls))
                        print('new is:{}; {}; {}'.format(self.C, lbl_values, ign_lbls))
                        raise ValueError('KPconv源代码存在bug，新训练集中的标签种类与已训练好的网络模型中不一致，若有新分类进来只能重新训练，当前判断方式只能在数量不一致时报错，后续需要改变分类的训练和预测方法')
            else:
                with open('./utils/seg_label_parameter.data', 'wb') as f:
                    pickle.dump((self.C, self.lbl_values, self.ign_lbls), f)
        else:
            print('KPconv源代码存在bug，新训练集中的标签种类与已训练好的网络模型中不一致，当前只能测试已在训练中见过的分类，后续需要改变分类的训练和预测方法')
            if exists('./utils/seg_label_parameter.data'):
                with open('./utils/seg_label_parameter.data', 'rb') as f:
                    self.C, self.lbl_values, self.ign_lbls = pickle.load(f)
            else:
                raise ValueError('分割用的参数（self.C, lbl_values, ign_lbls）不明')

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in self.lbl_values if c not in self.ign_lbls])
        print('self.valid_labels is:', self.valid_labels)
        print('ign_lbls is:', self.ign_lbls)
        print('lbl_values is:', self.lbl_values)

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        self.output_loss = 0
        t_idxs = torch.unique(target, sorted=True)
        for t_i, t_idx in enumerate(t_idxs):
            t_mask = target == t_idx

            # Reshape to have a minibatch size of 1
            outputs_ = torch.transpose(outputs[t_mask], 0, 1)
            outputs_ = outputs_.unsqueeze(0)
            target_ = target[t_mask].unsqueeze(0)

            # Cross entropy loss
            self.output_loss += self.criterion(outputs_, target_)

        # # Regularization of deformable offsets
        # if self.deform_fitting_mode == 'point2point':
        #     self.reg_loss = p2p_fitting_regularizer(self)
        # elif self.deform_fitting_mode == 'point2plane':
        #     raise ValueError('point2plane fitting mode not implemented yet.')
        # else:
        #     raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss# + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total





















