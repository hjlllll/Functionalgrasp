#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
# from functools import cache
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys

from zmq import device

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config
from sklearn.neighbors import KDTree

from models.blocks import KPConv
from models.gan import Discriminator, compute_gradient_penalty
from utils.write_xml import write_xml
from utils.rewrite_xml import rewrite_xml
from utils.FK_model import Shadowhand_FK  # , show_data_fast
from pyqtgraph.Qt import QtCore, QtGui

heloss = nn.HingeEmbeddingLoss()
l2loss = nn.MSELoss()


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Epoch index
        self.epoch = 0
        self.step = 0

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        self.optimizer = torch.optim.Adam([{'params': other_params},
                                           {'params': deform_params, 'lr': deform_lr}],
                                          lr=config.learning_rate)
        # momentum=config.momentum,
        # weight_decay=config.weight_decay)          4

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config, debug=False):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:  # 创造训练文件
                file.write('epochs steps                 loss                   train_accuracy time\n')

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)

            # xml directory
            results_xml_path = join(config.saving_path, 'xml')
            if not exists(results_xml_path):
                # os.rename(results_path, results_path[:-2] + 'aaa-' + str(time.strftime("%Y-%m-%d-%H-%M-%S")))
                makedirs(results_xml_path + '/train')
                makedirs(results_xml_path + '/val')

        else:
            checkpoint_directory = None
            PID_file = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start training loop
        for epoch in range(config.max_epoch):

            # Remove File for kill signal
            if epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)

            self.step = 0
            for ib, batch in enumerate(training_loader):
                train_dis_close_interme = []
                # Check kill signal (running_PID.txt deleted)
                if config.saving and not exists(PID_file):
                    continue

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs_r, outputs_t, outputs_a = net(batch, config)  # 输出三个参数
                outputs = torch.cat((outputs_r, outputs_t, outputs_a), 1)

                # # 角度loss
                angle_lower = torch.tensor(
                    [0., -20., 0., 0., -20., 0., 0., -20., 0., 0., -20., 0., 0., -60., 0., -12, -30.,
                     0.]).cuda() / 90.0  # * 1.5708 / 1.5708
                angle_upper = torch.tensor(
                    [45., 20., 90., 180., 20., 90., 180., 20., 90., 180., 20., 90., 180., 60., 70., 12., 30.,
                     90.]).cuda() / 90.0  # * 1.5708 / 1.5708
                angle_lower_pair = torch.zeros(
                    [2, outputs_a.reshape(-1).shape[0]]).cuda()  # [2,12*18]  outputs_a = [12,18]
                angle_upper_pair = torch.zeros([2, outputs_a.reshape(-1).shape[0]]).cuda()
                angle_lower_pair[0] = angle_lower.repeat(outputs_a.shape[0]) - outputs_a.reshape(
                    -1)  # [12*18] - [12*18]
                angle_upper_pair[0] = outputs_a.reshape(-1) - angle_upper.repeat(outputs_a.shape[0])
                loss_angles = (torch.max(angle_lower_pair, 0)[0] + torch.max(angle_upper_pair, 0)[0]).sum()

                angle_lower1 = batch.label_news - 0.25  # [9,1,18]
                angle_lower1 = angle_lower1.squeeze(1)
                angle_upper1 = batch.label_news + 0.25  # [9,1,18]
                angle_upper1 = angle_upper1.squeeze(1)
                angle_lower_pair1 = torch.zeros([2, outputs_a.reshape(-1).shape[0]]).cuda()  # [2,162]
                angle_upper_pair1 = torch.zeros([2, outputs_a.reshape(-1).shape[0]]).cuda()  # [2,162]
                angle_lower_pair1[0] = angle_lower1.reshape(-1) - outputs_a.reshape(-1)
                angle_upper_pair1[0] = outputs_a.reshape(-1) - angle_upper1.reshape(-1)
                loss_angles1 = (torch.max(angle_lower_pair1, 0)[0] + torch.max(angle_upper_pair1, 0)[0]).sum()

                # # 四元数loss(没效果，反效果)
                rotate = (outputs_r ** 2).sum(-1)
                loss_rotate = l2loss(rotate, torch.ones_like(rotate))

                # # 预训练loss
                if batch.labels.shape[1] > 1:
                    outputs_ = outputs.clone().detach().unsqueeze(1)
                    label_idx = torch.sqrt(torch.sum(torch.square(outputs_ - batch.labels[:, :, :25]), 2)).argmin(dim=1)
                    label_ = batch.labels[torch.arange(batch.labels.shape[0]), label_idx][:, :25]
                else:
                    label_ = batch.labels[:, :, :25].squeeze(1)
                    if batch.labels.shape[2] > 25:
                        kp_label = batch.labels[:, :, 25:].squeeze(1)

                # loss_pre = net.loss(outputs, label_) * 10
                loss1 = net.loss(outputs_r, label_[:, :4])
                loss2 = net.loss(outputs_t, label_[:, 4:7])
                loss3 = net.loss(outputs_a, label_[:, 7:])

                # outputs_r, outputs_t, outputs_a = label_[:, :4], label_[:, 4:7], label_[:, 7:]  # # 训练的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
                # # 输入正向运动学层
                # 正则化
                outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)
                # 3 + 4
                outputs_base = torch.cat((outputs_t / 5.0 * 1000, outputs_r), 1)
                # 17(18) -> 27(J)
                outputs_rotation = torch.zeros([outputs_a.shape[0], 27]).type_as(outputs_a)  # .cuda()
                # 20210706:因为graspit中shadowhand模型和运动学与真实手不一致，因此与预训练fk_cpu.py用的模型存在不同，
                #          目前有两种策略（详见onenote）：①网络预测指尖两关节和，两处模型不同，让他猜；②网络分别预测两关节，用loss进行约束。
                ####################################3
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
                #########################
                # outputs_rotation[:, 0:4] = outputs_a[:, 0:4]
                # outputs_rotation[:, 4] = outputs_rotation[:, 3] * 0.8
                # outputs_rotation[:, 6:9] = outputs_a[:, 4:7]
                # outputs_rotation[:, 9] = outputs_rotation[:, 8] * 0.8
                # outputs_rotation[:, 11:14] = outputs_a[:, 7:10]
                # outputs_rotation[:, 14] = outputs_rotation[:, 13] * 0.8
                # outputs_rotation[:, 16:19] = outputs_a[:, 10:13]
                # outputs_rotation[:, 19] = outputs_rotation[:, 18] * 0.8
                # outputs_rotation[:, 21:26] = outputs_a[:, 13:]  # all

                fk = Shadowhand_FK()
                outputs_FK = fk.run(outputs_base, outputs_rotation * 1.5708)  # [F, J+10, 3]  #原始J+1个关键点，加上10个关键点
                if debug:
                    jj_p = outputs_FK[:,
                           :28]  # [F, 10, 3]  # # 训练的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
                    app = QtGui.QApplication(
                        [])  # # 训练的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊

                # # 手自碰撞约束loss_handself
                # 中指尖与其他指尖
                hand_self_distance0 = ((outputs_FK[:, [16]].unsqueeze(2).expand(-1, -1,
                                                                                outputs_FK[:, [27, 21, 11, 6]].shape[1],
                                                                                -1)
                                        - outputs_FK[:, [27, 21, 11, 6]].unsqueeze(1).expand(-1,
                                                                                             outputs_FK[:, [16]].shape[
                                                                                                 1], -1, -1)) ** 2).sum(
                    -1).sqrt().reshape(outputs_FK.shape[0], -1)
                hand_self_distance01 = ((outputs_FK[:, [15]].unsqueeze(2).expand(-1, -1,
                                                                                 outputs_FK[:, [26, 20, 10, 5]].shape[
                                                                                     1], -1)
                                         - outputs_FK[:, [26, 20, 10, 5]].unsqueeze(1).expand(-1,
                                                                                              outputs_FK[:, [15]].shape[
                                                                                                  1], -1,
                                                                                              -1)) ** 2).sum(
                    -1).sqrt().reshape(outputs_FK.shape[0], -1)
                # 拇指指尖+无名指指尖 vs 食指指尖+小拇指指尖
                hand_self_distance1 = (
                            (outputs_FK[:, [27, 11]].unsqueeze(2).expand(-1, -1, outputs_FK[:, [21, 6]].shape[1], -1)
                             - outputs_FK[:, [21, 6]].unsqueeze(1).expand(-1, outputs_FK[:, [27, 11]].shape[1], -1,
                                                                          -1)) ** 2).sum(-1).sqrt().reshape(
                    outputs_FK.shape[0], -1)
                # 拇指指尖+小拇指指尖 vs 食指指尖+无名指指尖
                hand_self_distance2 = (
                            (outputs_FK[:, [27, 6]].unsqueeze(2).expand(-1, -1, outputs_FK[:, [21, 11]].shape[1], -1)
                             - outputs_FK[:, [21, 11]].unsqueeze(1).expand(-1, outputs_FK[:, [27, 6]].shape[1], -1,
                                                                           -1)) ** 2).sum(-1).sqrt().reshape(
                    outputs_FK.shape[0], -1)
                # 小拇指2\3关节 vs 其他指尖
                hand_self_distance3 = ((outputs_FK[:, [27, 24]].unsqueeze(2).expand(-1, -1, outputs_FK[:,
                                                                                            [21, 20, 16, 11, 6]].shape[
                    1], -1)
                                        - outputs_FK[:, [21, 20, 16, 11, 6]].unsqueeze(1).expand(-1, outputs_FK[:,
                                                                                                     [27, 24]].shape[1],
                                                                                                 -1, -1)) ** 2).sum(
                    -1).sqrt().reshape(outputs_FK.shape[0], -1)
                hand_self_distance03 = (
                            (outputs_FK[:, [5, 4]].unsqueeze(2).expand(-1, -1, outputs_FK[:, [21, 16, 11]].shape[1], -1)
                             - outputs_FK[:, [21, 16, 11]].unsqueeze(1).expand(-1, outputs_FK[:, [5, 4]].shape[1], -1,
                                                                               -1)) ** 2).sum(-1).sqrt().reshape(
                    outputs_FK.shape[0], -1)  # 21,20,16,11,9,10,14,15
                hand_self_distance4 = ((outputs_FK[:, [5, 4]].unsqueeze(2).expand(-1, -1, outputs_FK[:,
                                                                                          [27, 21, 16, 11, 10]].shape[
                    1], -1)
                                        - outputs_FK[:, [27, 21, 16, 11, 10]].unsqueeze(1).expand(-1, outputs_FK[:,
                                                                                                      [5, 4]].shape[1],
                                                                                                  -1, -1)) ** 2).sum(
                    -1).sqrt().reshape(outputs_FK.shape[0], -1)
                hand_self_distance5 = ((outputs_FK[:, [26, 5]].unsqueeze(2).expand(-1, -1,
                                                                                   outputs_FK[:, [20, 10, 15]].shape[1],
                                                                                   -1)
                                        - outputs_FK[:, [20, 10, 15]].unsqueeze(1).expand(-1,
                                                                                          outputs_FK[:, [26, 5]].shape[
                                                                                              1], -1, -1)) ** 2).sum(
                    -1).sqrt().reshape(outputs_FK.shape[0], -1)
                # hand_self_distance5 *= 1.2
                hand_self_distance6 = ((outputs_FK[:, [5, 4]].unsqueeze(2).expand(-1, -1,
                                                                                  outputs_FK[:, [21, 20, 16, 11]].shape[
                                                                                      1], -1)
                                        - outputs_FK[:, [21, 20, 16, 11]].unsqueeze(1).expand(-1, outputs_FK[:,
                                                                                                  [5, 4]].shape[1], -1,
                                                                                              -1)) ** 2).sum(
                    -1).sqrt().reshape(outputs_FK.shape[0], -1)
                # hand_self_distance6 *= 1.35
                hand_self_distance7 = (
                            (outputs_FK[:, [15, 14]].unsqueeze(2).expand(-1, -1, outputs_FK[:, [10, 9]].shape[1], -1)
                             - outputs_FK[:, [10, 9]].unsqueeze(1).expand(-1, outputs_FK[:, [15, 14]].shape[1], -1,
                                                                          -1)) ** 2).sum(-1).sqrt().reshape(
                    outputs_FK.shape[0], -1)

                # hand_self_distance = torch.cat([hand_self_distance0, hand_self_distance1, hand_self_distance2, hand_self_distance6,hand_self_distance7], 1).reshape(-1)/30
                # hand_self_distance = torch.cat([hand_self_distance0, hand_self_distance1, hand_self_distance2, hand_self_distance5], 1).reshape(-1)/30
                # hand_self_distance = torch.cat([hand_self_distance0, hand_self_distance1, hand_self_distance2, hand_self_distance5], 1).reshape(-1)/30
                # hand_self_pair = torch.zeros([2, hand_self_distance.shape[0]]).cuda()
                # hand_self_pair[0] = 1 - hand_self_distance
                # loss_handself = torch.max(hand_self_pair, 0)[0].sum() / outputs_FK.shape[0]
                hand_self_distance = torch.cat(
                    [hand_self_distance0, hand_self_distance1, hand_self_distance2, hand_self_distance01,
                     hand_self_distance03, hand_self_distance5], 1).reshape(-1)  # /240

                hand_self_pair = torch.zeros([2, hand_self_distance.shape[0]]).cuda()
                hand_self_pair[0] = 22 - hand_self_distance
                loss_handself = torch.max(hand_self_pair, 0)[0].sum() / (outputs_FK.shape[0] * 28)

                # # 接近和远离约束: loss_close / loss_away
                outputs_FK = outputs_FK[:, :28]  # [F, 28, 3]
                # print(outputs_FK.shape)
                batch_points = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])),
                                           3).cuda()  # [F, 20000, 3] 为了批量计算，以batch中点数最多为初始化
                batch_features_close = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])),
                                                   5).cuda()  # [F, 20000, 5] 只取5个part的数据
                # batch_features_close = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), 16).cuda()   #[F, 20000, 5] 只取5个part的数据
                batch_features_away = torch.ones(batch.model_inds.shape[0], int(max(batch.lengths[0])),
                                                 21).cuda()  # [F, 20000, 21]
                # fetures_mask = [False, False, False, False, True, False, False, True, False, False, True, False, False, True, False, False, True, False, False, False]  #, False, False, False, False
                fetures_mask = [True, False, False, True, False, False, True, False, False, True, False, False, True,
                                False, False, False]  # , False, False, False, False
                # fetures_mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]  #, False, False, False, False
                i_begin = 0
                for pi in range(batch.model_inds.shape[0]):
                    batch_points[pi, :batch.lengths[0][pi]] = batch.points[0][
                                                              i_begin:i_begin + batch.lengths[0][pi]] * 1000
                    batch_features_close[pi, :batch.lengths[0][pi]] = batch.features[
                                                                      i_begin:i_begin + batch.lengths[0][pi],
                                                                      fetures_mask]
                    batch_features_away[pi, :batch.lengths[0][pi]] = batch.features[
                                                                     i_begin:i_begin + batch.lengths[0][pi],
                                                                     [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11,
                                                                      12, 12, 13, 14, 15]]  # 扩展16位编码至21位
                    # batch_features_away[pi, :batch.lengths[0][pi]] = batch.features[i_begin:i_begin+batch.lengths[0][pi], [0+4,0+4,1+4,2+4,3+4,3+4,4+4,5+4,6+4,6+4,7+4,8+4,9+4,9+4,10+4,11+4,12+4,12+4,13+4,14+4,15+4]]  #扩展16位编码至21位
                    batch_features_away[pi] = (batch_features_away[pi] - 1) ** 2
                    i_begin = i_begin + batch.lengths[0][pi]
                    # if debug:
                    #     # # 训练的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
                    #     ppoints = batch.points[0][i_begin:i_begin + batch.lengths[0][pi]] * 1000
                    #     print('ppoints.shape', ppoints.shape)
                    #     ggrasppart = batch.features[i_begin:i_begin + batch.lengths[0][pi], fetures_mask]
                    #     print('ggrasppart.shape', ggrasppart.shape)
                    #     jj_pp = jj_p[pi]
                    #     print('jj_pp.shape', jj_pp.shape, 'jj_p.shape', jj_p.shape)
                    #     with open('grasp_obj_name_train.data', 'rb') as filehandle:
                    #         ggrasp_obj_name = pickle.load(filehandle)
                    #         print('object_idx&name:', pi, '--', ggrasp_obj_name[batch.model_inds[pi]])
                    #     llabel = label_[pi].clone().detach().cpu().numpy()
                    #     write_xml(ggrasp_obj_name[batch.model_inds[pi]], llabel[:4], llabel[4:7] / 5.0 * 1000.0, llabel[7:] * 1.5708, path='/home/lm/graspit/worlds/{}.xml'.format(pi), mode='train', rs=(21, 'pretrain_single'))
                    #     rewrite_xml('/home/lm/graspit/worlds/{}.xml'.format(pi))
                    #     show_data_fast(ppoints, ggrasppart, jj_pp, pi=None, show_jp=True)

                batch_distance = ((batch_points.unsqueeze(2).expand(-1, -1, outputs_FK.shape[1],
                                                                    -1) - outputs_FK.unsqueeze(1).expand(-1,
                                                                                                         batch_points.shape[
                                                                                                             1], -1,
                                                                                                         -1)) ** 2).sum(
                    -1).sqrt()  # [F, 20000, ?]
                batch_dis_close = batch_distance[:, :, [27, 21, 16, 11, 6]] * batch_features_close  # 20210707多了个关节，序号变动
                # batch_dis_close = batch_distance[:, :, [27,26,24,21,20,19,16,15,14,11,10,9,6,5,4,1]] * batch_features_close  # 20210707多了个关节，序号变动
                batch_dis_away = batch_distance[:, :,
                                 [27, 26, 24, 23, 21, 20, 19, 18, 16, 15, 14, 13, 11, 10, 9, 8, 6, 5, 4, 3,
                                  0]] * batch_features_away  # 20210707多了个关节，序号变动
                batch_dis_close[batch_features_close == 0] = float("inf")
                batch_dis_away[batch_features_away == 0] = float("inf")

                # 与师兄对比mean_dis_close
                train_dis_close = torch.min(batch_dis_close, -2)[0]
                train_dis_close[train_dis_close == float("inf")] = 0
                # mean_train_dis_close = train_dis_close.sum() / (train_dis_close.shape[0] * train_dis_close.shape[1])
                train_dis_close[train_dis_close >= 100] = 100
                train_dis_close_interme += [train_dis_close.cpu().detach().numpy()]

                loss_close = torch.min(batch_dis_close, -2)[0] * torch.tensor([10, 8, 5, 2, 2]).cuda()
                # loss_close = torch.min(batch_dis_close, -2)[0] * torch.tensor([10,10,10,8,8,8,2,2,2,2,2,2,2,2,2,2]).cuda()
                loss_close[loss_close == float("inf")] = 0
                loss_close = loss_close.sum() / batch_dis_close.shape[0]

                loss_away = torch.log2((torch.tensor(
                    [10, 10, 15, 20, 10, 10, 15, 20, 10, 10, 15, 20, 10, 10, 15, 20, 10, 10, 15, 20,
                     60]).cuda() + 5) / (torch.min(batch_dis_away, -2)[0] + 0.01))  # # 还未跑--编号a5
                loss_away = torch.tensor(
                    [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 5]).cuda() * loss_away
                # loss_away = torch.log2((torch.tensor([10, 10, 15, 20, 10, 10, 15, 20, 10, 10, 15, 20, 10, 10, 15, 20, 10, 10, 15, 20,40]).cuda() + 5) / (torch.min(batch_dis_away, -2)[0] + 0.01))  # # 还未跑--编号a5
                # loss_away = torch.tensor([1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1]).cuda() * loss_away
                loss_away[loss_away <= 0] = 0
                loss_away = loss_away.sum() / batch_dis_away.shape[0]
                ########################################3
                # tip_points = outputs_FK[:, [27,21,16]]#.to(device)#.clone()#.detach() # [B,3,3] approach thumb,first finger,middle finger
                # tip_points_5 = outputs_FK[:, [27,21,16,11,6]]#.to(device)#.clone()#.detach() # [B,3,3] approach thumb,first finger,middle finger
                hand_points = outputs_FK  # [:, [27,21,16,11,6]]
                object_points = batch.down_points
                tip_points = hand_points[:, :, :].clone().detach()
                B = hand_points.size()[0]
                n1 = tip_points.size()[1]
                n2 = object_points.size()[1]
                #  #手的顶点，在第二维增加维度
                matrix1 = tip_points.unsqueeze(1).repeat(1, n2, 1, 1)
                #  #物体点，在第三维增加维度并复制
                matrix2 = 1000.0 * object_points.unsqueeze(2).repeat(1, 1, n1, 1)
                # h0 o0是手和物体到中心（0，0，0）的l2距离
                # h0_dists = torch.sqrt(((matrix1 - 0)**2).sum(-1))
                # o0_dists = torch.sqrt(((matrix2 - 0)**2).sum(-1))
                # ho是手和物体之间的距离
                ho_dists = torch.sqrt(((matrix1 - matrix2) ** 2).sum(-1))
                # ho_idx是和每个关节点最近的物体点云索引
                ho_idx = ho_dists.argmin(1)
                # ho_nearest是最近的物体点云的坐标
                ho_neatest = torch.zeros(B, n1, 3).cuda()
                for i in range(B):
                    ho_neatest[i, :, :] = 1000.0 * object_points[i, ho_idx[i, :], :].clone().detach()
                dists = torch.zeros(2, B, n1).cuda()
                h0_dists = torch.sqrt(((tip_points - 0) ** 2).sum(-1))
                ho_neatest_dists = torch.sqrt(((ho_neatest - 0) ** 2).sum(-1))
                dists[0] = ho_neatest_dists - h0_dists
                permute_dists = torch.max(dists, 0)[0]
                chamfer_loss = permute_dists.sum() / B

                # 20220809 new chamfer_loss
                # obj_hand = torch.zeros([2, matrix2.reshape(-1).shape[0]]).cuda()
                # obj_hand[0] = 0.001 * (matrix2.reshape(-1) - matrix1.reshape(-1))
                # chamfer_loss = torch.max(obj_hand,0)[0].sum()/ B

                # h0_dists = matrix1.pow(2).sum(-1).sqrt()
                # o0_dists = matrix2.pow(2).sum(-1).sqrt()
                # obj_hand = torch.zeros([2, h0_dists.reshape(-1).shape[0]]).cuda()
                # obj_hand[0] = 0.001 * (o0_dists.reshape(-1) - h0_dists.reshape(-1))
                # chamfer_loss = torch.max(obj_hand,0)[0].sum()/ B

                #######force closure loss#######
                # tip_pointss = outputs_FK[:, [27,21,16]]#.to(device)#.clone()#.detach() # [B,3,3] approach thumb,first finger,middle finger
                # tip_points_5 = outputs_FK[:, [27,21,16,11,6]]#.to(device)#.clone()#.detach() # [B,3,3] approach thumb,first finger,middle finger

                # obj_points = batch.down_point_2048s
                # obj_points = batch.down_points
                # obj_points = obj_points.cuda() * 1000.0#.clone()#.detach() * 1000.0# [B,2048,3]
                # select_points = tip_points.clone()
                # select_points = select_points.cuda()
                # loss_distance  = torch.zeros(tip_points.shape[0],1)
                # loss_distance = loss_distance.cuda()
                # #
                # #select closest 3 point on object
                # matrix1 = tip_points.unsqueeze(1).repeat(1,obj_points.shape[1],1,1)
                # matrix2 = obj_points.unsqueeze(2).repeat(1,1,tip_points.shape[1],1)
                # ho_distance = torch.sqrt(((matrix1 - matrix2)**2).sum(-1))
                # ho_idx = ho_distance.argmin(1)
                # #
                # for i in range(tip_points.shape[0]): #batch
                #     select_points[i,:,:] = obj_points[i,ho_idx[i,:],:]
                #
                # ###touching distance loss
                # dists = torch.zeros(2,tip_points.shape[0],3).cuda()
                # ho_dists = torch.sqrt(((tip_points - 0)**2).sum(-1))
                # ho_nearest_dists = torch.sqrt(((select_points - 0)**2).sum(-1))
                # dists[0] = ho_nearest_dists - ho_dists
                # permute_dist = torch.max(dists,0)[0]
                # permute_dists = permute_dist.sum()/permute_dist.shape[0]

                ####normal loss
                # loss_distance  = torch.zeros(tip_pointss.shape[0],1).cuda()
                # select_points = ho_neatest.clone()
                # points_nomal = batch.normals
                # points_nomal = points_nomal.cuda()
                # select_normals = tip_pointss.clone()
                # select_normals = select_normals.cuda()
                # loss_normal  = torch.zeros(tip_pointss.shape[0],1)
                # loss_normal = loss_normal.cuda()
                # for i in range(points_nomal.shape[0]):
                #     select_normals[i,:,:] = points_nomal[i,ho_idx[i,[27,21,16]],:]
                # for i in range(select_normals.shape[0]):
                #     normal_a = select_normals[i,0,:]
                #     normal_b = select_normals[i,1,:]
                #     normal_c = select_normals[i,2,:]
                #
                #     #判断法向量是否正常指向物体外侧
                #     direction_a =  torch.acos(torch.dot(normal_a,select_points[i,0,:]) / torch.sum(normal_a.pow(2)+1e-8,dim=0).sqrt() / torch.sum(select_points[i,0,:].pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     direction_b =  torch.acos(torch.dot(normal_b,select_points[i,1,:]) / torch.sum(normal_b.pow(2)+1e-8,dim=0).sqrt() / torch.sum(select_points[i,1,:].pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     direction_c =  torch.acos(torch.dot(normal_c,select_points[i,2,:]) / torch.sum(normal_c.pow(2)+1e-8,dim=0).sqrt() / torch.sum(select_points[i,2,:].pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     if direction_a > 1.57:
                #         normal_a = -normal_a
                #     if direction_b > 1.57:
                #         normal_b = -normal_b
                #     if direction_c > 1.57:
                #         normal_c = -normal_c
                #
                #
                #     ab = select_points[i,1,:] -select_points[i,0,:]
                #     bc = select_points[i,2,:] -select_points[i,1,:]
                #     ca = select_points[i,0,:] -select_points[i,2,:]
                #     m00 = torch.acos(torch.dot(normal_a,-ab) / torch.sum(normal_a.pow(2)+1e-8,dim=0).sqrt() / torch.sum(ab.pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     m01 = torch.acos(torch.dot(normal_a,ca) / torch.sum(normal_a.pow(2)+1e-8,dim=0).sqrt() / torch.sum(ca.pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     m10 = torch.acos(torch.dot(normal_b,ab) / torch.sum(normal_b.pow(2)+1e-8,dim=0).sqrt() / torch.sum(ab.pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     m11 = torch.acos(torch.dot(normal_b,-bc) / torch.sum(normal_b.pow(2)+1e-8,dim=0).sqrt() / torch.sum(bc.pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     m20 = torch.acos(torch.dot(normal_c,bc) / torch.sum(normal_c.pow(2)+1e-8,dim=0).sqrt() / torch.sum(bc.pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     m21 = torch.acos(torch.dot(normal_c,-ca) / torch.sum(normal_c.pow(2)+1e-8,dim=0).sqrt() / torch.sum(ca.pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     if m00 > 1.57:
                #         m00 = m00 - 1.57
                #     else:
                #         m00 = 0
                #     if m01 > 1.57:
                #         m01 = m01 - 1.57
                #     else:
                #         m01 = 0
                #     if m10 > 1.57:
                #         m10 = m10 - 1.57
                #     else:
                #         m10 = 0
                #     if m11 > 1.57:
                #         m11 = m11 - 1.57
                #     else:
                #         m11 = 0
                #     if m20 > 1.57:
                #         m20 = m20 - 1.57
                #     else:
                #         m20 = 0
                #     if m21 > 1.57:
                #         m21 = m21 - 1.57
                #     else:
                #         m21 = 0
                #     loss_normal[i,:] = m00 + m01 +m10 + m11 + m20 + m21
                #     l0 = torch.dot(ab,bc) / torch.sum(ab.pow(2)+1e-8,dim=0).sqrt() / torch.sum(bc.pow(2)+1e-8,dim=0).sqrt()
                #     l1 = torch.acos(torch.dot(ab,bc) / torch.sum(ab.pow(2)+1e-8,dim=0).sqrt() / torch.sum(bc.pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     l1 = 3.14 - l1
                #     l2 = torch.acos(torch.dot(bc,ca) / torch.sum(bc.pow(2)+1e-8,dim=0).sqrt() / torch.sum(ca.pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     l2 = 3.14-l2
                #     # l3 = torch.acos(torch.dot(ca,ab) / torch.sum(ca.pow(2)+1e-8,dim=0).sqrt() / torch.sum(ab.pow(2)+1e-8,dim=0).sqrt() )#弧度
                #     l3 = 3.14 - l1 -l2
                #     l = l0
                #     if l1 > l:
                #         l = l1
                #     elif l2>l:
                #         l = l2
                #     if l > 2.093:
                #         l = l - 2.093
                #     else:
                #         l = 0
                #     loss_distance[i,:] = l
                # loss_distances  = loss_distance.sum()/loss_distance.shape[0]
                # loss_normals = loss_normal.sum()/loss_normal.shape[0]

                # if epoch <20:
                #     loss = 0.1 * loss_close  + 1.0* loss_away + 10 * loss_angles + 10 * loss_handself# + 1.0* loss_distances + 1*loss_normals + 1*permute_dists  #  20210105
                # else:
                #     loss = 0.1 * loss_close  + 1.0* loss_away + 10 * loss_angles + 10 * loss_handself + 2.5* loss_distances + 2.5*loss_normals + 2.5*permute_dists  #  20210105
                # loss = 0.1 * loss_close  + 1.0* loss_away + 10 * loss_angles + 10 * loss_handself + 2.5* loss_distances + 2.5*permute_dists  + 2.5*loss_normals #  20220226
                # loss = 0.15 * loss_close  + 1.5* loss_away + 10 * loss_angles + 15 * loss_handself + 1.5*permute_dists5  #+ 0.1*loss_normals + 0.5* loss_distances  #  20220227
                # loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself  + 1.5 *chamfer_loss  #+ 0.1*loss_normals + 0.5* loss_distances  #  20220227

                # 20220723 distance03
                # if epoch < 100:
                #     loss = 0.15 * loss_close  + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself# + 1.0* loss_distances + 1*loss_normals + 1*permute_dists  #  20210105
                # elif epoch >=100 and epoch < 200:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself + 2.0* chamfer_loss
                # elif epoch >= 200 and epoch < 300:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself
                # else:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself + 5.0* chamfer_loss

                # 20220727 distance04
                # if epoch < 100:
                #     loss = 0.15 * loss_close  + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself# + 1.0* loss_distances + 1*loss_normals + 1*permute_dists  #  20210105
                # elif epoch >=100 and epoch < 200:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5* chamfer_loss
                # elif epoch >= 200 and epoch < 300:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself
                # else:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5* chamfer_loss  #+ 1.5 *chamfer_loss #+ 0.00001 * loss_normals + 0.000005 * loss_distances  #  20220227

                # if epoch < 250:
                #     loss = 0.1 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself
                # else:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.0 * chamfer_loss

                # 20220730 distance5
                # if epoch < 50:
                #     loss = 0.25 * loss_close + 0.5 * loss_away + 10 * loss_angles + 5 * loss_handself + 15.0 * chamfer_loss
                # elif epoch >= 50 and epoch < 100:
                #     loss = 0.2 * loss_close + 1.0 * loss_away + 10 * loss_angles + 5 * loss_handself + 5.0 * chamfer_loss
                # else:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5 * chamfer_loss

                # 20220801 only preshape first preshape + delta-angles*0.1 second only delta-angles 5000 distance5
                # loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5* chamfer_loss

                # 20220802 preshape*0.5 + delta-angles
                # if epoch < 100:
                #     loss = 0.15 * loss_close  + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself# + 1.0* loss_distances + 1*loss_normals + 1*permute_dists  #  20210105
                # elif epoch >=100 and epoch < 200:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself + 2.0* chamfer_loss
                # elif epoch >= 200 and epoch < 300:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself
                # else:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself + 5.0* chamfer_loss

                # 20220806 regulazation only delta-angles bad
                # loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles1 + 10 * loss_handself + 1.5* chamfer_loss

                # #20220809
                # if epoch < 50:
                #     loss = 0.1 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself #+ 15.0 * chamfer_loss
                # elif epoch >= 50 and epoch < 100:
                #     loss = 0.02 * loss_close + 0.2 * loss_away + 10 * loss_angles + 10 * loss_handself + 10.0 * chamfer_loss
                # else:
                #     loss = 0.1 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself #+ 15.0 * chamfer_loss
                # 20220810
                # loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5* chamfer_loss

                # 20220810
                # loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself #+ 1.5* chamfer_loss

                # 20220811 20220812
                # if epoch < 50:
                #     loss = 0.1 * loss_close + 2 * loss_away + 10 * loss_angles + 10 * loss_handself #+ 1.5* chamfer_loss
                # else:
                #     loss = 0.1 * loss_close + 2 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5* chamfer_loss

                # 20220813
                # if epoch < 100:
                #     loss = 0.15 * loss_close  + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself# + 1.0* loss_distances + 1*loss_normals + 1*permute_dists  #  20210105
                # elif epoch >=100 and epoch < 200:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself + 2.0* chamfer_loss
                # elif epoch >= 200 and epoch < 300:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself
                # else:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself + 5.0* chamfer_loss
                # loss = 0.08 * loss_close  + 1.0 * loss_away + 10 * loss_angles + 12 * loss_handself  + chamfer_loss# + 1.0* loss_distances + 1*loss_normals + 1*permute_dists  #  20210105

                # 20220814
                # if epoch < 100:
                #     loss = 0.15 * loss_close  + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself# + 1.0* loss_distances + 1*loss_normals + 1*permute_dists  #  20210105
                # elif epoch >=100 and epoch < 200:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself + 2.0* chamfer_loss
                # elif epoch >= 200 and epoch < 300:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself
                # else:
                #     loss = 0.15 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself + 5.0* chamfer_loss

                # 20220815 basic
                # if epoch < 100:
                #     loss = 0.1 * loss_close  + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself# + 1.0* loss_distances + 1*loss_normals + 1*permute_dists  #  20210105
                # elif epoch >=100 and epoch < 200:
                #     loss = 0.1 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself + 2.0* chamfer_loss
                # elif epoch >= 200 and epoch < 300:
                #     loss = 0.1 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself
                # else:
                #     loss = 0.1 * loss_close + 1.0 * loss_away + 10 * loss_angles + 10 * loss_handself + 5.0* chamfer_loss

                # 20220819
                # if epoch < 50:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself #+ 1.5* chamfer_loss
                # else:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5* chamfer_loss

                # 0911 new correct loss handself
                # loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself #+ loss_rotate#+ 1.5* chamfer_loss
                # if epoch % 199 == 1 and epoch<600:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5 * chamfer_loss
                # elif epoch % 199 != 1 and epoch < 600:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself
                # else:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5 * chamfer_loss

                # if epoch % 19 == 1 and epoch < 50:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5 * chamfer_loss
                # elif epoch % 19 != 1 and epoch < 50:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself
                # else:
                #     loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5 * chamfer_loss
                if epoch < 50:
                    loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself #+ 1.5* chamfer_loss
                else:
                    loss = 0.15 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5* chamfer_loss

                # all fingers
                # if epoch % 19 == 1 and epoch<50:
                #     loss = 0.05 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5 * chamfer_loss
                # elif epoch % 19 != 1 and epoch < 50:
                #     loss = 0.05 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself
                # else:
                #     loss = 0.05 * loss_close + 1.5 * loss_away + 10 * loss_angles + 10 * loss_handself + 1.5 * chamfer_loss
                acc = loss
                t += [time.time()]

                # Backward + optimize
                loss.backward()

                if config.grad_clip_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # 与师兄比较mean_dis_close
                train_dis_close_interme = np.vstack(train_dis_close_interme)
                mean_train_dis_close_row = train_dis_close_interme.sum(-1) / (train_dis_close_interme != 0).sum(-1)
                mean_train_dis_close = mean_train_dis_close_row.sum() / mean_train_dis_close_row.shape[0]

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L=ang:{:.3f} + self:{:.3f} + (r{:.3f}+t{:.3f}+a{:.3f}) + close:{:.3f} + away:{:.3f}={:.3f} acc={:3.1f}% / lr={} / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                    print(message.format(self.epoch, self.step,
                                         loss_angles.item(), loss_handself.item(),
                                         loss1.item(), loss2.item(), loss3.item(),
                                         loss_close.item(), loss_away.item(), loss.item(),
                                         100 * acc, self.optimizer.param_groups[0]['lr'],
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         1000 * mean_dt[2]))
                    # print('permute_dists,loss_120,loss_normals:',chamfer_loss,loss_distances,loss_normals)
                    print('permute_dists:', chamfer_loss)
                    # print('loss_normals:',loss_normals)
                    print('mean dis_close of current epoch:', mean_train_dis_close)

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = 'e{:d} i{:d} loss_angle:{:.3f}, loss_handself:{:.3f}, loss_pre:{:.3f}-{:.3f}-{:.3f}:{:.3f}, loss_close:{:.3f}-_away:{:.3f}, acc:{:.3f}%, time:{:.3f}\n'
                        file.write(message.format(self.epoch, self.step,
                                                  loss_angles, loss_handself,
                                                  loss1, loss2, loss3, loss,
                                                  loss_close, loss_away,  # net.reg_loss,
                                                  acc * 100.0,
                                                  t[-1] - t0))

                self.step += 1

            ##############
            # End of epoch
            ##############

            # Check kill signal (running_PID.txt deleted)
            if config.saving and not exists(PID_file):
                break

            # Update learning rate
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]
                    # print('11111111111111-=-=-=-=-=-222222222222222222-=-=-=-=-=-=333333333333333333 : ', param_group['lr'])

            if self.epoch % 30 == 0:
                # if self.epoch == 0:

                grip_pkl = join(results_xml_path, 'train/grip_save_{}.pkl'.format(self.epoch))
                grip_r = outputs_r.clone().detach().cpu().numpy()
                grip_t = outputs_t.clone().detach().cpu().numpy() / 5.0 * 1000
                grip_a = outputs_a.clone().detach().cpu().numpy() * 1.5708
                grip_idx = batch.model_inds.clone().detach().cpu().numpy()
                grip_labels = label_.clone().detach().cpu().numpy()
                obj_name = training_loader.dataset.grasp_obj_name
                print('1111111111111111111111111111111111111', grip_r.shape, grip_t.shape, grip_a.shape, grip_idx.shape,
                      len(obj_name), grip_labels.shape)
                with open(grip_pkl, 'wb') as file:
                    pickle.dump((grip_r, grip_t, grip_a, grip_idx, obj_name, grip_labels), file)
                    print('save file to ', grip_pkl)

                for i in range(grip_r.shape[0]):
                    # print('22222222222222222222222222222222222222222222', grip_a[i].shape)
                    # print('22222222222222222222222222222222222222222222', grip_a[i][17])
                    write_xml(obj_name[grip_idx[i]], grip_r[i], grip_t[i], grip_a[i],
                              path=results_xml_path + '/train/epoch{}_{}_{}.xml'.format(self.epoch, grip_idx[i], i),
                              mode='train', rs=(21, 'real'))
                    write_xml(obj_name[grip_idx[i]], grip_labels[i][:4], grip_labels[i][4:7] / 5.0 * 1000,
                              grip_labels[i][7:] * 1.5708,
                              path=results_xml_path + '/train/epoch{}_{}_{}_label.xml'.format(self.epoch, grip_idx[i],
                                                                                              i), mode='train',
                              rs=(21, 'pretrain_single'))

            # Update epoch
            self.epoch += 1

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path, _use_new_zipfile_serialization=False)

                # Save checkpoints occasionally
                if (self.epoch) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch))
                    torch.save(save_dict, checkpoint_path, _use_new_zipfile_serialization=False)

            # Validation
            net.eval()
            self.validation(net, val_loader, config, results_xml_path, epoch=self.epoch - 1)
            net.train()

        print('Finished Training')
        return

    # Validation methods ，目前的验证方法是和稳定抓取的标签做对比，即预训练的验证，后续开源代码应该去掉
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config, results_path, epoch=None):

        if config.dataset_task == 'regression':
            self.object_grasping_validation(net, val_loader, config, results_path, epoch)
        else:
            raise ValueError('No validation method implemented for this network type')

    def object_grasping_validation(self, net, val_loader, config, results_xml_path, epoch):
        """
        Perform a round of validation and show/save results
        :param net: network object
        :param val_loader: data loader for validation set
        :param config: configuration object
        """

        #####################
        # Network predictions
        #####################

        val_grip_r = []
        val_grip_t = []
        val_grip_a = []
        val_grip_idx = []
        val_grip_labels = []
        val_obj_name = val_loader.dataset.grasp_obj_name

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in val_loader:
            train_dis_close_interme = []
            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs_r, outputs_t, outputs_a = net(batch, config)
            outputs = torch.cat((outputs_r, outputs_t, outputs_a), 1)
            if batch.labels.shape[1] > 1:
                outputs_ = outputs.clone().detach().unsqueeze(1)
                label_idx = torch.sqrt(torch.sum(torch.square(outputs_ - batch.labels[:, :, 0:25]), 2)).argmin(
                    dim=1)  # 取最小的一个
                label_ = batch.labels[torch.arange(batch.labels.shape[0]), label_idx][:, 0:25]
            else:
                label_ = batch.labels[:, :, 0:25].squeeze(1)

            loss1 = net.loss(outputs_r, label_[:, :4])
            loss2 = net.loss(outputs_t, label_[:, 4:7])
            loss3 = net.loss(outputs_a, label_[:, 7:] * torch.tensor(
                [1, 1, 1, 1.8, 1, 1, 1.8, 1, 1, 1.8, 1, 1, 1.8, 1, 1, 1, 1, 1]).cuda())
            outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)
            outputs_base = torch.cat((outputs_t / 5.0 * 1000, outputs_r), 1)
            outputs_rotation = torch.zeros([outputs_a.shape[0], 27]).type_as(outputs_a)

            # Get probs and labels
            val_grip_r += [outputs_r.cpu().detach().numpy()]
            val_grip_t += [outputs_t.cpu().detach().numpy() / 5.0 * 1000]
            val_grip_a += [outputs_a.cpu().detach().numpy() * 1.5708]
            val_grip_labels += [label_.cpu().numpy()]
            val_grip_idx += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)
            ####################################3
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
            ################################################33
            # outputs_rotation[:, 0:3] = outputs_a[ :, 0:3]
            # outputs_rotation[:,3] = outputs_a[:,3]
            # outputs_rotation[:,4] = 0.8*outputs_a[:,3]
            #
            # outputs_rotation[ :, 6:8] = outputs_a[ :, 4:6]
            # outputs_rotation[ :, 8] = outputs_a[ :, 6]
            # outputs_rotation[ :, 9] = 0.8*outputs_a[ :, 6]
            #
            # outputs_rotation[ :, 11:13] = outputs_a[ :, 7:9]
            # outputs_rotation[ :, 13] = outputs_a[ :, 9]
            # outputs_rotation[ :, 14] = 0.8*outputs_a[ :, 9]
            #
            # outputs_rotation[ :, 16:18] = outputs_a[ :, 10:12]
            # outputs_rotation[ :, 18] = outputs_a[ :, 12]
            # outputs_rotation[ :, 19] = 0.8*outputs_a[ :, 12]
            # outputs_rotation[ :, 21:26] = outputs_a[ :, 13:]
            fk = Shadowhand_FK()
            outputs_FK = fk.run(outputs_base, outputs_rotation * 1.5708)
            outputs_FK = outputs_FK[:, :28]
            batch_points = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])),
                                       3).cuda()  # [F, 20000, 3] 为了批量计算，以batch中点数最多为初始化
            batch_features_close = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), 5).cuda()
            fetures_mask = [True, False, False, True, False, False, True, False, False, True, False, False, True, False,
                            False, False]
            i_begin = 0
            for pi in range(batch.model_inds.shape[0]):
                batch_points[pi, :batch.lengths[0][pi]] = batch.points[0][i_begin:i_begin + batch.lengths[0][pi]] * 1000
                batch_features_close[pi, :batch.lengths[0][pi]] = batch.features[i_begin:i_begin + batch.lengths[0][pi],
                                                                  fetures_mask]
                i_begin = i_begin + batch.lengths[0][pi]

            batch_distance = ((batch_points.unsqueeze(2).expand(-1, -1, outputs_FK.shape[1], -1) - outputs_FK.unsqueeze(
                1).expand(-1, batch_points.shape[1], -1, -1)) ** 2).sum(-1).sqrt()  # [F, 20000, ?]
            batch_dis_close = batch_distance[:, :, [27, 21, 16, 11, 6]] * batch_features_close
            batch_dis_close[batch_features_close == 0] = float("inf")
            # 与师兄对比mean_dis_close
            train_dis_close = torch.min(batch_dis_close, -2)[0]
            train_dis_close[train_dis_close == float("inf")] = 0
            # mean_train_dis_close = train_dis_close.sum() / (train_dis_close.shape[0] * train_dis_close.shape[1])
            train_dis_close[train_dis_close >= 100] = 100
            train_dis_close_interme += [train_dis_close.cpu().detach().numpy()]
            train_dis_close_interme = np.vstack(train_dis_close_interme)
            mean_train_dis_close_row = train_dis_close_interme.sum(-1) / (train_dis_close_interme != 0).sum(-1)
            mean_train_dis_close = mean_train_dis_close_row.sum() / mean_train_dis_close_row.shape[0]

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation loss : {}% of {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format((loss1 + loss2 + loss3) * 1000,
                                     100 * len(val_grip_idx) / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))
        if epoch % 1 == 0:
            print('the mean dis_close of current validation:', mean_train_dis_close)
        if epoch % 20 == 10:  # and epoch != 0
            # Stack all validation predictions
            val_grip_r = np.vstack(val_grip_r)
            val_grip_t = np.vstack(val_grip_t)
            val_grip_a = np.vstack(val_grip_a)
            val_grip_idx = np.hstack(val_grip_idx)
            val_grip_labels = np.vstack(val_grip_labels)

            #####################
            # Save predictions
            #####################
            grip_path = join(results_xml_path, 'val', 'epoch_' + str(epoch))
            if not os.path.exists(grip_path):
                os.makedirs(grip_path)
            grip_pkl = join(grip_path, 'grip_save_{}_val.pkl'.format(epoch))
            print('1111111111111111111111111111111111111', val_grip_r.shape, val_grip_t.shape, val_grip_a.shape,
                  val_grip_idx.shape, len(val_obj_name), val_grip_labels.shape)
            print('the mean dis_close of current validation:', mean_train_dis_close)
            with open(grip_pkl, 'wb') as file:
                pickle.dump((val_grip_r, val_grip_t, val_grip_a, val_grip_idx, val_obj_name, val_grip_labels), file)
                print('save file to ', grip_pkl)

            for i in range(val_grip_r.shape[0]):
                # print('22222222222222222222222222222222222222222222', grip_a[i].shape)
                # print('22222222222222222222222222222222222222222222', grip_a[i][17])
                write_xml(val_obj_name[val_grip_idx[i]], val_grip_r[i], val_grip_t[i], val_grip_a[i],
                          path=join(grip_path, 'epoch{}_{}_{}_val.xml'.format(epoch, val_grip_idx[i], i)), mode='val',
                          rs=(21, 'real'))
                write_xml(val_obj_name[val_grip_idx[i]], val_grip_labels[i][:4], val_grip_labels[i][4:7] / 5.0 * 1000,
                          val_grip_labels[i][7:] * 1.5708,
                          path=grip_path + '/epoch{}_{}_{}_val_label.xml'.format(epoch, val_grip_idx[i], i), mode='val',
                          rs=(21, 'pretrain_single'))

        return loss1 + loss2 + loss3
