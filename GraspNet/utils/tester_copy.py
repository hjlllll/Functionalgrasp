#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
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
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import sys

# Basic libs
import pickle
import os
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix
from utils.write_xml import write_xml
from utils.rewrite_xml import rewrite_xml
from utils.FK_model import fk_run, show_data_fast, show_data_fast_subfigure

#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def grasping_test_old(self, net, test_loader, config, debug=True):

        ############
        # Initialize
        ############

        test_grip_r = []
        test_grip_t = []
        test_grip_a = []
        test_grip_idx = []
        test_grip_labels = []
        test_obj_name = test_loader.dataset.grasp_obj_name
        if debug:
            batch_points = []
            batch_feature = []
            batch_jp = []
            batch_jp_label = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in test_loader:

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
                label_idx = torch.sqrt(
                    torch.sum(torch.square(outputs_ - batch.labels[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]]), 2)).argmin(
                    dim=1)  # 去除了label中多出来的倒数第三项
                label_ = batch.labels[torch.arange(batch.labels.shape[0]), label_idx][:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]]
            else:
                label_ = batch.labels[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]].squeeze(1)  # 去除了label中多出来的倒数第三项

            loss1 = net.loss(outputs_r, label_[:, :4])
            loss2 = net.loss(outputs_t, label_[:, 4:7])
            loss3 = net.loss(outputs_a, label_[:, 7:])
            outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)

            # Get probs and labels
            test_grip_r += [outputs_r.cpu().detach().numpy()]
            test_grip_t += [outputs_t.cpu().detach().numpy() / 5.0 * 1000]
            test_grip_a += [outputs_a.cpu().detach().numpy() * 1.5708]
            test_grip_labels += [label_.cpu().numpy()]
            test_grip_idx += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)
            if debug:
                outputs_FK = fk_run(outputs_r, outputs_t, outputs_a)  # [F, J+10, 3]  #原始J+1个关键点，加上10个关键点
                jj_p = outputs_FK[:, :27]  # [F, 10, 3]
                # outputs_FK = outputs_FK[:, 27:]  # [F, 10, 3]
                outputs_r, outputs_t, outputs_a = label_[:, :4], label_[:, 4:7], label_[:, 7:]  # # 显示预测结果的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
                outputs_FK_label = fk_run(outputs_r, outputs_t, outputs_a)  # [F, J+10, 3]  #原始J+1个关键点，加上10个关键点
                jj_p_label = outputs_FK_label[:, :27]  # [F, 10, 3]
                # outputs_FK_label = outputs_FK_label[:, 27:]  # [F, 10, 3]
                i_begin = 0
                for pi in range(batch.model_inds.shape[0]):
                    batch_points.append((batch.points[0][i_begin:i_begin + batch.lengths[0][pi]] * 1000))
                    # batch_feature.append((batch.features[i_begin:i_begin + batch.lengths[0][pi], 4:]))
                    batch_feature.append((batch.features[i_begin:i_begin + batch.lengths[0][pi], :]))
                    assert batch_feature[pi].shape[1] == 16
                    # print('0000000000000',jj_p[pi], '\n111111111',jj_p_label[pi])
                    batch_jp.append(jj_p[pi].detach())
                    # batch_jp_label.append(jj_p_label[pi])
                    i_begin = i_begin + batch.lengths[0][pi]


            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation loss : {}% of {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format((loss1 + loss2 + loss3) * 1000,
                                     100 * len(test_grip_idx) / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Stack all validation predictions
        test_grip_r = np.vstack(test_grip_r)
        test_grip_t = np.vstack(test_grip_t)
        test_grip_a = np.vstack(test_grip_a)
        test_grip_idx = np.hstack(test_grip_idx)
        test_grip_labels = np.vstack(test_grip_labels)

        #####################
        # Save predictions
        #####################
        grip_path = join(config.current_path, 'xml', 'test', 'epoch_' + str(self.epoch))
        if not os.path.exists(grip_path):
            os.makedirs(grip_path)
        grip_pkl = join(grip_path, 'grip_save_{}_test.pkl'.format(self.epoch))
        print('1111111111111111111111111111111111111', test_grip_r.shape, test_grip_t.shape, test_grip_a.shape, test_grip_idx.shape, len(test_obj_name), test_grip_labels.shape, len(batch_points), batch_points[0].shape)
        with open(grip_pkl, 'wb') as file:
            pickle.dump((test_grip_r, test_grip_t, test_grip_a, test_grip_idx, test_obj_name, test_grip_labels), file)
            print('save file to ', grip_pkl)

        app = QtGui.QApplication([])  # # 训练的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
        for i in range(test_grip_r.shape[0]):
            # print('22222222222222222222222222222222222222222222', grip_a[i].shape)
            # print('22222222222222222222222222222222222222222222', grip_a[i][17])
            write_xml(test_obj_name[test_grip_idx[i]], test_grip_r[i], test_grip_t[i], test_grip_a[i],
                      path=join(grip_path, 'epoch{}_{}_{}_test.xml'.format(self.epoch, test_grip_idx[i], i)))
            write_xml(test_obj_name[test_grip_idx[i]], test_grip_labels[i][:4], test_grip_labels[i][4:7] / 5.0 * 1000, test_grip_labels[i][7:] * 1.5708,
                      path=grip_path + '/epoch{}_{}_{}_test_label.xml'.format(self.epoch, test_grip_idx[i], i))
            if debug:
                print('begin debug')
                write_xml(test_obj_name[test_grip_idx[i]], test_grip_r[i], test_grip_t[i], test_grip_a[i], path='/home/lm/graspit/worlds/{}.xml'.format(i))
                rewrite_xml('/home/lm/graspit/worlds/{}.xml'.format(i))
                show_data_fast(batch_points[i], batch_feature[i], batch_jp[i], pi=i, show_jp=True)
                # write_xml(test_obj_name[test_grip_idx[i]], test_grip_labels[i][:4], test_grip_labels[i][4:7] / 5.0 * 1000, test_grip_labels[i][7:] * 1.5708,
                #           path='/home/lm/graspit/worlds/{}.xml'.format(i))
                # rewrite_xml('/home/lm/graspit/worlds/{}.xml'.format(i))
                # show_data_fast(batch_points[i], batch_feature[i], batch_jp_label[i], pi=i, show_jp=True)

        return loss1 + loss2 + loss3

    def grasping_test_without_data_only_show(self, net, test_loader, config, is_show=True):  # 同时展示pyqtgraph的点云抓取效果，和GraspIt的结果

        ############
        # Initialize
        ############

        test_grip_r = []
        test_grip_t = []
        test_grip_a = []
        test_grip_idx = []
        test_grip_labels = []
        test_obj_name = test_loader.dataset.grasp_obj_name
        test_dis_close = []
        if is_show:
            batch_points = []
            batch_feature = []
            batch_jp = []
            batch_jp_label = []
            show_fakelabel = True

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in test_loader:

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
                label_idx = torch.sqrt(
                    torch.sum(torch.square(outputs_ - batch.labels[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]]), 2)).argmin(
                    dim=1)  # 去除了label中多出来的倒数第三项
                label_ = batch.labels[torch.arange(batch.labels.shape[0]), label_idx][:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]]
            else:
                label_ = batch.labels[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]].squeeze(1)  # 去除了label中多出来的倒数第三项
                if batch.labels.shape[2] > 25:
                    kp_label = batch.labels[:, :, 25:].squeeze(1)

            loss1 = net.loss(outputs_r, label_[:, :4])
            loss2 = net.loss(outputs_t, label_[:, 4:7])
            loss3 = net.loss(outputs_a, label_[:, 7:])
            outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)

            # Get probs and labels
            test_grip_r += [outputs_r.cpu().detach().numpy()]
            test_grip_t += [outputs_t.cpu().detach().numpy() / 5.0 * 1000]
            test_grip_a += [outputs_a.cpu().detach().numpy() * 1.5708]
            test_grip_labels += [label_.cpu().numpy()]
            test_grip_idx += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)

            outputs_FK = fk_run(outputs_r, outputs_t, outputs_a)  # [F, J+10, 3]  #原始J+1个关键点，加上10个关键点
            if is_show:
                if not show_fakelabel:
                    jj_p = outputs_FK[:, :27]  # [F, 10, 3]
                    # outputs_FK = outputs_FK[:, 27:]  # [F, 10, 3]
                else:
                    outputs_r, outputs_t, outputs_a = label_[:, :4], label_[:, 4:7], label_[:, 7:]  # # 显示预测结果的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
                    outputs_FK_label = fk_run(outputs_r, outputs_t, outputs_a)  # [F, J+10, 3]  #原始J+1个关键点，加上10个关键点
                    jj_p_label = outputs_FK_label[:, :27]  # [F, 10, 3]
                    # outputs_FK_label = outputs_FK_label[:, 27:]  # [F, 10, 3]
                i_begin = 0
                for pi in range(batch.model_inds.shape[0]):
                    batch_points.append((batch.points[0][i_begin:i_begin + batch.lengths[0][pi]] * 1000))
                    # batch_feature.append((batch.features[i_begin:i_begin + batch.lengths[0][pi], 4:]))
                    batch_feature.append((batch.features[i_begin:i_begin + batch.lengths[0][pi], :]))
                    assert batch_feature[pi].shape[1] == 16
                    if not show_fakelabel:
                        # print('0000000000000',jj_p[pi], '\n111111111',jj_p_label[pi])
                        batch_jp.append(jj_p[pi].detach())
                    else:
                        batch_jp_label.append(jj_p_label[pi])
                    i_begin = i_begin + batch.lengths[0][pi]
                jj_p = 0
                jj_p_label = 0

            # # 计算接近距离
            outputs_FK = outputs_FK[:, :27]  # [F, 10, 3]
            batch_points_tensor = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), 3).cuda()  # [F, 20000, 3]
            batch_features_close = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), 5).cuda()  # [F, 20000, 5] 只取5个part的数据
            fetures_mask = [True, False, False, True, False, False, True, False, False, True, False, False, True, False, False, False]  # , False, False, False, False
            i_begin = 0
            for pi in range(batch.model_inds.shape[0]):
                batch_points_tensor[pi, :batch.lengths[0][pi]] = batch.points[0][i_begin:i_begin + batch.lengths[0][pi]] * 1000
                batch_features_close[pi, :batch.lengths[0][pi]] = batch.features[i_begin:i_begin + batch.lengths[0][pi], fetures_mask]
                i_begin = i_begin + batch.lengths[0][pi]

            batch_distance = ((batch_points_tensor.unsqueeze(2).expand(-1, -1, outputs_FK.shape[1], -1) - outputs_FK.unsqueeze(1).expand(-1, batch_points_tensor.shape[1], -1, -1)) ** 2).sum(-1).sqrt()  # [F, 20000, ?]
            batch_dis_close = batch_distance[:, :, [26, 21, 16, 11, 6]] * batch_features_close
            batch_dis_close[batch_features_close == 0] = float("inf")

            dis_close = torch.min(batch_dis_close, -2)[0]
            dis_close[dis_close == float("inf")] = 0
            mean_dis_close = dis_close.sum() / (dis_close.shape[0] * dis_close.shape[1])
            dis_close[dis_close >= 100] = 100
            test_dis_close += [dis_close.cpu().detach().numpy()]

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 0:
                last_display = t[-1]
                message = 'Validation loss : {:.1f}% / mean_dis_close={:.3f} / (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * len(test_grip_idx) / config.validation_size,
                                     mean_dis_close.item(),
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Stack all validation predictions
        test_grip_r = np.vstack(test_grip_r)
        test_grip_t = np.vstack(test_grip_t)
        test_grip_a = np.vstack(test_grip_a)
        test_grip_idx = np.hstack(test_grip_idx)
        test_grip_labels = np.vstack(test_grip_labels)
        test_dis_close = np.vstack(test_dis_close)

        singel_max_arg = np.argmax(test_dis_close, -1)
        singel_max = np.max(test_dis_close, -1)
        mean_dis_close_row = test_dis_close.sum(-1) / (test_dis_close != 0).sum(-1)
        mean_dis_close = mean_dis_close_row.sum() / mean_dis_close_row.shape[0]
        dis_result = np.concatenate([np.arange(singel_max_arg.shape[0])[:,None], singel_max_arg[:,None], singel_max[:,None], mean_dis_close_row[:,None]], -1)

        np.set_printoptions(suppress=True)
        print('the number, maximum value and mean value on each object:')
        print(dis_result)
        print('the worst finger index is:', np.argmax(np.bincount(singel_max_arg)), '\ncount is:', np.bincount(singel_max_arg))  # 0:thumb, 1:index, 2:middle, 3:ring, 4:little
        print('mean_dis_close is: ', mean_dis_close)

        #####################
        # Save predictions
        #####################
        grip_path = join(config.current_path, 'xml', 'test', 'epoch_{}_{}'.format(self.epoch, config.mode_set))
        if not os.path.exists(grip_path):
            os.makedirs(grip_path)
        print('grip_path is:', grip_path)
        np.savetxt(grip_path + '/dis_result_{}.txt'.format(config.mode_set), dis_result)
        with open(grip_path + '/number_maxidx_max_mean_{}_allmean{:.3f}.txt'.format(config.mode_set, mean_dis_close), "a") as filee:
            filee.write(str(dis_result))
            filee.write('\nthe worst finger index is:{}'.format(np.argmax(np.bincount(singel_max_arg))))
            filee.write('\ncount is: {}'.format(np.bincount(singel_max_arg)))
            filee.write('\nmean_dis_close is: {}'.format(mean_dis_close))
        grip_pkl = join(grip_path, 'grip_save_{}_test.pkl'.format(self.epoch))
        print('1111111111111111111111111111111111111', test_grip_r.shape, test_grip_t.shape, test_grip_a.shape, test_grip_idx.shape, len(test_obj_name), test_grip_labels.shape, len(batch_points), batch_points[0].shape)
        with open(grip_pkl, 'wb') as file:
            pickle.dump((test_grip_r, test_grip_t, test_grip_a, test_grip_idx, test_obj_name, test_grip_labels), file)
            print('save file to ', grip_pkl)

        app = QtGui.QApplication([])
        for i in range(0, test_grip_r.shape[0]):
        # for i in range(103, test_grip_r.shape[0]):
        # for i in [2,4,5,12,14,16,17,20,21,22,24,27,34,45,46,52, 57, 60, 62, 73, 74, 77, 78, 80]:  # 消融实验
        # for i in [2,6,8,12,13,26,27,31,32,61]:  #  pretrain对比
        # for i in [21,70,71,73,74,79,91]:  # good选择
            write_xml(test_obj_name[test_grip_idx[i]], test_grip_r[i], test_grip_t[i], test_grip_a[i],
                      path=join(grip_path, 'epoch{}_{}_{}_test.xml'.format(self.epoch, test_grip_idx[i], i)))
            write_xml(test_obj_name[test_grip_idx[i]], test_grip_labels[i][:4], test_grip_labels[i][4:7] / 5.0 * 1000, test_grip_labels[i][7:] * 1.5708,
                      path=grip_path + '/epoch{}_{}_{}_test_label.xml'.format(self.epoch, test_grip_idx[i], i))
            if is_show:
                print('begin show the object No.', i)
                if not show_fakelabel:
                    write_xml(test_obj_name[test_grip_idx[i]], test_grip_r[i], test_grip_t[i], test_grip_a[i], path='/home/lm/graspit/worlds/{}.xml'.format(i))
                    rewrite_xml('/home/lm/graspit/worlds/{}.xml'.format(i))
                    show_data_fast(batch_points[i], batch_feature[i], batch_jp[i], pi=i, show_jp=True)
                else:
                    write_xml(test_obj_name[test_grip_idx[i]], test_grip_labels[i][:4], test_grip_labels[i][4:7] / 5.0 * 1000, test_grip_labels[i][7:] * 1.5708,
                              path='/home/lm/graspit/worlds/{}.xml'.format(i))
                    rewrite_xml('/home/lm/graspit/worlds/{}.xml'.format(i))
                    show_data_fast(batch_points[i], batch_feature[i], batch_jp_label[i], pi=i, show_jp=True)

        return loss1 + loss2 + loss3

    def grasping_test(self, net, test_loader, config, is_show=True):  # 使用pyqtgraph同时展示原始分割物体和分割后的抓取物体，同时展示GraspIt的结果

        ############
        # Initialize
        ############

        test_grip_r = []
        test_grip_t = []
        test_grip_a = []
        test_grip_idx = []
        test_grip_labels = []
        test_obj_name = test_loader.dataset.grasp_obj_name
        test_dis_close = []
        if is_show:
            batch_points = []
            batch_feature = []
            batch_jp = []
            batch_jp_label = []
            show_fakelabel = False
        if config.mode_set == 'new':
            for ti in range(len(test_obj_name)):
                test_obj_name[ti] = test_obj_name[ti][:-4]
            print(test_obj_name)

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in test_loader:

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
                label_idx = torch.sqrt(
                    torch.sum(torch.square(outputs_ - batch.labels[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]]), 2)).argmin(dim=1)  # 去除了label中多出来的倒数第三项
                label_ = batch.labels[torch.arange(batch.labels.shape[0]), label_idx][:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]]
            else:
                label_ = batch.labels[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]].squeeze(1)  # 去除了label中多出来的倒数第三项
                if batch.labels.shape[2] > 25:
                    kp_label = batch.labels[:, :, 25:].squeeze(1)

            loss1 = net.loss(outputs_r, label_[:, :4])
            loss2 = net.loss(outputs_t, label_[:, 4:7])
            loss3 = net.loss(outputs_a, label_[:, 7:])
            outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)

            # Get probs and labels
            test_grip_r += [outputs_r.cpu().detach().numpy()]
            test_grip_t += [outputs_t.cpu().detach().numpy() / 5.0 * 1000]
            test_grip_a += [outputs_a.cpu().detach().numpy() * 1.5708]
            test_grip_labels += [label_.cpu().numpy()]
            test_grip_idx += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)

            outputs_FK = fk_run(outputs_r, outputs_t, outputs_a)  # [F, J+10, 3]  #原始J+1个关键点，加上10个关键点
            if is_show:
                if not show_fakelabel:
                    jj_p = outputs_FK[:, :27]  # [F, 10, 3]
                    # outputs_FK = outputs_FK[:, 27:]  # [F, 10, 3]
                else:
                    outputs_r, outputs_t, outputs_a = label_[:, :4], label_[:, 4:7], label_[:, 7:]  # # 显示预测结果的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
                    outputs_FK_label = fk_run(outputs_r, outputs_t, outputs_a)  # [F, J+10, 3]  #原始J+1个关键点，加上10个关键点
                    jj_p_label = outputs_FK_label[:, :27]  # [F, 10, 3]
                    # outputs_FK_label = outputs_FK_label[:, 27:]  # [F, 10, 3]
                i_begin = 0
                for pi in range(batch.model_inds.shape[0]):
                    batch_points.append((batch.points[0][i_begin:i_begin + batch.lengths[0][pi]] * 1000))
                    # batch_feature.append((batch.features[i_begin:i_begin + batch.lengths[0][pi], 4:]))
                    batch_feature.append((batch.features[i_begin:i_begin + batch.lengths[0][pi], :]))
                    assert batch_feature[pi].shape[1] == 16
                    if not show_fakelabel:
                        # print('0000000000000',jj_p[pi], '\n111111111',jj_p_label[pi])
                        batch_jp.append(jj_p[pi].detach())
                    else:
                        batch_jp_label.append(jj_p_label[pi])
                    i_begin = i_begin + batch.lengths[0][pi]
                jj_p = 0
                jj_p_label = 0

            # # 计算接近距离
            outputs_FK = outputs_FK[:, :27]  # [F, 10, 3]
            batch_points_tensor = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), 3).cuda()  # [F, 20000, 3]
            batch_features_close = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), 5).cuda()  # [F, 20000, 5] 只取5个part的数据
            fetures_mask = [True, False, False, True, False, False, True, False, False, True, False, False, True, False, False, False]  # , False, False, False, False
            i_begin = 0
            for pi in range(batch.model_inds.shape[0]):
                batch_points_tensor[pi, :batch.lengths[0][pi]] = batch.points[0][i_begin:i_begin + batch.lengths[0][pi]] * 1000
                batch_features_close[pi, :batch.lengths[0][pi]] = batch.features[i_begin:i_begin + batch.lengths[0][pi], fetures_mask]
                i_begin = i_begin + batch.lengths[0][pi]

            batch_distance = ((batch_points_tensor.unsqueeze(2).expand(-1, -1, outputs_FK.shape[1], -1) - outputs_FK.unsqueeze(1).expand(-1, batch_points_tensor.shape[1], -1, -1)) ** 2).sum(-1).sqrt()  # [F, 20000, ?]
            batch_dis_close = batch_distance[:, :, [26, 21, 16, 11, 6]] * batch_features_close
            batch_dis_close[batch_features_close == 0] = float("inf")

            dis_close = torch.min(batch_dis_close, -2)[0]
            dis_close[dis_close == float("inf")] = 0
            mean_dis_close = dis_close.sum() / (dis_close.shape[0] * dis_close.shape[1])
            dis_close[dis_close >= 100] = 100
            test_dis_close += [dis_close.cpu().detach().numpy()]

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 0:
                last_display = t[-1]
                message = 'Validation loss : {:.1f}% / mean_dis_close={:.3f} / (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * len(test_grip_idx) / config.validation_size,
                                     mean_dis_close.item(),
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Stack all validation predictions
        test_grip_r = np.vstack(test_grip_r)
        test_grip_t = np.vstack(test_grip_t)
        test_grip_a = np.vstack(test_grip_a)
        test_grip_idx = np.hstack(test_grip_idx)
        test_grip_labels = np.vstack(test_grip_labels)
        test_dis_close = np.vstack(test_dis_close)

        singel_max_arg = np.argmax(test_dis_close, -1)
        singel_max = np.max(test_dis_close, -1)
        mean_dis_close_row = test_dis_close.sum(-1) / (test_dis_close != 0).sum(-1)
        mean_dis_close = mean_dis_close_row.sum() / mean_dis_close_row.shape[0]
        dis_result = np.concatenate([np.arange(singel_max_arg.shape[0])[:,None], singel_max_arg[:,None], singel_max[:,None], mean_dis_close_row[:,None]], -1)

        np.set_printoptions(suppress=True)
        print('the number, maximum value and mean value on each object:')
        print(dis_result)
        print('the worst finger index is:', np.argmax(np.bincount(singel_max_arg)), '\ncount is:', np.bincount(singel_max_arg))  # 0:thumb, 1:index, 2:middle, 3:ring, 4:little
        print('mean_dis_close is: ', mean_dis_close)

        #####################
        # Save predictions
        #####################
        grip_path = join(config.current_path, 'xml', 'test', 'epoch_{}_{}'.format(self.epoch, config.mode_set))
        if not os.path.exists(grip_path):
            os.makedirs(grip_path)
        print('grip_path is:', grip_path)
        np.savetxt(grip_path + '/dis_result_{}.txt'.format(config.mode_set), dis_result)
        with open(grip_path + '/number_maxidx_max_mean_{}_allmean{:.3f}.txt'.format(config.mode_set, mean_dis_close), "a") as filee:
            filee.write(str(dis_result))
            filee.write('\nthe worst finger index is:{}'.format(np.argmax(np.bincount(singel_max_arg))))
            filee.write('\ncount is: {}'.format(np.bincount(singel_max_arg)))
            filee.write('\nmean_dis_close is: {}'.format(mean_dis_close))
        grip_pkl = join(grip_path, 'grip_save_{}_test.pkl'.format(self.epoch))
        print('1111111111111111111111111111111111111', test_grip_r.shape, test_grip_t.shape, test_grip_a.shape, test_grip_idx.shape, len(test_obj_name), test_grip_labels.shape, len(batch_points), batch_points[0].shape)
        with open(grip_pkl, 'wb') as file:
            pickle.dump((test_grip_r, test_grip_t, test_grip_a, test_grip_idx, test_obj_name, test_grip_labels), file)
            print('save file to ', grip_pkl)

        app = QtGui.QApplication([])
        mw = QtGui.QMainWindow()
        mw.setWindowTitle('show partnet results')
        mw.resize(1000, 500)
        for i in range(0, test_grip_r.shape[0]):
        # for i in range(60, test_grip_r.shape[0]):
        # for i in [2,4,5,12,14,16,17,20,21,22,24,27,34,45,46,52, 57, 60, 62, 73, 74, 77, 78, 80]:  # 消融实验
        # for i in [4, 5, 8, 9, 17, 18, 27, 32, 35, 43, 45, 49, 53, 57, 60, 62, 68, 73, 74, 82, 87, 90, 101, 107, 113]:  # 测试集
        # for i in [2,6,8,12,13,26,27,31,32,61]:  #  pretrain对比
        # for i in [53,56,57,80]:  # good选择
        # for i in [24,80,74,61,78]:  # 主图选择
        # for i in [13, 65, 46,47,48,49,50,51]:  # 可怜的相机, F6
            write_xml(test_obj_name[test_grip_idx[i]], test_grip_r[i], test_grip_t[i], test_grip_a[i],
                      path=join(grip_path, 'epoch{}_{}_{}_test.xml'.format(self.epoch, test_grip_idx[i], i)), mode=config.mode_set)
            write_xml(test_obj_name[test_grip_idx[i]], test_grip_labels[i][:4], test_grip_labels[i][4:7] / 5.0 * 1000, test_grip_labels[i][7:] * 1.5708,
                      path=grip_path + '/epoch{}_{}_{}_test_label.xml'.format(self.epoch, test_grip_idx[i], i), mode=config.mode_set)
            if is_show:
                print('begin show the object No.', i)
                if not show_fakelabel:
                    write_xml(test_obj_name[test_grip_idx[i]], test_grip_r[i], test_grip_t[i], test_grip_a[i], path='/home/lm/graspit/worlds/{}.xml'.format(i), mode=config.mode_set)
                    rewrite_xml('/home/lm/graspit/worlds/{}.xml'.format(i))
                    show_data_fast_subfigure(batch_points[i], batch_feature[i], batch_jp[i], mw, pi=i)  # 可选：i / None  (pi为None时，不显示Graspit!; 为 i 时显示当前抓取)
                else:
                    write_xml(test_obj_name[test_grip_idx[i]], test_grip_labels[i][:4], test_grip_labels[i][4:7] / 5.0 * 1000, test_grip_labels[i][7:] * 1.5708,
                              path='/home/lm/graspit/worlds/{}.xml'.format(i), mode=config.mode_set)
                    rewrite_xml('/home/lm/graspit/worlds/{}.xml'.format(i))
                    show_data_fast_subfigure(batch_points[i], batch_feature[i], batch_jp_label[i], mw, pi=i)  # 可选：i / None  (pi为None时，不显示Graspit!; 为 i 时显示当前抓取)

        return loss1 + loss2 + loss3

    def draw_pic(self, points, graspparts, w):
        colpart = np.unique(graspparts, axis=0)
        col = {0: [1, 1, 1, 0.4],  # 0
               1: [0, 0, 1, 1],  # 1
               2: [1, 0, 0, 1],  #[1, 0, 0, 1],  # 2
               4: [0, 1, 0, 1],  # 4
               7: [1, 1, 0, 1],  # 7
               10: [1, 0, 1, 1],  # 10
               13: [0, 1, 1, 1],  # 13
               16: [0.6, 0.3, 0.3, 1],  # 16
               5: [0.5, 0.5, 0, 1],  # 5
               'other': [0.5, 0.5, 0, 1]}

        print('colpart: {}'.format(colpart))

        points_list = []
        for j in range(colpart.shape[0]):
            ppp = []
            for i in range(points.shape[0]):
                if (graspparts[i] == colpart[j]).all():
                    ppp.append(points[i])
            ppp = np.concatenate(ppp, 0).reshape(-1, 3)
            points_list.append(ppp)

        p_sum = 0
        for i, ppp in enumerate(points_list):
            sp1 = gl.GLScatterPlotItem(pos=ppp, size=0.2 * (colpart[i] + 3), color=col[colpart[i]], pxMode=False)
            w.addItem(sp1)
            p_sum += ppp.shape[0]
            print(colpart[i], ': ', ppp.shape)
        print('p_sum', p_sum)

    def object_segmentation_test(self, net, val_loader, config, debug=False, is_show=False):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        # Number of classes predicted by the model
        # nc_model = config.num_classes - len(val_loader.dataset.ignored_labels)
        nc_model = config.num_classes - len(net.ign_lbls)

        #print(nc_tot)
        #print(nc_model)

        # Initiate global prediction over validation clouds
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model))
                                     for l in val_loader.dataset.input_labels]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in val_loader.dataset.label_values:
                if label_value not in val_loader.dataset.ignored_labels:
                    self.val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in val_loader.dataset.input_labels])
                    i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        data_for_show = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)
            acc = net.accuracy(outputs, batch.labels)
            with open(join(config.saving_path, 'test.txt'), "a") as file:
                message = 'step:{:02d}, model_inds:{}, acc={:3.0f}%\n'
                file.write(message.format(i, batch.model_inds.cpu().numpy(), 100*acc))
            predicted = torch.argmax(outputs.data, dim=1).cpu().detach().numpy()
            valid_labels = net.valid_labels[::-1]
            v_idx = np.arange(len(valid_labels))[::-1]
            for i, n in enumerate(valid_labels):
                predicted[predicted==v_idx[i]] = n

            # Get probs and labels
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            labels = batch.labels.cpu().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            m_inds = batch.model_inds.cpu().numpy()
            torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                target = labels[i0:i0 + length]
                probs = stacked_probs[i0:i0 + length]
                m_i = m_inds[b_i]
                # data for show
                data_for_show.append((batch.points[0].cpu().detach().numpy()[i0:i0 + length] * 1000.0, target, predicted[i0:i0 + length]))

                # Update current probs in whole cloud
                self.validation_probs[m_i] = val_smooth * self.validation_probs[m_i] + (1 - val_smooth) * probs

                # Stack all prediction for this epoch
                predictions.append(probs)
                targets.append(target)
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            last_display = t[-1]
            message = 'Validation : {:.1f}%, model_inds:{}, acc={:3.0f}% / (timings : {:4.2f} {:4.2f})'
            print(message.format(100 * i / config.validation_size,
                                 batch.model_inds.cpu().numpy(),
                                 100*acc,
                                 1000 * (mean_dt[0]),
                                 1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)


        t3 = time.time()

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        # for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
        #     if label_value in val_loader.dataset.ignored_labels:
        for l_ind, label_value in reversed(list(enumerate(net.lbl_values))):
            if label_value in net.ign_lbls:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)


        t4 = time.time()

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            # Name of saving file
            test_file = join(config.saving_path, 'val_IoUs.txt')

            # Line to write:
            line = ''
            for IoU in IoUs:
                line += '{:.3f} '.format(IoU)
            line = line + '\n'

            # Write in file
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

        t6 = time.time()

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))
        with open(join(config.saving_path, 'test.txt'), "a") as file:
            file.write('{:s} mean IoU = {:.1f}%\n'.format(config.dataset, mIoU))
        # write_ply() ??????

        # Display timings
        t7 = time.time()
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('Confs bis . {:.1f}s'.format(t4 - t3))
            print('IoU ....... {:.1f}s'.format(t5 - t4))
            print('Save1 ..... {:.1f}s'.format(t6 - t5))
            print('Save2 ..... {:.1f}s'.format(t7 - t6))
            print('\n************************\n')

        t8 = time.time()
        if is_show:
            app = QtGui.QApplication([])
            mw = QtGui.QMainWindow()
            mw.setWindowTitle('show partnet results')
            mw.resize(800, 400)

            for p_i in range(0, len(data_for_show)):
            # for p_i in range(78, len(data_for_show)):
                print('***=======------ object No.{} ------========***'.format(p_i))
                cw = QtGui.QWidget()
                mw.takeCentralWidget()
                mw.setCentralWidget(cw)
                l = QtGui.QHBoxLayout()  # QH水平  QV垂直
                cw.setLayout(l)

                w = gl.GLViewWidget()
                w.opts['distance'] = 300
                w.setWindowTitle('label')
                l.addWidget(w)

                w2 = gl.GLViewWidget()
                w2.opts['distance'] = 300
                w2.setWindowTitle('predction')
                l.addWidget(w2)

                mw.show()

                points, labels, probs = data_for_show[p_i]

                # draw points and labels
                print('=======- label -=========')
                self.draw_pic(points, labels, w)

                # draw points and probs
                print('=======- probs -=========')
                self.draw_pic(points, probs, w2)

                if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                    QtGui.QApplication.instance().exec_()

        return



















