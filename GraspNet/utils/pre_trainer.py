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
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys

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
from utils.FK_model import Shadowhand_FK#, show_data_fast
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
                                         # weight_decay=config.weight_decay)

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
            with open(join(config.saving_path, 'training.txt'), "w") as file:
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
                outputs_r, outputs_t, outputs_a = net(batch, config)
                outputs = torch.cat((outputs_r, outputs_t, outputs_a), 1)

                # # 角度loss
                angle_lower = torch.tensor([ 0., -15., -15.,   0., -15., -15.,   0., -15., -15.,   0., -15., -15.,   0., -60.,  0., -12, -30.,  0.]).cuda() / 90.0  # * 1.5708 / 1.5708
                angle_upper = torch.tensor([45.,  15.,  90., 180.,  15.,  90., 180.,  15.,  90., 180.,  15.,  90., 180.,  60., 70., 12.,  30., 90.]).cuda() / 90.0  # * 1.5708 / 1.5708
                angle_lower_pair = torch.zeros([2, outputs_a.reshape(-1).shape[0]]).cuda()
                angle_upper_pair = torch.zeros([2, outputs_a.reshape(-1).shape[0]]).cuda()
                angle_lower_pair[0] = angle_lower.repeat(outputs_a.shape[0]) - outputs_a.reshape(-1)
                angle_upper_pair[0] = outputs_a.reshape(-1) - angle_upper.repeat(outputs_a.shape[0])
                loss_angles = (torch.max(angle_lower_pair, 0)[0] + torch.max(angle_upper_pair, 0)[0]).sum()

                # # 预训练loss
                label_ = batch.labels[:, :, :25].squeeze(1)
                kp_label = batch.labels[:, :, 25:].squeeze(1)

                loss1 = net.loss(outputs_r, label_[:, :4])
                loss2 = net.loss(outputs_t, label_[:, 4:7])
                loss3 = net.loss(outputs_a, label_[:, 7:] * torch.tensor([1,1,1,1.8, 1,1,1.8, 1,1,1.8, 1,1,1.8, 1,1,1,1,1]).cuda())
                # loss = alpha1*loss1 + alpha2*loss2 + alpha3*loss3

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
                outputs_rotation[:, 0:3] = outputs_a[:, 0:3]
                outputs_rotation[:, 3] = outputs_a[:, 3] / 1.8
                outputs_rotation[:, 4] = 0.8 * outputs_a[:, 3] / 1.8
                outputs_rotation[:, 6:8] = outputs_a[:, 4:6] / 1.8
                outputs_rotation[:, 8] = outputs_a[:, 6] / 1.8
                outputs_rotation[:, 9] = 0.8 * outputs_a[:, 6] / 1.8
                outputs_rotation[:, 11:13] = outputs_a[:, 7:9]
                outputs_rotation[:, 13] = outputs_a[:, 9] / 1.8
                outputs_rotation[:, 14] = 0.8 * outputs_a[:, 9] / 1.8
                outputs_rotation[:, 16:18] = outputs_a[:, 10:12]
                outputs_rotation[:, 18] = outputs_a[:, 12] / 1.8
                outputs_rotation[:, 19] = 0.8 * outputs_a[:, 12] / 1.8
                outputs_rotation[:, 21:26] = outputs_a[:, 13:]  # all
                fk = Shadowhand_FK()
                outputs_FK = fk.run(outputs_base, outputs_rotation * 1.5708)  #[F, J+10, 3]  #原始J+1个关键点，加上10个关键点
                if debug:
                    jj_p = outputs_FK[:, :28]  # [F, 10, 3]  # # 训练的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
                    app = QtGui.QApplication([])  # # 训练的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊

                # outputs_FK = outputs_FK[:, [26, 21, 16, 11, 6]]  #[F, 5, 3]
                outputs_FK = outputs_FK[:, :28]  #[F, 28, 3]   # 详见onenete（有重复关节）
                # print('outputs_FK is :', outputs_FK.shape)

                # # 关键点约束: loss_kp
                loss_kp = net.loss(outputs_FK.reshape(kp_label.shape[0], -1), kp_label)

                # # 接近和远离约束: loss_close / loss_away
                # print(outputs_FK.shape)
                batch_points = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), 3).cuda()   #[F, 20000, 3] 为了批量计算，以batch中点数最多为初始化
                batch_features_close = torch.zeros(batch.model_inds.shape[0], int(max(batch.lengths[0])), 5).cuda()   #[F, 20000, 5] 只取5个part的数据
                batch_features_away = torch.ones(batch.model_inds.shape[0], int(max(batch.lengths[0])), 21).cuda()   #[F, 20000, 21]
                # fetures_mask = [False, False, False, False, True, False, False, True, False, False, True, False, False, True, False, False, True, False, False, False]  #, False, False, False, False
                fetures_mask = [True, False, False, True, False, False, True, False, False, True, False, False, True, False, False, False]  #, False, False, False, False
                i_begin = 0
                for pi in range(batch.model_inds.shape[0]):
                    batch_points[pi, :batch.lengths[0][pi]] = batch.points[0][i_begin:i_begin+batch.lengths[0][pi]] * 1000
                    batch_features_close[pi, :batch.lengths[0][pi]] = batch.features[i_begin:i_begin+batch.lengths[0][pi], fetures_mask]
                    batch_features_away[pi, :batch.lengths[0][pi]] = batch.features[i_begin:i_begin+batch.lengths[0][pi], [0,0,1,2,3,3,4,5,6,6,7,8,9,9,10,11,12,12,13,14,15]]  #扩展16位编码至21位
                    batch_features_away[pi] = (batch_features_away[pi]-1)**2
                    i_begin = i_begin+batch.lengths[0][pi]
                    if debug:
                        # # 训练的时候一定要注释掉啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
                        ppoints = batch.points[0][i_begin:i_begin + batch.lengths[0][pi]] * 1000
                        print('ppoints.shape', ppoints.shape)
                        ggrasppart = batch.features[i_begin:i_begin + batch.lengths[0][pi], fetures_mask]
                        print('ggrasppart.shape', ggrasppart.shape)
                        jj_pp = jj_p[pi]
                        print('jj_pp.shape', jj_pp.shape, 'jj_p.shape', jj_p.shape)
                        with open('grasp_obj_name_train.data', 'rb') as filehandle:
                            ggrasp_obj_name = pickle.load(filehandle)
                            print('object_idx&name:', pi, '--', ggrasp_obj_name[batch.model_inds[pi]])
                        llabel = label_[pi].clone().detach().cpu().numpy()
                        write_xml(ggrasp_obj_name[batch.model_inds[pi]], llabel[:4], llabel[4:7] / 5.0 * 1000.0, llabel[7:] * 1.5708, path='/home/lm/graspit/worlds/{}.xml'.format(pi), mode='train', rs=(21, 'pretrain_single'))
                        rewrite_xml('/home/lm/graspit/worlds/{}.xml'.format(pi))
                        #show_data_fast(ppoints, ggrasppart, jj_pp, pi=None, show_jp=True)

                batch_distance = ((batch_points.unsqueeze(2).expand(-1, -1, outputs_FK.shape[1], -1) - outputs_FK.unsqueeze(1).expand(-1, batch_points.shape[1], -1, -1))**2).sum(-1).sqrt()  #[F, 20000, ?]
                batch_dis_close = batch_distance[:, :, [27, 21, 16, 11, 6]] * batch_features_close  # 20210707多了个关节，序号变动
                batch_dis_away = batch_distance[:, :, [27, 25, 24, 23, 21, 20, 19, 18, 16, 15, 14, 13, 11, 10, 9, 8, 6, 5, 4, 3, 0]] * batch_features_away  # 20210707多了个关节，序号变动
                batch_dis_close[batch_features_close==0] = float("inf")
                batch_dis_away[batch_features_away==0] = float("inf")

                loss_close = torch.min(batch_dis_close, -2)[0] * torch.tensor([10,8,5,1,1]).cuda()
                loss_close[loss_close == float("inf")] = 0
                loss_close = loss_close.sum() / batch_dis_close.shape[0]
                loss_away = torch.log2((torch.tensor([10,10,15,20,10,10,15,20,10,10,15,20,10,10,15,20,10,10,15,20,60]).cuda()+5)/(torch.min(batch_dis_away, -2)[0] + 0.01))  # # 还未跑--编号a5
                loss_away = torch.tensor([1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,5]).cuda() * loss_away  # # 还未跑--编号a5
                loss_away[loss_away <= 0] = 0
                loss_away = loss_away.sum() / batch_dis_away.shape[0]

                # loss = 1*loss1 + 1*loss2 + 1*loss3  # 20210707
                loss = 1*loss1 + 1*loss2 + 1*loss3 + 0.00002*loss_close + 0.001*loss_away + 0.2*loss_angles + 0.00002*loss_kp  #20210708
                acc = loss
                t += [time.time()]

                # Backward + optimize
                loss.backward()


                if config.grad_clip_norm > 0:
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L=ang:{:.3f} + kp:{:.3f} + (r{:.3f}+t{:.3f}+a{:.3f}) + close:{:.3f} + away:{:.3f}={:.3f} acc={:3.1f}% / lr={} / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                    print(message.format(self.epoch, self.step,
                                         loss_angles.item(), loss_kp.item(),
                                         loss1.item(), loss2.item(), loss3.item(),
                                         loss_close.item(), loss_away.item(), loss.item(),
                                         100*acc, self.optimizer.param_groups[0]['lr'],
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         1000 * mean_dt[2]))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = 'e{:d} i{:d} loss_angle:{:.3f}, loss_kp:{:.3f}, loss_pre:{:.3f}-{:.3f}-{:.3f}:{:.3f}, loss_close:{:.3f}-_away:{:.3f}, acc:{:.3f}%, time:{:.3f}\n'
                        file.write(message.format(self.epoch, self.step,
                                                  loss_angles, loss_kp,
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

            if self.epoch % 20 == 10:
                grip_pkl = join(results_xml_path, 'train/grip_save_{}.pkl'.format(self.epoch))
                grip_r = outputs_r.clone().detach().cpu().numpy()
                grip_t = outputs_t.clone().detach().cpu().numpy() / 5.0 * 1000
                grip_a = outputs_a.clone().detach().cpu().numpy() * 1.5708
                grip_idx = batch.model_inds.clone().detach().cpu().numpy()
                grip_labels = label_.clone().detach().cpu().numpy()
                obj_name = training_loader.dataset.grasp_obj_name
                print('1111111111111111111111111111111111111', grip_r.shape,  grip_t.shape, grip_a.shape, grip_idx.shape, len(obj_name), grip_labels.shape)
                with open(grip_pkl, 'wb') as file:
                    pickle.dump((grip_r, grip_t, grip_a, grip_idx, obj_name, grip_labels), file)
                    print('save file to ', grip_pkl)

                for i in range(grip_r.shape[0]):
                    # print('22222222222222222222222222222222222222222222', grip_a[i].shape)
                    # print('22222222222222222222222222222222222222222222', grip_a[i][17])
                    write_xml(obj_name[grip_idx[i]], grip_r[i], grip_t[i], grip_a[i], path=results_xml_path + '/train/epoch{}_{}_{}.xml'.format(self.epoch, grip_idx[i], i), mode='train', rs=(21, 'pretrain_sum'))
                    write_xml(obj_name[grip_idx[i]], grip_labels[i][:4], grip_labels[i][4:7] / 5.0 * 1000, grip_labels[i][7:] * 1.5708, path=results_xml_path + '/train/epoch{}_{}_{}_label.xml'.format(self.epoch, grip_idx[i], i), mode='train', rs=(21, 'pretrain_single'))

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
            self.validation(net, val_loader, config, results_xml_path, epoch=self.epoch-1)
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

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs_r, outputs_t, outputs_a = net(batch, config)
            outputs = torch.cat((outputs_r, outputs_t, outputs_a), 1)

            label_ = batch.labels[:, :, :25].squeeze(1)

            loss1 = net.loss(outputs_r, label_[:, :4])
            loss2 = net.loss(outputs_t, label_[:, 4:7])
            loss3 = net.loss(outputs_a, label_[:, 7:] * torch.tensor([1,1,1,1.8, 1,1,1.8, 1,1,1.8, 1,1,1.8, 1,1,1,1,1]).cuda())
            outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)

            # Get probs and labels
            val_grip_r += [outputs_r.cpu().detach().numpy()]
            val_grip_t += [outputs_t.cpu().detach().numpy() / 5.0 * 1000]
            val_grip_a += [outputs_a.cpu().detach().numpy() * 1.5708]
            val_grip_labels += [label_.cpu().numpy()]
            val_grip_idx += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)

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

        if epoch % 20 == 10: # and epoch != 0
            # Stack all validation predictions
            val_grip_r = np.vstack(val_grip_r)
            val_grip_t = np.vstack(val_grip_t)
            val_grip_a = np.vstack(val_grip_a)
            val_grip_idx = np.hstack(val_grip_idx)
            val_grip_labels = np.vstack(val_grip_labels)

            #####################
            # Save predictions
            #####################
            grip_path = join(results_xml_path, 'val', 'epoch_'+str(epoch))
            if not os.path.exists(grip_path):
                os.makedirs(grip_path)
            grip_pkl = join(grip_path, 'grip_save_{}_val.pkl'.format(epoch))
            print('1111111111111111111111111111111111111', val_grip_r.shape, val_grip_t.shape, val_grip_a.shape, val_grip_idx.shape, len(val_obj_name), val_grip_labels.shape)
            with open(grip_pkl, 'wb') as file:
                pickle.dump((val_grip_r, val_grip_t, val_grip_a, val_grip_idx, val_obj_name, val_grip_labels), file)
                print('save file to ', grip_pkl)

            for i in range(val_grip_r.shape[0]):
                # print('22222222222222222222222222222222222222222222', grip_a[i].shape)
                # print('22222222222222222222222222222222222222222222', grip_a[i][17])
                write_xml(val_obj_name[val_grip_idx[i]], val_grip_r[i], val_grip_t[i], val_grip_a[i], path=join(grip_path, 'epoch{}_{}_{}_val.xml'.format(epoch, val_grip_idx[i], i)), mode='val', rs=(21, 'pretrain_sum'))
                write_xml(val_obj_name[val_grip_idx[i]], val_grip_labels[i][:4], val_grip_labels[i][4:7] / 5.0 * 1000, val_grip_labels[i][7:] * 1.5708,
                          path=grip_path + '/epoch{}_{}_{}_val_label.xml'.format(epoch, val_grip_idx[i], i), mode='val', rs=(21, 'pretrain_single'))

        return loss1 + loss2 + loss3
