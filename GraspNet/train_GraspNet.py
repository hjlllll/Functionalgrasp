# -*- coding:utf-8 -*-#
#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on GraspNet dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Zhutq & Wrina - 15/11/2020
#
#  CUDA_LAUNCH_BLOCKING=1 python train_GraspNet.py

# Log_2022-03-03_10-57-07
#
#           Imports and global variables
#       \**********************************/
#

# import time
# print('sleep sleep sleep sleep sleep sleep sleep sleep sleep sleep')
# time.sleep(3600)

# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.GraspNet import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from utils.pre_trainer import ModelTrainer as pre_trainer
from models.architectures import KPCNN_G


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class GraspnetConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'GraspNet'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 10

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'global_average']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.002

    # Radius of convolution in "number grid cell". (2.5 is the standard value)  # ratio of rigid kp,固定核的卷积球半径相比方格采样尺寸的比例
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out  # ratio of deformable kp,形变核的卷积球半径相比方格采样尺寸的比例
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)  # The kernel points influence distance，每个核点的影响范围
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128  # Dimension of the first feature maps, for out_dim
    in_features_dim = 16   # Dimension of input features, == in_dim; 1, 4, 20
    num_classes = 4 + 3 + 18  # the last output dim, old:4 + 3 + 18 -> new:4 + 3 + 17

    # Can the network learn modulations  # choose if kernel weights are modulated in addition to deformed 选择是否除变形外还调制内核权重
    modulated = True

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.05

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 1020#2018

    # Learning rate management
    learning_rate = 1e-2#1e-2#1e-2
    # momentum = 0.98
    lr_decays = {i: 0.1**(1/500) for i in range(1, max_epoch)}
    # lr_decays = {i: 0.1**(1/500) for i in range(1, 3*max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 8#2#8#15#20#8

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 15#20

    # Number of epoch between each checkpoint
    checkpoint_gap = 100

    # Augmentations
    is_aug = False
    augment_scale_anisotropic = False  # anisotropic:各向异性
    augment_symmetries = [False, False, False]
    augment_rotation = 'none'
    augment_scale_min = 1.0
    augment_scale_max = 1.0
    augment_trans = 0.000  #0.05
    augment_noise = 0.000  #0.001

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None

    # dataset imformation
    object_dir = r'/home/lm/Documents/ddg_data/grasp_dataset/good_shapes'
    grasp_dir = r'/home/lm/Documents/ddg_data/grasp_dataset/grasps'
    objectEX_dir = r'../Data/expand_all'  # 在预训练时使用，因为预训练使用的是全部物体
    object_label_dir = r'../Data/labeled_data2/use'
    print(object_label_dir)

    # grasp_dir = r'/home/lm/Documents/ddg_data/grasp_dataset/ggg'
    use_expand_data = True
    use_seg_data = True

    is_pretrain = False #如果要预训练，则该参数设为True，然后再依据提示去修改后面的previous_training_path 和 use_pretrain
    if is_pretrain:
        use_seg_data = False

    if use_seg_data:
        use_expand_data = True

    specify_dataset = True
    if specify_dataset:
        # 以训练、验证和测试为关键字的字典，每项中包含该集的名称，以及use_seg_data的取值，
        if is_pretrain:
            dataset_names = {
                # '索引': ['自定义或已生成的pkl文件名', use_seg_data值, object_label_dir值]
                'train': ['0.002_s420_465-100_ex-True_seg-False_train_record.pkl', False, r'../Data/for_pretrain'],
                'val': ['0.002_s420_465-100_ex-True_seg-False_val_record.pkl', False, r'../Data/for_pretrain']}
        else:
            dataset_names = {
                # '索引': ['自定义或已生成的pkl文件名', use_seg_data值, object_label_dir值]
                'train': ['0.002_s420_465-100_ex-True_seg-True_train_record-is111111-new_one_grasptype20220118_downsampled_addhandpoints_normals_20220911.pkl', True, r'../Data/labeled_data2/use'],  # 0.002_s420_465-100_ex-True_label-True_train_record-is11111
                'val': ['0.002_s420_465-100_ex-True_seg-True_train_record-is111111-new_one_grasptype20220118_downsampled_addhandpoints_normals.pkl', True, r'../Data/labeled_data2/use'],  # 0.002_s420_465-100_ex-True_label-True_train_record-no11111
                'test': ['0.002_s420_465-100_ex-True_seg-True_val_record.pkl', True, r'../Data/labeled_data2/use'],
                'new': ['0.002_new.pkl', True, r'../Data/new0/seg_label']}

    num_objects = 565
    train_size = 465  # 465
    test_size = 100   # 如果test_size + train_size == num_objects, 则验证集为测试集
    val_size = num_objects - train_size - test_size  # if val_size==0, 则自动将test设为验证集
    seed = 420


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############


    # Choose here if you want to start training from a previous snapshot (None for new training)
    # 使用说明：
    # 预训练时，上面config中is_pretrain设为True；若无需加载之前的网络模型，则下面previous_training_path设为‘’；若batch需要从0开始，use_pretrain设为True
    # 正规训练时，is_pretrain设为False，则previous_training_path为预训练模型所在文件夹，use_pretrain设为True则batch从0开始；
    # 若训练中断，改变previous_training_path为中断前模型所在文件夹，use_pretrain设为False使batch从中断位置开始
    #previous_training_path = 'A-pre-[gen_fea-loss123-ang-kp-c-a]-[new]'  # 'allloss-[pre-kp-c-a1]-[new]'
    use_pretrain = True
    previous_training_path='/home/GraspNet_kpconv/Graspnet-kpconv-hjl/results/Log_2022-09-12_12-08-09'
    # previous_training_path = ''
    # use_pretrain = False  # False决定batch是继续之前的，还是True=从0开始

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None

    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = GraspnetConfig()
    if previous_training_path and (not use_pretrain):
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None

    # Get path from argument if given
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]

    # Initialize datasets
    training_dataset = GraspNetDataset(config, mode='train')
    test_dataset = GraspNetDataset(config, mode='val')

    # Initialize samplers
    training_sampler = GraspNetSampler(training_dataset, use_potential=False)
    test_sampler = GraspNetSampler(test_dataset, use_potential=True)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=GraspNetCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=GraspNetCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader)
    test_sampler.calibration(test_loader)

    #debug_timing(test_dataset, test_sampler, test_loader)
    #debug_show_clouds(training_dataset, training_sampler, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPCNN_G(config)

    # Define a trainer class
    if config.is_pretrain:
        print('here is pretraining -=- here is pretraining -=- here is pretraining')
        trainer = pre_trainer(net, config, chkp_path=chosen_chkp, finetune=use_pretrain)
    else:
        trainer = ModelTrainer(net, config, chkp_path=chosen_chkp, finetune=use_pretrain)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    try:
        trainer.train(net, training_loader, test_loader, config)
    except:
        print('Caught an error in : {}\n'.format(os.getpid()))
        os.kill(os.getpid(), signal.SIGINT)
        # sys.exit(1)

    print('Forcing exit now')
    # os.kill(os.getpid(), signal.SIGINT)
    print(os.getpid())
    sys.exit(0)


