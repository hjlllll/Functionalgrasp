# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.PartNet import *
from datasets.GraspNet import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN_G, KPFCNN
from train_GraspNet import GraspnetConfig


def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    # # # ===================================================== - Grasp - ========================================================
    # mode_set = 'new'  # 'val' 'new'
    # # chosen_log = '/media/ztq/new/ubuntu/ubuntu/Grasp/results/Log_2021-01-31_05-23-21-test-103-24-[nopre]'
    # # chosen_log = '/media/ztq/new/ubuntu/ubuntu/Grasp/results/Log_2021-02-02_09-57-22-test-103-24-[c1]-[nopre]-[newdata]'
    # # chosen_log = '/media/ztq/new/ubuntu/ubuntu/Grasp/results/olddata/Log_2021-01-06_09-15-30-[c1-a5]-[ispre]'
    # # chosen_log = '/media/ztq/new/ubuntu/ubuntu/Grasp/results/olddata/Log_2021-01-06_09-13-49-[c1-a5-10ang]-[ispre]'
    # # chosen_log = '/media/ztq/new/ubuntu/ubuntu/Grasp/results/olddata/Log_2021-01-05_02-41-50-[c1-a5-10ang-self2p]-[ispre]'  #  bad
    # chosen_log = '/media/ztq/new/ubuntu/ubuntu/Grasp/results/Log_2021-02-02_09-45-21-test-103-24-[newpre]'
    # # # # # #
    # # chosen_log = '/media/ztq/new/ubuntu/Graspnet-kpconv20201231-[c1-a3-ang-self2]/results/Log_2021-01-07_16-23-42-[c1-a5-10ang-self2p]-[ispre]-[newdata]'
    # # chosen_log = '/media/ztq/new/ubuntu/Graspnet-kpconv20201231-[c1-a3-ang-self2]/results/Log_2021-01-01_10-29-21-[c1-a3-10ang-self2]'
    # # chosen_log = '/media/ztq/new/ubuntu/Graspnet-kpconv20201231-[c1-a3-ang-self]/results/Log_2021-01-04_09-27-28-[c1-a5-10ang-self-rot-l3]'
    # # chosen_log = '/media/ztq/new/ubuntu/Graspnet-kpconv20201231-[c1-a3-ang-self2]/results/Log_2021-01-05_06-01-18-[c1-a5-10ang-self2p]-[nopre]'
    # # weiba = '1'
    # weiba = '1-new'

    # # ===================================================== - main - ========================================================
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None  # None

    # Choose to test on test split
    on_val = True

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

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

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)
    print('chosen_chkp is : ', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    if config.dataset == 'GraspNet':
        config.mode_set = mode_set
        # config.augment_noise = 0.0001
        # config.augment_symmetries = False
        config.batch_num = 5
        # config.in_radius = 4
        config.validation_size = 20
        config.input_threads = 10
        config.val_size = 0
        config.test_size = 25
        config.train_size = 104

        config.seed = 420
        config.is_aug = False
        config.current_path = chosen_log
        config.use_expand_data = True
        config.specify_dataset = True
        if config.specify_dataset:
            # 以训练、验证、测试以及new为关键字的字典，每项中包含该集的名称、use_seg_data的取值，以及object_label_dir的取值
            config.dataset_names = {
                #  '索引':['自定义或已生成的pkl文件名', use_seg_data值, object_label_dir值]
                # 'train': ['0.002_s420_465-100_ex-True_label-True_train_record-is11111{}.pkl'.format(weiba), True],
                # 'test': ['0.002_s420_465-100_ex-True_label-True_val_record.pkl', True, r'/home/lm/Data/labeled_data2/use'],
                # 'val': ['0.002_s420_465-100_ex-True_label-True_train_record-is11111{}.pkl'.format(weiba), True],  # 0.002_s420_465-100_ex-True_label-True_train_record-is11111{}.pkl
                # 'val': ['0.002_s420_465-100_ex-True_label-False_train_record.pkl', False],  # 0.002_s420_465-100_ex-True_label-True_train_record-no11111
                # 'val': ['0.002_s420_465-100_ex-True_label-False_val_record.pkl', False],
                # 'val': ['0.002_s420_465-100_ex-True_label-True_train_record-no111111.pkl', True],  # 0.002_s420_465-100_ex-True_label-True_train_record-no11111
                # 'val': ['0.002_s420_465-100_ex-True_label-True_val_record.pkl', True]
                # 'new': ['0.002_new.pkl', True, r'../Data/new1/seg_label']
                'new': ['real_model_before.pkl',
                        True, r'../Data/labeled_data2/use']

            }

    elif config.dataset == 'PartNet':
        config.validation_size = 200
        config.input_threads = 10
        config.chosen_log = chosen_log
        config.object_label_dir = '/home/GraspNet_kpconv/Data/new0/seg_label/'  # '/home/GraspNet_kpconv/Data/labeled_data2/use/'

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initiate dataset
    if config.dataset == 'PartNet':
        test_dataset = PartNetDataset(config, set=mode_set, is_show=False, is_aug=False)
        test_sampler = PartNetSampler(test_dataset, is_train=False)
        collate_fn = PartNetCollate
    elif config.dataset == 'GraspNet':
        test_dataset = GraspNetDataset(config, mode=mode_set)
        test_sampler = GraspNetSampler(test_dataset, use_potential=False, is_arange=True)
        collate_fn = GraspNetCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    print(config.dataset_task)
    if config.dataset_task == 'regression':
        net = KPCNN_G(config)
    elif config.dataset_task == 'segmentation':
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels, is_train=False)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # test
    if config.dataset_task == 'regression':
        tester.grasping_test(net, test_loader, config, is_show=True)
    elif config.dataset_task == 'segmentation':
        tester.object_segmentation_test(net, test_loader, config, is_show=True)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
