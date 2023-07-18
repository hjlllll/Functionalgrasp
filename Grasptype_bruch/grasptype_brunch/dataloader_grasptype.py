from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import pickle
import os
#from write_visualize import
import copy
'''
----2021-12-01----
取出带有label的标签进行数据预处理
'''

def augmentation_grasp(points, augment_rotation):#-----data augmentation------
    # Initialize rotation matrix
    R = np.eye(points.shape[1])

    if points.shape[1] == 3:
        if augment_rotation == 'vertical':

            # Create random rotations
            theta = np.random.rand() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        elif augment_rotation == 'all':

            # Choose two random angles for the first vector in polar coordinates
            theta = np.random.rand() * 2 * np.pi
            phi = (np.random.rand() - 0.5) * np.pi

            # Create the first vector in carthesian coordinates
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

    R = R.astype(np.float32)

    ####### Noise

    augment_noise = 0.005
    augment_trans = float(0.000) # no trans
    noise = (np.random.randn(points.shape[0], points.shape[1]) * augment_noise).astype(np.float32)

    ####### Rigid Transformation
    tran = (np.random.randn(points.shape[1]) * augment_trans).astype(np.float32)
    # tran1 = (torch.randn(points.shape[1]) * self.config.augment_trans).numpy().astype(np.float32)

    ####### Apply transforms
    # Do not use np.dot because it is multi-threaded
    # augmented_points = np.dot(points, R) * scale + noise
    augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) + noise + tran

    return augmented_points

def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3,
                  t7 - t9,
                  t11 + t12,
                  t7 + t9,
                  t1 + t2 * t15,
                  t19 - t20,
                  t11 - t12,
                  t19 + t20,
                  t1 + t2 * t24], axis=1)

    return np.reshape(R, (-1, 3, 3))


def load_data_hand(path): # 加载手的点云
    input_points = []
    input_labels = []

    frame_id = 0
    dl = 0.002
    for name in sorted(listdir(path)):  # # 在生成的抓取数据集文件夹中遍历数据文件
        if name.endswith('.pcd'):  # os.path.splitext(name)[1] == '.xml':
            print(
                '============================================{}================================================='.format(frame_id))
            print('Object No.{}, name is:{}'.format(frame_id, name))
            frame_id += 1
            object_name = name[:-4]
            #points, labels = load_data(path, object_name, is_show=is_show)

            file = open(path +object_name + '.pcd', 'r')
            data = file.readlines()
            file.close()

            for ii in range(5):
                if data[ii].split(' ')[0] == 'VERSION':
                    begin_num = ii
            if 'DATA ascii' != data[begin_num + 9].strip('\n'):
                raise ('Not a valid PCD header')
            is_labeled = data[begin_num + 1].split(' ')[
                             4] == 'label'  # 该参数为新数据（new）使用，当新数据没有分割标注时设为True；而训练（training）、验证（val）和测试（test）时，一定是用人工分割好的数据，这样才能将分割结果与标签对比；

            pts_num = data[begin_num + 7].strip('\n').split(' ')
            pts_num = eval(pts_num[-1])
            data_new = np.zeros((pts_num, 3)).astype(np.float32)
            label_new = np.zeros((pts_num, 1)).astype(np.int32)
            idx = -1
            for line in data[begin_num + 10:]:
                idx += 1
                line = line.strip('\n')
                xyzlabel = line.split(' ')
                if is_labeled:
                    x, y, z, label = [eval(i) for i in xyzlabel[:4]]
                    label_new[idx] = np.array([label]).astype(np.int32)
                else:
                    x, y, z = [eval(i) for i in xyzlabel]
                data_new[idx] = np.array([x, y, z]).astype(np.float32)
            # Subsample cloud
            # sub_points, sub_labels = grid_subsampling(data_new / 1000.0,
            #                                           labels=np.squeeze(label_new),
            #                                           sampleDl=dl)#dl采样参数
            # sub_labels = np.squeeze(sub_labels)  # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉,无用

            #is_label += [np.unique(labels, axis=0)]  # np.unique 去除数组中的重复数字，并进行排序之后输出
            # print('==========================', is_label, np.unique(labels, axis=0))

            # input_points += [data_new]#不采样
            # input_labels += [np.squeeze(label_new)]#不采样



        # -----训练时发现不能使用list，使用numpy类型-----
        else:
            data_new = None
            label_new = None

        input_points.append(data_new)
        input_labels.append(label_new)



    input_points = np.array(input_points)
    input_labels = np.array(input_labels)
    return input_points


def index_points(points, idx):

    #device = points#.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)#.to(device)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):

    #device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)#.to(device)
    distance = torch.ones(B, N) * 1e10#.to(device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long)#.to(device)
    batch_indices = torch.arange(B, dtype=torch.long)#.to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B,1,3)#.view(B, 1, 3)
        xyzs = torch.tensor(xyz)
        centroidss = torch.tensor(centroid)
        dist = torch.sum((xyzs - centroidss) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def sample_and_group(npoint, xyz):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    return new_xyz

def load_data_rt_mulitigrasp(mode): #读取上采样以后的数据
    hand_points = []
    with open(
            '/media/hjl/2b807652-227e-41f5-afff-0d749d7f5e23/hjl/hjl_transformer/labeled_data2/use/about_grasp/0.002_s420_465-100_ex-True_seg-True_train_record-is111111-new_withlabelgrasp-expanded-100-unsampled.pkl',
            'rb') as file:
        input_points,input_features,num_samples,grasp_obj_name,label_grasps = pickle.load(file)
        val_idx = np.array([4, 5, 8, 9, 17, 18, 27, 32, 35, 43, 45, 49, 53,
                            57, 60, 62, 68, 73, 74, 82, 87, 90, 101, 107, 113])  # 原始数据中验证集物体的编号

        val_idxs = copy.deepcopy(val_idx)
        for i in range(99):
            val_idxs = np.concatenate((val_idxs,val_idx+ (i+1)*129),axis=0)
        if mode == 'train':
            data_idx = np.setdiff1d(np.arange(len(grasp_obj_name)), val_idxs)
            assert len(grasp_obj_name) == 12900
        elif mode == 'test':
            data_idx = val_idxs
        else:
            raise ValueError('Wrong set, must be in [train, test]')

        input_points_, input_features_, grasp_obj_name_, label_grasps_ = [], [], [], []
        num_samples = data_idx.shape[0]
        for iidx in range(num_samples):
            input_points_ += [input_points[data_idx[iidx]]]
            input_features_ += [input_features[data_idx[iidx]]]
            grasp_obj_name_ += [grasp_obj_name[data_idx[iidx]]]
            label_grasps_ += [label_grasps[data_idx[iidx]]]
        input_points, input_features, grasp_obj_name, grasp_label = input_points_, input_features_, grasp_obj_name_, label_grasps_


        file_h = open('Functionalgrasp/dataset_hjl/labeled_hand/labeled_shadow_hand.pcd', 'r')
        data_h = file_h.readlines()
        file_h.close()

        for ii in range(5):
            if data_h[ii].split(' ')[0] == 'VERSION':
                begin_num = ii
        if 'DATA ascii' != data_h[begin_num + 9].strip('\n'):
            raise ('Not a valid PCD header')

        pts_num = data_h[begin_num + 7].strip('\n').split(' ')
        pts_num = eval(pts_num[-1])
        data_new = np.zeros((pts_num, 3)).astype(np.float32)
        idx = -1
        for line in data_h[begin_num + 10:]:
            idx += 1
            line = line.strip('\n')
            xyzlabel = line.split(' ')
            x, y, z, label = [eval(i) for i in xyzlabel[:4]]
            data_new[idx] = np.array([x, y, z]).astype(np.float32)
        data_new = torch.tensor(data_new,dtype=torch.float32)
        data_new = data_new.unsqueeze(0)
        if mode == 'train':
            hand_points = np.repeat(data_new,10400,axis=0)
        else:
            hand_points = np.repeat(data_new, 2500, axis=0)
    return input_points,input_features,num_samples,grasp_obj_name ,hand_points, label_grasps




class RTJlabel(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.input_points, self.input_features, \
        self.num_samples, self.grasp_obj_name ,self.handpoints, self.label_grasps = load_data_rt_mulitigrasp(mode)
        self.label_max = 198
        self.all_labels = True


        return
    def __len__(self):
        return self.num_samples # * self.label_max

    def __getitem__(self, idx_list):
        tp_list = []
        tf_list = []
        tl_list = []
        ti_list = []
        input_list = []
        p_ij = idx_list

        p_i = p_ij #// self.label_max
        p_j = p_ij % self.label_max

        # Get points and labels
        points = self.input_points[p_i].astype(np.float32)

        graspparts = self.input_features[p_i].astype(np.float32)[:, :16]

        label_grasp = self.label_grasps[p_i].astype(np.float32)
        tp_list += [points]
        tf_list += [graspparts]
        ti_list += [p_i]

        stacked_points = np.concatenate(tp_list, axis=0)
        model_inds = np.array(ti_list, dtype=np.int32)
        stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)
        hand = self.handpoints[p_i] * 0.1
        name = self.grasp_obj_name[p_i]
        stacked_features = np.concatenate(tf_list, axis=0)

        input_list += [stacked_points, stack_lengths, model_inds, hand, name, label_grasp]
        return input_list


    def get_grasp_label(self, labels, random=True, all_labels=False):
        if all_labels:
            label = labels
        else:
            if random:
                labels_len = len(labels)
                label_idx = torch.randint(0, labels_len, (1,)).item()
                label = labels[label_idx].reshape(1, -1)
            else:
                label = labels[0].reshape(1, -1)
        # print(labels_len, label_idx, type(label_idx))
        return label

class GraspNetCustomBatch:
    """Custom batch definition with memory pinning for GraspNet"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        # stacked_points, labels, stack_lengths, model_inds, hand, name
        L = (len(input_list) - 5) // 4  # 与网络层数有关，与input_list的形式有关

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = torch.tensor(input_list[ind],dtype=torch.float32)
        ind += 1
        self.lengths = input_list[ind]
        ind += 1
        self.model_inds = torch.tensor(input_list[ind],dtype=torch.float32)
        ind += 1
        self.hand = input_list[ind]
        ind += 1
        self.name = input_list[ind]
        ind += 1
        self.features = torch.tensor(input_list[ind],dtype=torch.float32)
        ind += 1
        self.grasps = torch.tensor(input_list[ind], dtype=torch.float32)


        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.grasps = self.grasps.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.model_inds = self.model_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.grasps = self.grasps.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.model_inds = self.model_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def GraspNetCollate(batch_data):
    return GraspNetCustomBatch(batch_data)


if __name__ == '__main__':

    path = 'Functionalgrasp/dataset_hjl/Grasp Dataset/grasps/'

    train_dataloader = DataLoader(RTJlabel(mode='train'), num_workers=8, batch_size=2,
                                  shuffle=False, drop_last=False)#, collate_fn=GraspNetCollate)
    for i, batch in enumerate(train_dataloader):#25个数qxqyqzqw,xyz,angle
        batch[0] = batch[0].cpu().detach().numpy()
        print(batch[0])
        print(batch[5].shape)

