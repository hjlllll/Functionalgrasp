from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import pickle
import os
'''
----2021-12-01----
取出带有label的标签进行数据预处理
'''
#读取上采样的文件txt，写入
with open('/home/hjl/hjl_transformer/labeled_data2/use/about_grasp/0.002_s420_465-100_ex-True_seg-True_train_record-is111111-new.pkl', 'rb') as file:
    input_points, input_features, input_labels, num_samples, grasp_obj_name = pickle.load(file)
#print(input_points,input_features,input_labels,num_samples,grasp_obj_name)
# path='/home/hjl/hjl_transformer/dataset_hjl/expand_all/'
# for j in range(len(grasp_obj_name)):
#     #print(grasp_obj_name[j])
#     for name in sorted(listdir(path)):
#         if name.split('_scaled')[0] == grasp_obj_name[j]:
#             file = open(path+name,'r')
#             data = file.readlines()
#             file.close()
#             idx = -1
#             a = []
#             for line in data[0:]:
#                 idx += 1
#                 line.strip('\n')
#                 xyz = line.split(' ')
#                 x,y,z = [eval(i) for i in xyz[:3]]
#                 a.append(np.array([x,y,z]).astype(np.float32))
#             b = np.array(a)
#             input_points[j] = b * 0.001
#             #print(b.shape)
# filename_pkl = '/home/hjl/hjl_transformer/about_grasp/0.002_s420_465-100_ex-True_seg-False_train_record_upsampled.pkl'
# with open(filename_pkl, 'wb') as file:
#     pickle.dump((input_points, input_labels, num_samples, grasp_obj_name), file)
#     print('save file to ', filename_pkl)
#

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

def load_data_rt_mulitigrasp(): #读取上采样以后的数据
    hand_points = []
    with open(
            '/home/hjl/hjl_transformer/labeled_data2/use/about_grasp/0.002_s420_465-100_ex-True_seg-True_train_record-is111111-new.pkl',
            'rb') as file:
        input_points,input_features,input_labels,num_samples,grasp_obj_name = pickle.load(file)






        file_h = open('/home/hjl/hjl_transformer/dataset_hjl/labeled_hand/labeled_shadow_hand.pcd', 'r')
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
        #print(data_new.shape)
        #input_points = data_new
        data_new = torch.tensor(data_new,dtype=torch.float32)
        data_new = data_new.unsqueeze(0)
        hand_points = np.repeat(data_new,129,axis=0)

        #hand_points.append(input_points)
    return input_points,input_features,input_labels,num_samples,grasp_obj_name ,hand_points




class RTJlabel(Dataset):
    def __init__(self):
        self.input_points, self.input_features, self.input_labels, \
        self.num_samples, self.grasp_obj_name ,self.handpoints = load_data_rt_mulitigrasp()
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

        label = self.get_grasp_label(self.input_labels[p_i], random=False, all_labels=self.all_labels).astype(np.float32)
        graspparts = self.input_features[p_i].astype(np.float32)[:, :16]

        tp_list += [points]
        tf_list += [graspparts]
        tl_list += [label]
        ti_list += [p_i]

        stacked_points = np.concatenate(tp_list, axis=0)
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        labels = np.array(tl_list, dtype=np.float32)/1.5708
        model_inds = np.array(ti_list, dtype=np.int32)
        stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)
        hand = self.handpoints[p_i] * 0.1
        name = self.grasp_obj_name[p_i]
        stacked_features = np.concatenate(tf_list, axis=0)

        input_list += [stacked_points, labels, stack_lengths, model_inds, hand, name,stacked_features]
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
        self.labels = torch.tensor(input_list[ind],dtype=torch.float32)
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
        self.labels = self.labels.pin_memory()
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
        self.labels = self.labels.to(device)
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

    path = '/home/hjl/hjl_transformer/dataset_hjl/Grasp Dataset/grasps/'
    #load_data_rt(path)

    train_dataloader = DataLoader(RTJlabel(), num_workers=8, batch_size=8,
                                  shuffle=True, drop_last=False, collate_fn=GraspNetCollate)
    for i, batch in enumerate(train_dataloader):#25个数qxqyqzqw,xyz,angle
        #print(batch[2])

        print(batch.features)
        print(batch.points)
        print(batch.labels.shape)
        print(batch.model_inds)
        print(batch.name)
        # print(1)
        # print(batch[3])
        # print(batch[4])
        # print(i)
        # filename = '/good_shapes/'+ eval(str(batch.name).split('(')[1].split(',')[0]) + '_scaled.obj.smoothed' + '.xml'
        # print(filename)
        #print(d) #('.xml'),
        #print(data[2].shape)

