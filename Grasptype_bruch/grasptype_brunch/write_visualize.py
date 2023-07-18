import open3d as o3d
import numpy as np
import torch

#from dataloader_grasptype import RTJlabel, GraspNetCollate
from data_xml_process_Baseline_withlabel import RTJlabel,GraspNetCollate
from torch.utils.data.dataloader import DataLoader
from scipy.spatial.transform import Rotation as R
'''
得到rt，得到pose。乘积得到手型，保存为pcd文件，与物体一起加载到open3d中，可视化结果
输入的是物体的名字，输出的手的pose的numpy形式
'''

def write_hand(points, save_pcd_path):
    n = len(points)
    lines = []
    for i in range(n):
        x, y, z = points[i]
        lines.append('{:.6f} {:.6f} {:.6f} {}'.format( \
            x, y, z, 0))
    with open(save_pcd_path, 'w') as f:
        HEADER = '''\
VERSION .7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 10
HEIGHT 1
POINTS 21
VIEWPOINT 0 0 0 1 0 0 0
DATA ascii
        '''
        f.write(HEADER.format(n, n))
        f.write('\n'.join(lines))

def write_pcd(obj, shadow, epoch,batch): #obj是str类型的名字，hand是（batchsize,21,3）大小的矩阵
    obj_names = '/home/hjl/hjl_transformer/dataset_hjl/labeled_data2/use/'\
                + str(obj) + '_scaled.obj.smoothed' + '.pcd'
    #obj_pcd = o3d.io.read_point_cloud(obj_names)
    #obj_pcd.paint_uniform_color([1, 0.706, 0]) #黄色
    paths = []
    #for i in range(shadow.shape[0]):

    path='./data/result/{}_{}.pcd'.format(epoch,batch)
    write_hand(shadow,path)
    paths.append(path)

    return obj_names, paths

def visualize_pcd(obj_names, shadow_pcd_paths):
    obj_pcd = o3d.io.read_point_cloud(obj_names)
    obj_pcd.paint_uniform_color([1, 0.706, 0]) #黄色

    for i in range(len(shadow_pcd_paths)):
        shadow_pcd = o3d.io.read_point_cloud(shadow_pcd_paths[i])
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 6], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [3, 12],
                 [12, 13], [13, 14], [4, 15], [15, 16],
                 [16, 17], [5, 18], [18, 19], [19, 20]]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_pcd = o3d.geometry.LineSet()
        line_pcd.lines = o3d.utility.Vector2iVector(lines)
        line_pcd.colors = o3d.utility.Vector3dVector(colors)
        line_pcd.points = o3d.utility.Vector3dVector(shadow_pcd.points)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='open3d')
        #
        vis.add_geometry(obj_pcd)
        # vis.add_geometry(label)
        vis.add_geometry(shadow_pcd)
        vis.add_geometry(line_pcd)
        vis.run()

if __name__ == '__main__':
    path = '/home/hjl/hjl_transformer/dataset_hjl/Grasp Dataset/grasps/'
    # load_data_rt(path)

    train_dataloader = DataLoader(RTJlabel(), num_workers=8, batch_size=1,
                                  shuffle=False, drop_last=False, collate_fn=GraspNetCollate)
    for i, batch in enumerate(train_dataloader):  # 25个数qxqyqzqw,xyz,angle
        rt_label = batch.labels[:,1,:7]  # 25 个数
        r0 = rt_label[:,:4] # qw qx qy qz
        r1 = torch.zeros((1,4)) # qx qy qz qw
        r1[:, 0] = r0[:, 1]
        r1[:, 1] = r0[:, 2]
        r1[:, 2] = r0[:, 3]
        r1[:, 3] = r0[:, 0]
        t = rt_label[:,4:7]*1000.0
        #r = np.squeeze(r, axis=(1,))
        #t = np.squeeze(t, axis=(1,))
        Rm = R.from_quat(r1)
        rotation_matrix = Rm.as_matrix()
        r_ = np.squeeze(rotation_matrix)
        z_ = np.zeros(3)
        z__ = np.ones(4)
        rs = np.insert(r_, 3, values=z_, axis=0)
        rss = np.insert(rs, 3, values=z__, axis=1)
        rss[:3,3] = t

        print(batch.name)
        j_p = np.load('shadow.npy')
        jp = j_p[:21, :]
        jp = jp.reshape(21, 3)
        trans1 = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        trans2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        trans3 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        trans = np.dot(trans1, trans2)
        a = np.matrix(trans)
        a = a.I
        a = np.array(a)
        # trans = np.array([[0,0,-1],[1,0,0],[0,-1,0]])

        jp1 = np.dot(jp, a)
        t = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        z = np.ones(21)
        jp2 = np.insert(jp1, 3, values=z, axis=1)
        jp3 = np.dot(t, jp2.T)

        jp4 = np.dot(rss,jp3)
        jp5 = jp4.T[:, :3]
        #jp5 = jp5.reshape(1, 21, 3)

        obj_names, paths = write_pcd(batch.name, jp5, 0, i)
        visualize_pcd(obj_names,paths)
