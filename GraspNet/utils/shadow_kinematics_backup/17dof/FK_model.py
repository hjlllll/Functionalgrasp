# -*- coding: UTF-8 -*-
import torch
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


class FK(object):
    def __init__(self):
        pass

    def transforms_multiply(self, t0s, t1s):
        return torch.matmul(t0s, t1s)

    def transforms_blank(self, rotations):
        """
        transforms : (F, J, 4, 4) ndarray
            Array of identity transforms for
            each frame F and joint J
        """
        diagonal = torch.eye(4)
        ts = diagonal.expand(int(rotations.shape[0]), int(rotations.shape[1]), 4, 4)
        return ts

    def transforms_rotations(self, rotations):
        """
        rotations : (F, J, 4) for each frame F and joint J
        """
        q_length = torch.sqrt(torch.sum(rotations.pow(2), dim=-1))   # [F,J,1]
        qw = rotations[..., 0] / q_length  # [F,J,1]
        qx = rotations[..., 1] / q_length  # [F,J,1]
        qy = rotations[..., 2] / q_length  # [F,J,1]
        qz = rotations[..., 3] / q_length  # [F,J,1]
        """Unit quaternion based rotation matrix computation""" """坐标系的定义和构建隐含在这部分了 """
        x2 = qx + qx  # [F,J,1]
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], -1)
        dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], -1)
        dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], -1)
        m = torch.stack([dim0, dim1, dim2], -2)

        return m   # [F,J,3,3]

    def transforms_local(self, positions, rotations):
        transforms = self.transforms_rotations(rotations)  # [F,J,3,3]
        transforms = torch.cat([transforms, positions[:, :, :, None]], -1)  # [F,J,3,4]
        zeros = torch.zeros([int(transforms.shape[0]), int(transforms.shape[1]), 1, 3])  # [F,J,1,3]
        ones = torch.ones([int(transforms.shape[0]), int(transforms.shape[1]), 1, 1])  # [F,J,1,1]
        zerosones = torch.cat([zeros, ones], -1)  # [F,J,1,4]
        transforms = torch.cat([transforms, zerosones], -2)  # [F,J,4,4]
        return transforms

    def transforms_global(self, parents, positions, rotations):
        locals = self.transforms_local(positions, rotations)  # 四元数转旋转矩阵 [F,J,4,4]
        globals = self.transforms_blank(rotations)  # [F,J,4,4]

        globals = torch.cat([locals[:, 0:1], globals[:, 1:]], 1)  # 因为0号关节已经在local的计算过程中得到了，所以不需要再进行计算
        globals = torch.split(globals, 1, 1)  # 因为torch.split输出是tuple型，后续无法迭代，所以需要变成list型
        globals = list(globals)
        for i in range(1, positions.shape[1]):  # 从1号而非0号开始, [1,22]
            globals[i] = self.transforms_multiply(globals[parents[i]][:, 0], locals[:, i])[:, None, :, :]  #这里实质就是通过右乘新矩阵得到本关节相对初始坐标系的变换关系，恰好4×4矩阵最右上角三个数就是本关节在初始坐标系的坐标

        return torch.cat(globals, 1)  # [F,J,4,4]

    def run(self, parents, positions, rotations):  # parents:[22,], positions:[F,J,3], rotations:[F,J,4]
        positions = self.transforms_global(parents, positions, rotations)[:, :, :, 3]  # [F,J,1,4] --> [F,J,4]
        return positions[:, :, :3]  # positions[:, :, :3] / positions[:, :, 3, None]   # [F,J,3]


class Shadowhand_FK(object):
    """
    self.run()为主函数
    输入为base的姿态 (F, 7): x y z qw qx qy qz
         rotations (F, J): J个关节的旋转角度，弧度制
    输出：各个关节的空间坐标（包括base的），与base相同参考系

    使用示例：
    from FK_model import Shadowhand_FK
    fk = Shadowhand_FK()
    j_p = fk.run(base, rotations)

    关节值定义为26个，不包括基座，即 J=26
    0号坐标是基座，直接由网络预测，不参与计算
    """
    def __init__(self):
        pass

    def transforms_multiply(self, t0s, t1s):
        return torch.matmul(t0s, t1s)

    def transforms_blank(self, shape0, shape1):
        """
        transforms : (F, J, 4, 4) ndarray
            Array of identity transforms for
            each frame F and joint J
        """
        diagonal = torch.eye(4).cuda()
        ts = diagonal.expand(shape0, shape1, 4, 4)
        return ts

    def transforms_base(self, base):
        """
        base：（F, 7）—— x y z qw qx qy qz
        """
        base = base.unsqueeze(-2)
        rotations = base[:, :, 3:]
        q_length = torch.sqrt(torch.sum(rotations.pow(2), dim=-1))  # [F,J,1]
        qw = rotations[..., 0] / q_length  # [F,J,1]
        qx = rotations[..., 1] / q_length  # [F,J,1]
        qy = rotations[..., 2] / q_length  # [F,J,1]
        qz = rotations[..., 3] / q_length  # [F,J,1]
        """Unit quaternion based rotation matrix computation"""
        x2 = qx + qx  # [F,J,1]
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], -1)
        dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], -1)
        dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], -1)
        R = torch.stack([dim0, dim1, dim2], -2).cuda()   # [F,1,3,3]

        T = base[..., :3].unsqueeze(-1).cuda()  # (F, 1, 3, 1)
        zeros = torch.zeros([int(base.shape[0]), 1, 1, 3]).cuda()  # (F, 1, 1, 3)
        ones = torch.ones([int(base.shape[0]), 1, 1, 1]).cuda()  # (F, 1, 1, 1)
        base_M = torch.cat([torch.cat([R, zeros], -2), torch.cat([T, ones], -2)], -1) # (F, 1, 4, 4)

        return base_M   # [F,1,4,4]

    def transforms_rotations(self, rotations):
        """
        角度输入暂定为弧度制，最好加上非线性处理，使角度值不超过限制
        rotations : (F, J) , Angle for each frame F and joint J
        M_r: 将角度转为绕z轴的旋转矩阵
        """
        m11 = torch.cos(rotations)  # (F, J)
        m12 = -torch.sin(rotations)  # (F, J)
        m21 = torch.sin(rotations)  # (F, J)
        m22 = torch.cos(rotations)  # (F, J)
        mr = torch.stack([torch.stack([m11, m21], -1), torch.stack([m12, m22], -1)], -1)  # (F, J, 2, 2)
        zeros = torch.zeros([int(rotations.shape[0]), int(rotations.shape[1]), 2, 2]).cuda()  # (F, J, 2, 2)
        eyes = torch.eye(2).expand(int(rotations.shape[0]), int(rotations.shape[1]), 2, 2).cuda()  # (F, J, 2, 2)
        M_r = torch.cat([torch.cat([mr, zeros], -2), torch.cat([zeros, eyes], -2)], -1)  # (F, J, 4, 4)

        return M_r   # [F,J,4,4]

    def transforms_local(self, M_sh, rotations):
        M_r = self.transforms_rotations(rotations).cuda()# [F,J,4,4]
        # print(M_r.shape)
        # print(M_sh.shape)
        M_sh = M_sh.expand(int(rotations.shape[0]), int(rotations.shape[1]), 4, 4).cuda()
        transforms = self.transforms_multiply(M_sh, M_r)  # [F,J,4,4]
        # print("transforms.shape:", transforms.shape)
        return transforms

    def transforms_global(self, base, parents, M_sh, rotations):
        locals = self.transforms_local(M_sh, rotations)  # 角度+预设生成旋转矩阵 [F,J,4,4]
        globals = self.transforms_blank(int(rotations.shape[0]), int(rotations.shape[1]))  # [F,J,4,4]
        base_M = self.transforms_base(base)   # [F,1,4,4]

        globals = torch.cat([base_M, globals], 1)  # 0号坐标是基座，直接由网络预测，不参与计算，但是需要给定值 # [F,J+1,4,4]
        globals = torch.split(globals, 1, 1)  # 因为torch.split输出是tuple型，后续无法迭代，所以需要变成list型
        globals = list(globals)  # list长度为J+1，每个元素[F, 1, 4, 4]
        # print(len(globals), locals.shape)

        # # all ass key joints
        # ass = self.transforms_blank(int(rotations.shape[0]), 60)  # 辅助点个数，5×3×4=60，总尺寸[F, 60, 4, 4]
        # ass = torch.split(ass, 1, 1)  # 因为torch.split输出是tuple型，后续无法迭代，所以需要变成list型
        # ass = list(ass)  # list长度为60，每个元素[F, 1, 4, 4]
        # index = [-1, -1, -1, -1, 0, 1, 2, -1, -1, 0, 1, 2, -1, -1, 0, 1, 2, -1, -1, 0,  1,  2, -1, -1,  3,  4,  5]   # 哪些关节需要构建辅助点，-1则不需要，非-1数字表示使用M_ass的序号，M_ass共6组，来自两种手指的三个link
        # i_ass = [-1, -1, -1, -1, 0, 1, 2, -1, -1, 3, 4, 5, -1, -1, 6, 7, 8, -1, -1, 9, 10, 11, -1, -1, 12, 13, 14]   # 哪些关节需要构建辅助点，-1则不需要，非-1数字表示关节序号
        # M_ass = np.load('./utils/M_ass.npy')  # [6,4,4,4]
        # M_ass = torch.from_numpy(M_ass).float().cuda()
        # for i in range(1, len(parents)):  # 从1号而非0号开始, [1,26]
        #     globals[i] = self.transforms_multiply(globals[parents[i]][:, 0], locals[:, i-1])[:, None, :, :]  # 这里实质就是通过右乘新矩阵得到本关节相对初始坐标系的变换关系，恰好4×4矩阵最右上角三个数就是本关节在初始坐标系的坐标
        #     if index[i] != -1:
        #         ass[4*i_ass[i]] = self.transforms_multiply(globals[parents[i]][:, 0], M_ass[index[i], 0])[:, None, :, :]    # o
        #         ass[4*i_ass[i]+1] = self.transforms_multiply(ass[4*i_ass[i]][:, 0], M_ass[index[i], 1])[:, None, :, :]  # up
        #         ass[4*i_ass[i]+2] = self.transforms_multiply(ass[4*i_ass[i]][:, 0], M_ass[index[i], 2])[:, None, :, :]  # left
        #         ass[4*i_ass[i]+3] = self.transforms_multiply(ass[4*i_ass[i]][:, 0], M_ass[index[i], 3])[:, None, :, :]  # right
        #
        # globals = torch.cat(globals, 1)  # [F,J,4,4]
        # ass = torch.cat(ass, 1)  # [F,60,4,4]
        # globals_ass = torch.cat([globals, ass], 1)  # [F,J+60,4,4]

        # chose ass key joints
        ass = self.transforms_blank(int(rotations.shape[0]), 10)  # 辅助点个数，5×1×2=10，总尺寸[F, 10, 4, 4]
        ass = torch.split(ass, 1, 1)  # 因为torch.split输出是tuple型，后续无法迭代，所以需要变成list型
        ass = list(ass)  # list长度为60，每个元素[F, 1, 4, 4]
        index = [-1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, 2, -1, -1, -1, -1, 2, -1, -1, -1,  -1,  2, -1, -1,  -1,  -1,  5]   # 哪些关节需要构建辅助点，-1则不需要，非-1数字表示使用M_ass的序号，M_ass共6组，来自两种手指的三个link
        i_ass = [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 1, -1, -1, -1, -1, 2, -1, -1, -1, -1, 3, -1, -1, -1, -1, 4]   # 哪些关节需要构建辅助点，-1则不需要，非-1数字用于计算在ass中的编号
        M_ass = np.load('./utils/M_ass.npy')  # [6,4,4,4] 加载辅助点信息
        M_ass = torch.from_numpy(M_ass).float().cuda()
        for i in range(1, len(parents)):  # 从1号而非0号开始, [1,26]
            globals[i] = self.transforms_multiply(globals[parents[i]][:, 0], locals[:, i-1])[:, None, :, :]  # 这里实质就是通过右乘新矩阵得到本关节相对初始坐标系的变换关系，恰好4×4矩阵最右上角三个数就是本关节在初始坐标系的坐标
            if index[i] != -1:
                ass[2*i_ass[i]] = self.transforms_multiply(globals[parents[i]][:, 0], M_ass[index[i], 0])[:, None, :, :]    # o
                ass[2*i_ass[i]+1] = self.transforms_multiply(ass[2*i_ass[i]][:, 0], M_ass[index[i], 1])[:, None, :, :]  # up
                # ass[4*i_ass[i]+2] = self.transforms_multiply(globals[parents[i]][:, 0], M_ass[index[i], 2])[:, None, :, :]  # left
                # ass[4*i_ass[i]+3] = self.transforms_multiply(globals[parents[i]][:, 0], M_ass[index[i], 3])[:, None, :, :]  # right

        globals = torch.cat(globals, 1)  # [F,J,4,4]
        ass = torch.cat(ass, 1)  # [F,10,4,4]
        globals_ass = torch.cat([globals, ass], 1)  # [F,J+10,4,4]

        return globals_ass   # [F,J+60,4,4]

    def run(self, base, rotations):  # parents:[22,], positions:[F,J,3], rotations:[F,J,4]
        parents = [-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0, 17, 18, 19, 20, 0, 22, 23, 24, 25]   # 依据graspit!中shadowhand_simple顺序,图见OneNote-学习笔记-机器人学-坐标变换-shadowhand
        M_sh = np.load('./utils/M_shadowhand.npy')  # 加载shadowhand的运动学关系
        M_sh = torch.from_numpy(M_sh).float()
        # print(M_sh.shape)
        positions = self.transforms_global(base, parents, M_sh, rotations)[:, :, :, 3]  # [F,J,1,4] --> [F,J,4]
        return positions[:, :, :3]  # positions[:, :, :3] / positions[:, :, 3, None]   # [F,J,3]


def fk_run(outputs_r, outputs_t, outputs_a):
    # # 输入正向运动学层
    # 3 + 4
    outputs_base = torch.cat((outputs_t / 5.0 * 1000, outputs_r), 1)
    # 17(18) -> 26
    outputs_rotation = torch.zeros([outputs_a.shape[0], 26]).type_as(outputs_a)  # .cuda()
    outputs_rotation[:, 0:4] = outputs_a[:, 0:4]
    outputs_rotation[:, 4] = 0.8 * outputs_a[:, 3]
    outputs_rotation[:, 6:9] = outputs_a[:, 4:7]
    outputs_rotation[:, 9] = 0.8 * outputs_a[:, 6]
    outputs_rotation[:, 11:14] = outputs_a[:, 7:10]
    outputs_rotation[:, 14] = 0.8 * outputs_a[:, 9]
    outputs_rotation[:, 16:19] = outputs_a[:, 10:13]
    outputs_rotation[:, 19] = 0.8 * outputs_a[:, 12]
    # outputs_rotation[:, 21:25] = outputs_a[:, [13,14,16,17]]  # if output_a.shape[0] == 18
    outputs_rotation[:, 21:25] = outputs_a[:, 13:]  # if output_a.shape[0] == 17
    fk = Shadowhand_FK()
    outputs_FK = fk.run(outputs_base, outputs_rotation * 1.5708)  # [F, J+1+A, 3]  #原始J+1个关键点，再加上A个辅助点

    return outputs_FK

def show_data(points, graspparts, j_pp=None):
    print(points.shape)
    points, graspparts = points.clone().detach().cpu().numpy(), graspparts.clone().detach().cpu().numpy()
    colpart = []
    for i in range(graspparts.shape[0]):
        hh = 0
        for cc in colpart:
            if not (graspparts[i]==cc).all():
                hh += 1
        if hh==len(colpart):
            colpart.append(graspparts[i])
    col = np.random.random([len(colpart), 3])
    print('len(col):', len(colpart))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.scatter([100, 0, 0], [0, 100, 0], [0, 0, 100], c=np.array([[1, 0, 0]]), marker='o', linewidths=1)
    for i in range(points.shape[0]):
        for j, cc in enumerate(colpart):
            if (graspparts[i]==cc).all():
                col_idx = j
        ax1.scatter(points[i, 0], points[i, 1], points[i, 2], c=col[col_idx].reshape(1,-1), marker='.')  # [0.5, 0.5, 0.5]

    if j_pp is not None:
        print('j_pp.shape', j_pp.shape)
        j_pp =j_pp.clone().detach().cpu().numpy()
        Joints = j_pp[:27]
        ax1.scatter(Joints[:,0], Joints[:,1], Joints[:,2], c=np.array([[0, 0, 0]]), marker='x', linewidths=2)
        lines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [10, 11], [0, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                 [0, 17], [17, 18], [18, 19], [19, 20], [20, 21], [0, 22], [22, 23], [23, 24], [24, 25], [25, 26]]
        for line in lines:
            x = [Joints[line[0]][0], Joints[line[1]][0]]
            y = [Joints[line[0]][1], Joints[line[1]][1]]
            z = [Joints[line[0]][2], Joints[line[1]][2]]
            ax1.plot(x, y, z,  color='r', linewidth=2)
        #
        # Ass = j_pp[27:]
        # ax1.scatter(Ass[:,0], Ass[:,1], Ass[:,2], c=np.array([[0, 1, 0]]), marker='x', linewidths=2)
        # for i in range(10):
        #     for j in range(1,4):
        #         x = [Ass[4*i,0], Ass[4*i+j,0]]
        #         y = [Ass[4*i,1], Ass[4*i+j,1]]
        #         z = [Ass[4*i,2], Ass[4*i+j,2]]
        #         ax1.plot(x, y, z, color='b', linewidth=2)


    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend()
    plt.axis('off')
    plt.show()

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import sys
import os
from multiprocessing import Process

def run(name, xml_name):
    os.environ['GRASPIT'] = '/home/lm/graspit'
    os.system('/home/lm/graspit/build/graspit_simulator --world {} >/dev/null 2>&1'.format(xml_name))

def show_data_fast(points, graspparts, j_pp, pi=None, show_jp=False):
    w = gl.GLViewWidget()
    w.opts['distance'] = 500
    w.show()
    w.setWindowTitle('show Grasping by pyqtgraph')

    # g = gl.GLGridItem()
    # w.addItem(g)

    points, graspparts = points.clone().detach().cpu().numpy(), graspparts.clone().detach().cpu().numpy()
    print('points.shape', points.shape, 'graspparts.shape', graspparts.shape)
    colpart = np.unique(graspparts, axis=0)
    col = [[1, 1, 1, 0.5],  # 白色
           [0, 1, 0, 1],  # 绿色
           [1, 0.1, 0.1, 1],  # 砖红
           [0, 1, 1, 1],  # 蓝绿色
           [1, 0, 1, 1],  # 紫红，洋红
           [0.8, 0.4, 0.1, 1],  # 亮橙
           [1, 0.6, 0, 1],  # 橙色
           [1, 1, 0, 1],  # 黄色
           [1, 0.9, 0.6, 1],  #小麦色
           [1, 0.4, 0.7, 1]]  # 活力粉

    print('colpart:\n', colpart)
    print('colpart.shape:', colpart.shape)

    points_list = []
    for j in range(colpart.shape[0]):
        ppp = []
        for i in range(points.shape[0]):
            if (graspparts[i]==colpart[j]).all():
                ppp.append(points[i])
        ppp = np.concatenate(ppp, 0).reshape(-1, 3)
        points_list.append(ppp)

    p_sum=0
    for i, ppp in enumerate(points_list):
        sp1 = gl.GLScatterPlotItem(pos=ppp, size=0.2 * (3*i + 1), color=col[i], pxMode=False)
        w.addItem(sp1)
        p_sum += ppp.shape[0]
        print(i, ppp.shape)
    print('p_sum', p_sum)
    assert p_sum==points.shape[0]

    if show_jp:
        print('j_pp.shape', j_pp.shape)
        j_pp =j_pp.clone().detach().cpu().numpy()
        Joints = j_pp[:27]
        sp1 = gl.GLScatterPlotItem(pos=Joints, size=2.5, color=(1,0,0,1), pxMode=False)
        w.addItem(sp1)
        lines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [10, 11], [0, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                 [0, 17], [17, 18], [18, 19], [19, 20], [20, 21], [0, 22], [22, 23], [23, 24], [24, 25], [25, 26]]
        lines_col = [(0,1,0,1), (0.1,0.2,0.4,1), (0.1,0.9,0.4,1), (0.6,0.2,0.7,1), (0.2,0.5,0.6,1), (0.7,0.1,0,1),
                     (0,1,0,1),                  (0.1,0.9,0.4,1), (0.6,0.2,0.7,1), (0.2,0.5,0.6,1), (0.7,0.1,0,1),
                     (0,1,0,1),                  (0.1,0.9,0.4,1), (0.6,0.2,0.7,1), (0.2,0.5,0.6,1), (0.7,0.1,0,1),
                     (0,1,0,1),                  (0.1,0.9,0.4,1), (0.6,0.2,0.7,1), (0.2,0.5,0.6,1), (0.7,0.1,0,1),
                     (0,1,0,1),                  (0.1,0.9,0.4,1), (0.6,0.2,0.7,1), (0.2,0.5,0.6,1), (0.7,0.1,0,1)]
        for ii, line in enumerate(lines):
            plt = gl.GLLinePlotItem(pos=Joints[line], color=lines_col[ii], width=2, antialias=True)
            w.addItem(plt)

    if pi is not None:
        p1 = Process(target=run, args=('anne', pi))  # 必须加,号
        p1.start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

def show_data_fast_subfigure(points, graspparts, j_pp, mw, pi=None, show_jp=True):
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

    points, graspparts = points.clone().detach().cpu().numpy(), graspparts.clone().detach().cpu().numpy()
    print('points.shape', points.shape, 'graspparts.shape', graspparts.shape)
    colpart = np.unique(graspparts, axis=0)
    col = [[1, 1, 1, 0.5],  # 白色
           [0, 1, 0, 1],  # 绿色
           [1, 0.1, 0.1, 1],  # 砖红
           [0, 1, 1, 1],  # 蓝绿色
           [1, 0, 1, 1],  # 紫红，洋红
           [0.8, 0.4, 0.1, 1],  # 亮橙
           [1, 0.6, 0, 1],  # 橙色
           [1, 1, 0, 1],  # 黄色
           [1, 0.9, 0.6, 1],  #小麦色
           [1, 0.4, 0.7, 1]]  # 活力粉

    print('colpart:\n', colpart)
    print('colpart.shape:', colpart.shape)

    points_list = []
    for j in range(colpart.shape[0]):
        ppp = []
        for i in range(points.shape[0]):
            if (graspparts[i]==colpart[j]).all():
                ppp.append(points[i])
        ppp = np.concatenate(ppp, 0).reshape(-1, 3)
        points_list.append(ppp)

    p_sum=0
    for i, ppp in enumerate(points_list):
        sp1 = gl.GLScatterPlotItem(pos=ppp, size=0.2 * (3*i + 1), color=col[i], pxMode=False)
        w.addItem(sp1)
        w2.addItem(sp1)
        p_sum += ppp.shape[0]
        print(i, ppp.shape)
    print('p_sum', p_sum)
    assert p_sum==points.shape[0]

    if show_jp:
        print('j_pp.shape', j_pp.shape)
        j_pp =j_pp.clone().detach().cpu().numpy()
        Joints = j_pp[:27]
        sp1 = gl.GLScatterPlotItem(pos=Joints, size=2.5, color=(1,0,0,1), pxMode=False)
        w2.addItem(sp1)
        lines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [10, 11], [0, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                 [0, 17], [17, 18], [18, 19], [19, 20], [20, 21], [0, 22], [22, 23], [23, 24], [24, 25], [25, 26]]
        lines_col = [(0,1,0,1), (0.1,0.2,0.4,1), (0.1,0.9,0.4,1), (0.6,0.2,0.7,1), (0.2,0.5,0.6,1), (0.7,0.1,0,1),
                     (0,1,0,1),                  (0.1,0.9,0.4,1), (0.6,0.2,0.7,1), (0.2,0.5,0.6,1), (0.7,0.1,0,1),
                     (0,1,0,1),                  (0.1,0.9,0.4,1), (0.6,0.2,0.7,1), (0.2,0.5,0.6,1), (0.7,0.1,0,1),
                     (0,1,0,1),                  (0.1,0.9,0.4,1), (0.6,0.2,0.7,1), (0.2,0.5,0.6,1), (0.7,0.1,0,1),
                     (0,1,0,1),                  (0.1,0.9,0.4,1), (0.6,0.2,0.7,1), (0.2,0.5,0.6,1), (0.7,0.1,0,1)]
        for ii, line in enumerate(lines):
            plt = gl.GLLinePlotItem(pos=Joints[line], color=lines_col[ii], width=2, antialias=True)
            w2.addItem(plt)

    if pi is not None:
        p1 = Process(target=run, args=('anne', pi))  # 必须加,号
        p1.start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
