import numpy as np
import math
# import csv
from mayavi import mlab
# import cv2
from mano.mano_m.utils import Mesh
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d


class Map_Loader():
    def __init__(self):
        # load data

        #self.label = np.load('taxonomy.npy')
        self.label = np.load('hand_keypoint.npy')
        self.labels = np.zeros(((33,21,3)))
        self.labels[:,0,:] = self.label[:,0,:]
        self.labels[:,1,:] = self.label[:,13,:]
        self.labels[:,2,:] = self.label[:,1,:]
        self.labels[:,3,:] = self.label[:,4,:]
        self.labels[:,4,:] = self.label[:,10,:]
        self.labels[:,5,:] = self.label[:,7,:]

        self.labels[:,6,:] = self.label[:,14,:]
        self.labels[:,7,:] = self.label[:,15,:]
        self.labels[:,8,:] = self.label[:,16,:]

        self.labels[:,9,:] = self.label[:,2,:]
        self.labels[:,10,:] = self.label[:,3,:]
        self.labels[:,11,:] = self.label[:,17,:]

        self.labels[:,12,:] = self.label[:,5,:]
        self.labels[:,13,:] = self.label[:,6,:]
        self.labels[:,14,:] = self.label[:,18,:]

        self.labels[:,15,:] = self.label[:,11,:]
        self.labels[:,16,:] = self.label[:,12,:]
        self.labels[:,17,:] = self.label[:,19,:]
        self.labels[:, 18, :] = self.label[:, 8, :]
        self.labels[:, 19, :] = self.label[:, 9, :]
        self.labels[:, 20, :] = self.label[:, 20, :]

        self.shadow = self.new_tams_shadow_model()

    def map(self, start):
        rh_palm, rh_middle_pip, rh_dip_middle, rh_tip_middle, rh_tf_pip_wrist = self.shadow
        # the joint order is
        # [Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP,
        # TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP]
        # for index in range(start, start + batch_size):
        frame = self.label[start,0,:]
        keypoints  = self.label[start,:,:]#*1000.0
        keypoints = keypoints.reshape(21, 3)

        tf_palm = keypoints[1] - keypoints[0]
        ff_palm = keypoints[5] - keypoints[0]
        mf_palm = keypoints[9] - keypoints[0]
        rf_palm = keypoints[13] - keypoints[0]
        lf_palm = keypoints[17] - keypoints[0]
        palm = np.array([tf_palm, ff_palm, mf_palm, rf_palm, lf_palm])

        # local wrist frame build
        wrist_z = np.mean(palm[2:4], axis=0)#计算每一列的均值 access middle finger and ring finger
        wrist_z /= np.linalg.norm(wrist_z)#范数，默认为矩阵二范数
        wrist_y = np.cross(rf_palm, mf_palm)#叉乘 返回垂直于1，2的向量
        wrist_y /= np.linalg.norm(wrist_y)
        wrist_x = np.cross(wrist_y, wrist_z)
        if np.linalg.norm(wrist_x) != 0:
            wrist_x /= np.linalg.norm(wrist_x)

        local_frame = np.vstack([wrist_x, wrist_y, wrist_z])
        local_points = np.dot((keypoints - keypoints[0]), local_frame.T) #矩阵相乘

        local_palm = np.array([local_points[1], local_points[5], local_points[9], local_points[13], local_points[17]])
        hh_palm = np.linalg.norm(local_palm, axis=1) #length

        tf_pip_mcp = local_points[2] - local_points[1]
        tf_dip_pip = local_points[3] - local_points[2]
        tf_tip_pip = local_points[4] - local_points[3]

        ff_pip_mcp = local_points[6] - local_points[5]
        ff_dip_pip = local_points[7] - local_points[6]
        ff_tip_pip = local_points[8] - local_points[7]

        mf_pip_mcp = local_points[10] - local_points[9]
        mf_dip_pip = local_points[11] - local_points[10]
        mf_tip_pip = local_points[12] - local_points[11]

        rf_pip_mcp = local_points[14] - local_points[13]
        rf_dip_pip = local_points[15] - local_points[14]
        rf_tip_pip = local_points[16] - local_points[15]

        lf_pip_mcp = local_points[18] - local_points[17]
        lf_dip_pip = local_points[19] - local_points[18]
        lf_tip_pip = local_points[20] - local_points[19]

        #hand
        pip_mcp = np.array([tf_pip_mcp, ff_pip_mcp, mf_pip_mcp, rf_pip_mcp, lf_pip_mcp])
        tip_pip = np.array([tf_tip_pip, ff_tip_pip, mf_tip_pip, rf_tip_pip, lf_tip_pip])
        dip_pip = np.array([tf_dip_pip, ff_dip_pip, mf_dip_pip, rf_dip_pip, lf_dip_pip])
        hh_pip_mcp = np.linalg.norm(pip_mcp, axis=1)
        hh_tip_pip = np.linalg.norm(tip_pip, axis=1)
        hh_dip_pip = np.linalg.norm(dip_pip, axis=1)

        # hh_len = hh_palm + hh_pip_mcp + hh_dip_pip + hh_tip_dip
        # rh_palm, rh_middle_pip, rh_tip_middle, rh_tf_pip_wrist
        hh_pip_wrist = np.linalg.norm(local_points[2])
        th_pip_key = hh_pip_wrist / rh_tf_pip_wrist * local_points[2]

        # 换算长度 换算为shadowhand的关节长度
        coe_palm = rh_palm / hh_palm
        #rh_wrist_mcp_key = np.multiply(coe_palm.reshape(-1, 1), local_palm)
        # rh_wrist_mcp_key[0][2] = rh_wrist_mcp_key[0][2] + 29
        #rh_wrist_mcp_key[0] = [0, 0, 0]
        rh_wrist_mcp_key = np.array([[34,0,29],
                                     [33,0,95],
                                     [11,0,99],
                                     [-11,0,95],
                                     [-33,0,86.6]])


        coe_pip_mcp = rh_middle_pip / hh_pip_mcp
        rh_pip_mcp_key = np.multiply(coe_pip_mcp.reshape(-1, 1), pip_mcp) + rh_wrist_mcp_key
        #rh_pip_mcp_key[0] = th_pip_key

        coe_dip_pip = rh_dip_middle / hh_dip_pip
        rh_dip_pip_key = np.multiply(coe_dip_pip.reshape(-1, 1), dip_pip) + rh_pip_mcp_key

        coe_tip_pip = rh_tip_middle / hh_tip_pip
        rh_tip_pip_key = np.multiply(coe_tip_pip.reshape(-1, 1), tip_pip) + rh_dip_pip_key
        # rh_tip_pip_key[0] = local_points[8]



        shadow_points = np.vstack([np.array([0, 0, 0]), rh_wrist_mcp_key, # rh_wrist_mcp_key——5*3
                                    rh_pip_mcp_key[0],  rh_dip_pip_key[0], rh_tip_pip_key[0],
                                    rh_pip_mcp_key[1],  rh_dip_pip_key[1], rh_tip_pip_key[1],
                                    rh_pip_mcp_key[2],  rh_dip_pip_key[2], rh_tip_pip_key[2],
                                    rh_pip_mcp_key[3],  rh_dip_pip_key[3], rh_tip_pip_key[3],
                                    rh_pip_mcp_key[4],  rh_dip_pip_key[4], rh_tip_pip_key[4]]) #按垂直方向（行顺序）堆叠数组构成一个新的数组

        # tip_keys = rh_tip_pip_key/1000
        # pip_keys = rh_pip_mcp_key/1000
        # mcp_keys = rh_wrist_mcp_key/1000
        # dip_keys = rh_dip_pip_key/1000
        # from IPython import embed;embed()
        return rh_tip_pip_key/1000, rh_pip_mcp_key/1000, pip_mcp/1000, tip_pip/1000, frame, local_points, shadow_points

    def new_tams_shadow_model(self):
        # shadow hand length
        rh_tf_palm = math.sqrt(math.pow(34, 2) + math.pow(29, 2))
        rh_ff_palm = math.sqrt(math.pow(95, 2) + math.pow(33, 2))
        rh_mf_palm = math.sqrt(math.pow(99, 2) + math.pow(11, 2))
        rh_rf_palm = math.sqrt(math.pow(95, 2) + math.pow(11, 2))
        rh_lf_palm = math.sqrt(math.pow(86.6, 2) + math.pow(33, 2))
        rh_palm = np.array([rh_tf_palm, rh_ff_palm, rh_mf_palm, rh_rf_palm, rh_lf_palm])

        rh_tf_pip_wrist = math.sqrt(math.pow(34, 2) + math.pow(29, 2) + math.pow(38, 2)) # pow(底数，指数) sqrt平方根

        rh_tf_middle_pip = 38
        rh_tf_tip_middle = 27.5#20 + math.sqrt(math.pow(32, 2) + math.pow(4, 2))
        rh_tf_dip_middle = 32

        rh_ff_middle_pip = 45
        rh_ff_tip_middle = 26#20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))
        rh_ff_dip_middle = 25

        rh_mf_middle_pip = 45
        rh_mf_tip_middle = 26#20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))
        rh_mf_dip_middle = 25

        rh_rf_middle_pip = 45
        rh_rf_tip_middle = 26#20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))
        rh_rf_dip_middle = 25

        rh_lf_middle_pip = 45
        rh_lf_tip_middle = 26#20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))
        rh_lf_dip_middle = 25

        rh_middle_pip = np.array(
            [rh_tf_middle_pip, rh_ff_middle_pip, rh_mf_middle_pip, rh_rf_middle_pip, rh_lf_middle_pip])
        rh_tip_middle = np.array(
            [rh_tf_tip_middle, rh_ff_tip_middle, rh_mf_tip_middle, rh_rf_tip_middle, rh_lf_tip_middle])
        rh_dip_middle = np.array(
            [rh_tf_dip_middle, rh_ff_dip_middle, rh_mf_dip_middle, rh_rf_dip_middle, rh_lf_dip_middle])
        # rh_len = rh_palm + rh_middle_pip + rh_tip_middle
        return [rh_palm, rh_middle_pip, rh_dip_middle, rh_tip_middle, rh_tf_pip_wrist]

    def new_tams_shadow_model_bake(self):
        # shadow hand length
        rh_tf_palm = 34
        rh_ff_palm = math.sqrt(math.pow(95 - 29, 2) + math.pow(33, 2))
        rh_mf_palm = math.sqrt(math.pow(99 - 29, 2) + math.pow(11, 2))
        rh_rf_palm = math.sqrt(math.pow(95 - 29, 2) + math.pow(11, 2))
        rh_lf_palm = math.sqrt(math.pow(86.6 - 29, 2) + math.pow(33, 2))
        rh_palm = np.array([rh_tf_palm, rh_ff_palm, rh_mf_palm, rh_rf_palm, rh_lf_palm])

        rh_tf_pip_wrist = math.sqrt(math.pow(34, 2) + math.pow(38, 2)) # pow(底数，指数) sqrt平方根

        rh_tf_middle_pip = 38
        rh_tf_tip_middle = 20 + math.sqrt(math.pow(32, 2) + math.pow(4, 2))

        rh_ff_middle_pip = 45
        rh_ff_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_mf_middle_pip = 45
        rh_mf_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_rf_middle_pip = 45
        rh_rf_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_lf_middle_pip = 45
        rh_lf_tip_middle = 20 + math.sqrt(math.pow(29, 2) + math.pow(4, 2))

        rh_middle_pip = np.array(
            [rh_tf_middle_pip, rh_ff_middle_pip, rh_mf_middle_pip, rh_rf_middle_pip, rh_lf_middle_pip])
        rh_tip_middle = np.array(
            [rh_tf_tip_middle, rh_ff_tip_middle, rh_mf_tip_middle, rh_rf_tip_middle, rh_lf_tip_middle])

        # rh_len = rh_palm + rh_middle_pip + rh_tip_middle
        return [rh_palm, rh_middle_pip, rh_tip_middle, rh_tf_pip_wrist]

def show_line(un1, un2, color='g', scale_factor=1):
    # for shadow and human scale_factor=1
    if color == 'b':
        color_f = (0.8, 0, 0.9)
    elif color == 'r':
        color_f = (0.3, 0.2,0.7)
    elif color == 'p':
        color_f = (0.1, 1, 0.8)
    elif color == 'y':
        color_f = (0.5, 1, 1)
    elif color == 'g':
        color_f = (1, 1, 0)
    elif isinstance(color, tuple):
        color_f = color
    else:
        color_f = (1, 1, 1)
    mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)

def show_points(point, color='b', scale_factor=5):
    # for shadow and human scale_factor=5
    if color == 'b':
        color_f = (0, 0, 1)
    elif color == 'r':
        color_f = (1, 0, 0)
    elif color == 'g':
        color_f = (0, 1, 0)
    else:
        color_f = (1, 1, 1)
    if point.size == 3:  # vis for only one point
        mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
    else:  # vis for multiple points
        mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)

def show_hand(points, type='human'):
    show_points(points)
    if type == "human":
        show_line(points[0], points[1], color='r')
        show_line(points[1], points[6], color='r')
        show_line(points[7], points[6], color='r')
        show_line(points[7], points[8], color='r')

        show_line(points[0], points[2], color='y')
        show_line(points[9], points[2], color='y')
        show_line(points[9], points[10], color='y')
        show_line(points[11], points[10], color='y')

        show_line(points[0], points[3], color='g')
        show_line(points[12], points[3], color='g')
        show_line(points[12], points[13], color='g')
        show_line(points[14], points[13], color='g')

        show_line(points[0], points[4], color='b')
        show_line(points[15], points[4], color='b')
        show_line(points[15], points[16], color='b')
        show_line(points[17], points[16], color='b')

        show_line(points[0], points[5], color='p')
        show_line(points[18], points[5], color='p')
        show_line(points[18], points[19], color='p')
        show_line(points[20], points[19], color='p')
    elif type == "shadow":
        show_line(points[0], points[1], color='r')
        show_line(points[1], points[6], color='r')
        show_line(points[7], points[6], color='r')
        show_line(points[7], points[8], color='r')

        show_line(points[0], points[2], color='y')
        show_line(points[9], points[2], color='y')
        show_line(points[9], points[10], color='y')
        show_line(points[11], points[10], color='y')

        show_line(points[0], points[3], color='g')
        show_line(points[12], points[3], color='g')
        show_line(points[12], points[13], color='g')
        show_line(points[14], points[13], color='g')

        show_line(points[0], points[4], color='b')
        show_line(points[15], points[4], color='b')
        show_line(points[15], points[16], color='b')
        show_line(points[17], points[16], color='b')

        show_line(points[0], points[5], color='p')
        show_line(points[18], points[5], color='p')
        show_line(points[18], points[19], color='p')
        show_line(points[20], points[19], color='p')
    else:
     show_line(points[0], points[11], color='r')
     show_line(points[11], points[6], color='r')
     show_line(points[6], points[1], color='r')

     show_line(points[0], points[12], color='y')
     show_line(points[12], points[7], color='y')
     show_line(points[7], points[2], color='y')

     show_line(points[0], points[13], color='g')
     show_line(points[13], points[8], color='g')
     show_line(points[8], points[3], color='g')

     show_line(points[0], points[14], color='b')
     show_line(points[14], points[9], color='b')
     show_line(points[9], points[4], color='b')

     show_line(points[0], points[15], color='p')
     show_line(points[15], points[10], color='p')
     show_line(points[10], points[5], color='p')

    tf_palm = points[1] - points[0]
    ff_palm = points[2] - points[0]
    mf_palm = points[3] - points[0]
    rf_palm = points[4] - points[0]
    lf_palm = points[5] - points[0]
    # palm = np.array([tf_palm, ff_palm, mf_palm, rf_palm, lf_palm])
    palm = np.array([ff_palm, mf_palm, rf_palm, lf_palm])

    # local wrist frame build
    wrist_z = np.mean(palm, axis=0)
    wrist_z /= np.linalg.norm(wrist_z)
    wrist_y = np.cross(lf_palm, rf_palm)
    wrist_y /= np.linalg.norm(wrist_y)
    wrist_x = np.cross(wrist_y, wrist_z)
    if np.linalg.norm(wrist_x) != 0:
        wrist_x /= np.linalg.norm(wrist_x)

    mlab.quiver3d(points[0][0], points[0][1], points[0][2], wrist_x[0], wrist_x[1], wrist_x[2],
                  scale_factor=50, line_width=0.5, color=(1, 0, 0), mode='arrow')
    mlab.quiver3d(points[0][0], points[0][1], points[0][2], wrist_y[0], wrist_y[1], wrist_y[2],
                  scale_factor=50, line_width=0.5, color=(0, 1, 0), mode='arrow')
    mlab.quiver3d(points[0][0], points[0][1], points[0][2], wrist_z[0], wrist_z[1], wrist_z[2],
                  scale_factor=50, line_width=0.5, color=(0, 0, 1), mode='arrow')

def show_data(joints):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    #ax1.scatter([100, 0, 0], [0, 100, 0], [0, 0, 100], c=np.array([[1, 0, 0]]), marker='o', linewidths=1)

    ax1.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=np.array([[0, 0, 0]]), marker='x', linewidths=2)
    # Ass = j_pp.squeeze(0)[27:]
    # ax1.scatter(Ass[:, 0], Ass[:, 1], Ass[:, 2], c=np.array([[0, 1, 0]]), marker='x', linewidths=2)
    # [2,3][7,8][12,13][17,18][22,23]
    lines = [[0, 1], [1, 6], [6, 7], [7, 8],
             [0, 2], [2, 9], [9, 10], [10, 11],
             [0, 3], [3, 12], [12, 13], [13, 14],
             [0, 4], [4, 15], [15, 16], [16, 17],
             [0, 5], [5, 18], [18, 19], [19, 20]]
    lines = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]
    for line in lines:
        x = [joints[line[0]][0], joints[line[1]][0]]
        y = [joints[line[0]][1], joints[line[1]][1]]
        z = [joints[line[0]][2], joints[line[1]][2]]
        ax1.plot(x, y, z, color='r', linewidth=2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend()
    plt.axis('off')
    plt.show()

def show_data2(joints):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    #ax1.scatter([100, 0, 0], [0, 100, 0], [0, 0, 100], c=np.array([[1, 0, 0]]), marker='o', linewidths=1)

    ax1.scatter(joints[:21, 0], joints[:21, 1], joints[:21, 2], c=np.array([[1, 0, 0]]), marker='o', linewidths=0.2)
    ax1.scatter(joints[21:42, 0], joints[21:42, 1], joints[21:42, 2], c=np.array([[0, 1, 0]]), marker='o', linewidths=0.2)
    ax1.scatter(joints[42:, 0], joints[42:, 1], joints[42:, 2], c=np.array([[0, 0, 1]]), marker='o', linewidths=0.2)
    # Ass = j_pp.squeeze(0)[27:]
    # ax1.scatter(Ass[:, 0], Ass[:, 1], Ass[:, 2], c=np.array([[0, 1, 0]]), marker='x', linewidths=2)
    # [2,3][7,8][12,13][17,18][22,23]
    # lines = [[0, 1], [1, 6], [6, 7], [7, 8],
    #          [0, 2], [2, 9], [9, 10], [10, 11],
    #          [0, 3], [3, 12], [12, 13], [13, 14],
    #          [0, 4], [4, 15], [15, 16], [16, 17],
    #          [0, 5], [5, 18], [18, 19], [19, 20],
    # [0 + 21, 1 + 21], [1 + 21, 6 + 21], [6 + 21, 7 + 21], [7 + 21, 8 + 21],
    # [0 + 21, 2 + 21], [2 + 21, 9 + 21], [9 + 21, 10 + 21], [10 + 21, 11 + 21],
    # [0 + 21, 3 + 21], [3 + 21, 12 + 21], [12 + 21, 13 + 21], [13 + 21, 14 + 21],
    # [0 + 21, 4 + 21], [4 + 21, 15 + 21], [15 + 21, 16 + 21], [16 + 21, 17 + 21],
    # [0 + 21, 5 + 21], [5 + 21, 18 + 21], [18 + 21, 19 + 21], [19 + 21, 20 + 21],
    lines = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20],
             [0 + 21, 1 + 21], [1 + 21, 6 + 21], [6 + 21, 7 + 21], [7 + 21, 8 + 21],
             [0 + 21, 2 + 21], [2 + 21, 9 + 21], [9 + 21, 10 + 21], [10 + 21, 11 + 21],
             [0 + 21, 3 + 21], [3 + 21, 12 + 21], [12 + 21, 13 + 21], [13 + 21, 14 + 21],
             [0 + 21, 4 + 21], [4 + 21, 15 + 21], [15 + 21, 16 + 21], [16 + 21, 17 + 21],
             [0 + 21, 5 + 21], [5 + 21, 18 + 21], [18 + 21, 19 + 21], [19 + 21, 20 + 21],
             [0+42, 1+42], [1+42, 2+42], [2+42, 3+42], [3+42, 4+42], [4+42, 5+42], [5+42, 6+42],
             [0+42, 7+42], [7+42, 8+42], [8+42, 9+42], [9+42, 10+42], [10+42, 11+42], [0+42, 12+42],
             [12+42, 13+42], [13+42, 14+42], [14+42, 15+42], [15+42, 16+42],
             [0+42, 17+42], [17+42, 18+42], [18+42, 19+42], [19+42, 20+42], [20+42, 21+42],
             [0+42, 22+42], [22+42, 23+42], [23+42, 24+42], [24+42, 25+42], [25+42, 26+42]
             ]
    i = 0
    for line in lines:
        i += 1
        if i<=20:
            x = [joints[line[0]][0], joints[line[1]][0]]
            y = [joints[line[0]][1], joints[line[1]][1]]
            z = [joints[line[0]][2], joints[line[1]][2]]
            ax1.plot(x, y, z, color='r', linewidth=2)
        if i<=42 and i>20:
            x = [joints[line[0]][0], joints[line[1]][0]]
            y = [joints[line[0]][1], joints[line[1]][1]]
            z = [joints[line[0]][2], joints[line[1]][2]]
            ax1.plot(x, y, z, color='g', linewidth=2)
        if i>=42:
            x = [joints[line[0]][0], joints[line[1]][0]]
            y = [joints[line[0]][1], joints[line[1]][1]]
            z = [joints[line[0]][2], joints[line[1]][2]]
            ax1.plot(x, y, z, color='b', linewidth=2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend()
    plt.axis('off')
    plt.show()

def write_pcd(points, save_pcd_path):
    n = len(points)
    lines = []
    for i in range(n):
        x, y, z = points[i]
        lines.append('{:.6f} {:.6f} {:.6f} {}'.format( \
            x, y, z, 0))
    with open(save_pcd_path, 'w') as f:
        f.write(HEADER.format(n, n))
        f.write('\n'.join(lines))

if __name__ == '__main__':
    map_loader = Map_Loader()
    shadows = []
    for i in range(map_loader.label.shape[0]):
        tip_keys, pip_keys, pip_mcp, tip_pip, frame, local_points, shadow_points = map_loader.map(i)
        meshes = []
        radius = 0.002
        vc = [1.0, 0.0, 0.0]
        #points = np.append(local_points,shadow_points/1000.0)
        joint_mesh = Mesh(vertices=local_points, radius=radius, vc=vc)
        joint_mesh1 = Mesh(vertices=shadow_points/1000.0, radius=radius, vc=vc)

        meshes.append(joint_mesh)
        meshes.append(joint_mesh1)

        #meshes[0].show()
        #meshes[1].show()
        j_p = np.load('j_p_open.npy')
        jp = j_p[:,:27,:]
        jp = jp.reshape(27,3)
        trans1 = np.array([[0,0,1],[0,-1,0],[1,0,0]])
        trans2 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        trans3 = np.array([[1,0,0],[0,0,-1],[0,1,0]])

        trans = np.dot(trans1,trans2)
        #trans = np.array([[0,0,-1],[1,0,0],[0,-1,0]])
        jp = np.dot(jp,trans)
        t = np.array([[1,0,0,-11.0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,1]])
        z = np.ones(27)
        jp = np.insert(jp, 3, values=z, axis=1)
        jp = np.dot(t,jp.T)
        jp = jp.T[:,:3]
        #show_hand(shadow_points, 'shadow')
        points = np.vstack((local_points, shadow_points))
        points2 = np.vstack((points, jp))
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

        #----------------写成pcd文件并保存在data文件夹下面------------------
        #path='./data/{}.pcd'.format(i)
        #write_pcd(shadow_points,path)

        #show_data(local_points)
        #show_data(shadow_points)
        #np.save('shadow',shadow_points)


        show_data(local_points)

        show_data2(points2)
        shadows.append(shadow_points)
    shadows = np.array(shadows)
    #np.save('shadow_keypoints_new',shadows)
    print(shadows)