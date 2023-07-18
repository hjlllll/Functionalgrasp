#coding=utf-8
# ############################################################################################################################################
# import os
# def file_name(file_dir):
#     i = 0
#     listName = []
#     #for root, dirs, files in os.walk(file_dir):
#     for dir in sorted(os.listdir(file_dir)):
#         if os.path.splitext(dir)[1] == '.xml':
#             i += 1
#             print(dir[:-24])  # 当前目录路径
#             listName.append(dir);
#             #print(dirs)  # 当前路径下所有子目录
#             # print(files)  # 当前路径下所有非目录子文件
#     print(i)
#     return listName
#
# file_dir = r'/home/lm/Documents/ddg_data/grasp_dataset/good_shapes'
# file_name(file_dir)

# ############################################################################################################################################
# def getFileName2(path, suffix):
#     i=0
#     object_name = []
#     object_name_plus = []
#     object_name_plus_path = []
#     for root, dirs, files in os.walk(path, topdown=False):
#         for name in sorted(files):
#             if os.path.splitext(name)[1] == suffix:
#                 # print(os.path.splitext(name)[0][:-20])
#                 i += 1
#                 object_name.append(name[:-24])
#                 object_name_plus.append(name)
#                 object_name_plus_path.append(os.path.join(root, name))
#                 # print(name[:-24])
#     print(i)
#     return object_name_plus, object_name_plus, object_name_plus_path
#
# # getFileName2(file_dir,'.xml')

# ############################################################################################################################################
# def file_name111(file_dir,dir2):
#     i = 0
#     j = 0
#     listName = []
#     #for root, dirs, files in os.walk(file_dir):
#     for dir in sorted(os.listdir(file_dir)):
#         if os.path.splitext(dir)[1] == '.xml':
#             listName.append(dir[:-24]);
#     for dir in sorted(os.listdir(dir2)):
#         i+=1
#         # print(dir[:-10])
#         if dir[:-10] in listName:
#             j+=1
#         else:
#             print("nonononoonoonono")
#     print(i,j)
#
# file_dir = r'/home/lm/Documents/ddg_data/grasp_dataset/good_shapes'
# dir2 = r'/home/lm/Documents/ddg_data/grasp_dataset/grasps'
# # file_name111(file_dir,dir2)

# ###############################################################################################################################################
# # # https://www.cnblogs.com/fnng/p/3581433.html
# import xml.dom.minidom
# import numpy as np
#
# #打开xml文档
# dom = xml.dom.minidom.parse('/home/lm/Documents/ddg_data/grasp_dataset/grasps/bigbird_3m_high_tack_spray_adhesive.xml_0_0_0/grasp1.xml')
#
# #得到文档元素对象
# filename_tag = dom.getElementsByTagName('filename')
# filename = filename_tag[0].firstChild.data[:-4]
# print(filename + '.off')
# pose_tag = dom.getElementsByTagName('fullTransform')
# object_pose = pose_tag[0].firstChild.data
# object_pose = object_pose.replace('(',' ').replace(')',' ').replace('[',' ').replace(']',' ').replace('+',' ').split()
# print(object_pose)
# gripper_pose = pose_tag[1].firstChild.data
# gripper_pose = gripper_pose.replace('(',' ').replace(')',' ').replace('[',' ').replace(']',' ').replace('+',' ').split()
# print(gripper_pose)
# gripper_tag = dom.getElementsByTagName('dofValues')
# gripper_dof = gripper_tag[0].firstChild.data.replace('+','').split()
# print(gripper_dof)
#
# gripper = np.zeros((1, 4+3+18))
# obj_pose = np.zeros((1, 4+3))
# print(gripper, object_pose[0], object_pose[1], object_pose[2], object_pose[3], object_pose[4], object_pose[5], object_pose[6])
# gripper[0, :7] = np.asarray(object_pose).astype("float64")
# gripper[0, 7:] = np.asarray(gripper_dof).astype("float64")
# print(gripper)
# obj_pose[0] = np.asarray(gripper_pose).astype("float64")
# print(gripper_pose)
# print(obj_pose)



# ############################################################################################################################################
# # https://www.cnblogs.com/xiugeng/p/8635862.html
# # https://www.pythonheidong.com/blog/article/271644/
# # https://blog.csdn.net/kizgel/article/details/80597954
#
# import numpy as np
# import time
#
# # def read_off(dir):
# #     file = open(dir, 'r')
# #     if 'OFF' != file.readline().strip():
# #         raise('Not a valid OFF header')
# #     n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
# #     verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
# #     verts = np.asarray(verts)
# #     faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
# #     faces = np.asarray(faces)
# #     return verts, faces
#
# dir = r'/home/lm/Documents/ddg_data/grasp_dataset/good_shapes/bigbird_3m_high_tack_spray_adhesive_scaled.obj.smoothed.off'
# def safe_float(number):
#     try:
#         return float(number)
#     except:
#         return None
# # verts, faces = read_off(dir)
# time1 = time.time()
# file1 = open(dir, 'r')
# if 'OFF' != file1.readline().strip():
#     raise('Not a valid OFF header')
# n_verts, n_faces, n_dontknow = tuple([int(s) for s in file1.readline().strip().split(' ')])
# off_inf = file1.read().splitlines()
# verts0 = off_inf[:n_verts]
# verts = np.zeros((n_verts,3))
# i = 0
# for vert in verts0:
#     print(vert.split())
#     verts[i] = np.asarray(list(map(float,vert.split())))
# time2 = time.time()
# print('Time1:', time2-time1)
#
# time3 = time.time()
# file = open(dir, 'r')
# if 'OFF' != file.readline().strip():
#     raise('Not a valid OFF header')
# n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
# verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
# verts = np.asarray(verts)
# faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
# faces = np.asarray(faces)
# time4 = time.time()
# print('Time2:', time4-time3)
#
# time5 = time.time()
# file2 = open(dir, 'r')
# if 'OFF' != file2.readline().strip():
#     raise('Not a valid OFF header')
# n_verts, n_faces, n_dontknow = tuple([int(s) for s in file2.readline().strip().split(' ')])
# verts = np.loadtxt(dir, skiprows=2, max_rows=n_verts)
# faces = np.loadtxt(dir, skiprows=2+n_verts, max_rows=n_faces)
# time6 = time.time()
# print('Time3:', time6-time5)

############################################################################################################################################
import numpy as np
import random
import time

# t1 = time.time()
# aa = list(range(1,101,2))
# aa = np.asarray(aa)
# t2 = time.time()
# aaa = a = np.arange(1,101,2)
# t3 = time.time()
# # print(aa)
# # print(aaa)
# #
# # print(t2-t1)
# # print(t3-t2)
# bb = np.asarray(aa)
# print(bb)
#
# cc = list(range(50))
# print(cc)
# random.shuffle(cc)
# print(cc)
# print(aa[cc[:10]])

# cc = list(range(20))
# print(cc)
# a = 10
# b = 6
# print(cc[:a])
# print(cc[a:-b])
# print(cc[-b:])
# random.seed(420)
# random.shuffle(cc)
# print(cc)

# import random
# for i in range(5):
#     list1 = ['a', 1, 'faf', '3dfdf', 4, 'df']
#     print(list1)
#     random.seed(420)
#     random.shuffle(list1)
#     print(list1)
# print(list1[-3:])

################################################################################################################################
import torch
from FK_model import Shadowhand_FK
# parents:[22,], positions:[F,J,3], rotations:[F,J,4]
# base = torch.Tensor([[100,200,50,1,0,0,0], [100,200,50,1,0,0,0]])
# rotations = torch.Tensor([[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

base = torch.Tensor([[0,0,0,1,0,0,0]])
rotations = torch.Tensor([[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

print(base.shape, rotations.shape)
fk = Shadowhand_FK()
pp = fk.run(base, rotations)
print(pp)
ppp = pp.numpy()
