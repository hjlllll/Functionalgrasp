'''
D-H parameter: https://blog.csdn.net/xiaolongwoaini99/article/details/80391462
'''
import numpy as np
import math

def quat2mat(q):
    R = np.array([1-2*(q[1]*q[1]+q[2]*q[2]), 2*(q[0]*q[1]-q[2]*q[3]), 2*(q[0]*q[2]+q[1]*q[3])],
                 [2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[0]*q[0]+q[2]*q[2]), 2*(q[1]*q[2]-q[0]*q[3])],
                 [2*(q[0]*q[2]-q[1]*q[3]), 2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[0]*q[0]+q[1]*q[1])])
    return R

def quat_multi(q1, q2):
    q3 = np.array([0, 0, 0, 0])
    q3[0]=q1[3]*q2[0]+q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]
    q3[1]=q1[3]*q2[1]-q1[0]*q2[2]+q1[1]*q2[3]+q1[2]*q2[0]
    q3[2]=q1[3]*q2[2]+q1[0]*q2[1]-q1[1]*q2[0]+q1[2]*q2[3]
    q3[3]=q1[3]*q2[3]-q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]
    q3=q3/q3.std()
    return q3

def mat4x4_xyz(x, y, z):
    trans = np.mat([[1, 0, 0, x],[0, 1, 0, y],[0, 0, 1, z],[0, 0, 0, 1]])
    return trans

def mat4x4_theta_x(theta_x):
    theta_x = theta_x*np.pi/180
    trans = np.mat([[1,                 0,                  0, 0],
                    [0, math.cos(theta_x), -math.sin(theta_x), 0],
                    [0, math.sin(theta_x),  math.cos(theta_x), 0],
                    [0,                 0,                  0, 1]])
    return trans

def mat4x4_theta_y(theta_y):
    theta_y = theta_y * np.pi / 180
    trans = np.mat([[ math.cos(theta_y), 0, math.sin(theta_y), 0],
                    [                 0, 1,                 0, 0],
                    [-math.sin(theta_y), 0, math.cos(theta_y), 0],
                    [                 0, 0,                 0, 1]])
    return trans

def mat4x4_theta_z(theta_z):
    theta_z = theta_z * np.pi / 180
    trans = np.mat([[math.cos(theta_z), -math.sin(theta_z), 0, 0],
                    [math.sin(theta_z),  math.cos(theta_z), 0, 0],
                    [                0,                  0, 1, 0],
                    [                0,                  0, 0, 1]])
    return trans

p0 = np.mat([0,0,0,1]).T
px = np.mat([1,0,0,1]).T
py = np.mat([0,1,0,1]).T
pz = np.mat([0,0,1,1]).T
p00 = np.hstack((p0, px, py, pz))
# p01 = p00+10
# p02 = p00+20
# p03 = p00+5
# print(p00)
# print(p01)

# # little finger
ml10 = mat4x4_xyz(46.4, 3.6, 0)
ml11 = mat4x4_theta_x(90)
ml12 = mat4x4_theta_y(55)
ml13 = mat4x4_xyz(0, 0, 21)
ML1 = ml10*ml11*ml12*ml13

ml20 = mat4x4_xyz(37, 0, 0)
ml21 = mat4x4_theta_x(-90)
ML2 = ml20*ml21

# print(ml10)
# print((ml10*p0).T)
# print(ml10*ml11)
# print((ml10*ml11*p0).T)
# print(ml10*ml11*ml12*ml13)
# print((ml10*ml11*ml12*ml13*p0).T)
# print(ml10*ml11*ml12*ml13*ml20)
# print((ml10*ml11*ml12*ml13*ml20*p0).T)

ml30 = mat4x4_theta_z(305)
ml31 = mat4x4_theta_x(90)
ML3 = ml30*ml31

ml40 = mat4x4_xyz(45, 0, 0)
ML4 = ml40

ml50 = mat4x4_xyz(25, 0, 0)
ML5 = ml50

ml60 = mat4x4_xyz(30, 0, 0)
ML6 = ml60

pl1 = np.hstack((ML1*p0, ML1*px, ML1*py, ML1*pz))
pl2 = np.hstack((ML1*ML2*p0, ML1*ML2*px, ML1*ML2*py, ML1*ML2*pz))
pl3 = np.hstack((ML1*ML2*ML3*p0, ML1*ML2*ML3*px, ML1*ML2*ML3*py, ML1*ML2*ML3*pz))
pl4 = np.hstack((ML1*ML2*ML3*ML4*p0, ML1*ML2*ML3*ML4*px, ML1*ML2*ML3*ML4*py, ML1*ML2*ML3*ML4*pz))
pl5 = np.hstack((ML1*ML2*ML3*ML4*ML5*p0, ML1*ML2*ML3*ML4*ML5*px, ML1*ML2*ML3*ML4*ML5*py, ML1*ML2*ML3*ML4*ML5*pz))
pl6 = np.hstack((ML1*ML2*ML3*ML4*ML5*ML6*p0, ML1*ML2*ML3*ML4*ML5*ML6*px, ML1*ML2*ML3*ML4*ML5*ML6*py, ML1*ML2*ML3*ML4*ML5*ML6*pz))

# # ring finger
mr10 = mat4x4_xyz(92, 0, 0)
mr11 = mat4x4_theta_x(90)
mr12 = mat4x4_theta_y(55)
mr13 = mat4x4_theta_x(-90)
MR1 = mr10*mr11*mr12*mr13

mr20 = mat4x4_theta_z(305)
mr21 = mat4x4_theta_x(90)
MR2 = mr20*mr21

mr30 = mat4x4_xyz(45, 0, 0)
MR3 = mr30

mr40 = mat4x4_xyz(25, 0, 0)
MR4 = mr40

mr50 = mat4x4_xyz(30, 0, 0)
MR5 = mr50

pr1 = np.hstack((MR1*p0, MR1*px, MR1*py, MR1*pz))
pr2 = np.hstack((MR1*MR2*p0, MR1*MR2*px, MR1*MR2*py, MR1*MR2*pz))
pr3 = np.hstack((MR1*MR2*MR3*p0, MR1*MR2*MR3*px, MR1*MR2*MR3*py, MR1*MR2*MR3*pz))
pr4 = np.hstack((MR1*MR2*MR3*MR4*p0, MR1*MR2*MR3*MR4*px, MR1*MR2*MR3*MR4*py, MR1*MR2*MR3*MR4*pz))
pr5 = np.hstack((MR1*MR2*MR3*MR4*MR5*p0, MR1*MR2*MR3*MR4*MR5*px, MR1*MR2*MR3*MR4*MR5*py, MR1*MR2*MR3*MR4*MR5*pz))

# # middle finger
mm10 = mat4x4_xyz(95, -22, 1)
MM1 = mm10

mm20 = mat4x4_theta_x(90)
MM2 = mm20

mm30 = mat4x4_xyz(45, 0, 0)
MM3 = mm30

mm40 = mat4x4_xyz(25, 0, 0)
MM4 = mm40

mm50 = mat4x4_xyz(30, 0, 0)
MM5 = mm50

pm1 = np.hstack((MM1*p0, MM1*px, MM1*py, MM1*pz))
pm2 = np.hstack((MM1*MM2*p0, MM1*MM2*px, MM1*MM2*py, MM1*MM2*pz))
pm3 = np.hstack((MM1*MM2*MM3*p0, MM1*MM2*MM3*px, MM1*MM2*MM3*py, MM1*MM2*MM3*pz))
pm4 = np.hstack((MM1*MM2*MM3*MM4*p0, MM1*MM2*MM3*MM4*px, MM1*MM2*MM3*MM4*py, MM1*MM2*MM3*MM4*pz))
pm5 = np.hstack((MM1*MM2*MM3*MM4*MM5*p0, MM1*MM2*MM3*MM4*MM5*px, MM1*MM2*MM3*MM4*MM5*py, MM1*MM2*MM3*MM4*MM5*pz))

# # index finger
mi10 = mat4x4_xyz(91, -44, 2)
MI1 = mi10

mi20 = mat4x4_theta_x(90)
MI2 = mi20

mi30 = mat4x4_xyz(45, 0, 0)
MI3 = mi30

mi40 = mat4x4_xyz(25, 0, 0)
MI4 = mi40

mi50 = mat4x4_xyz(30, 0, 0)
MI5 = mi50

pi1 = np.hstack((MI1*p0, MI1*px, MI1*py, MI1*pz))
pi2 = np.hstack((MI1*MI2*p0, MI1*MI2*px, MI1*MI2*py, MI1*MI2*pz))
pi3 = np.hstack((MI1*MI2*MI3*p0, MI1*MI2*MI3*px, MI1*MI2*MI3*py, MI1*MI2*MI3*pz))
pi4 = np.hstack((MI1*MI2*MI3*MI4*p0, MI1*MI2*MI3*MI4*px, MI1*MI2*MI3*MI4*py, MI1*MI2*MI3*MI4*pz))
pi5 = np.hstack((MI1*MI2*MI3*MI4*MI5*p0, MI1*MI2*MI3*MI4*MI5*px, MI1*MI2*MI3*MI4*MI5*py, MI1*MI2*MI3*MI4*MI5*pz))

# # thumb
mt10 = mat4x4_xyz(-70.5, -128, 8)
mt11 = mat4x4_theta_z(45)
mt12 = mat4x4_theta_x(90)
mt13 = mat4x4_xyz(130, 0, 0)
mt14 = mat4x4_theta_z(-90)
MT1 = mt10*mt11*mt12*mt13*mt14

mt20 = mat4x4_theta_x(90)
MT2 = mt20

mt30 = mat4x4_theta_z(90)
mt31 = mat4x4_xyz(38, 0, 0)
mt32 = mat4x4_theta_x(-90)
MT3 = mt30*mt31*mt32

mt40 = mat4x4_xyz(32, 0, 0)
MT4 = mt40

mt50 = mat4x4_xyz(33, 0, 0)
MT5 = mt50

pt1 = np.hstack((MT1*p0, MT1*px, MT1*py, MT1*pz))
pt2 = np.hstack((MT1*MT2*p0, MT1*MT2*px, MT1*MT2*py, MT1*MT2*pz))
pt3 = np.hstack((MT1*MT2*MT3*p0, MT1*MT2*MT3*px, MT1*MT2*MT3*py, MT1*MT2*MT3*pz))
pt4 = np.hstack((MT1*MT2*MT3*MT4*p0, MT1*MT2*MT3*MT4*px, MT1*MT2*MT3*MT4*py, MT1*MT2*MT3*MT4*pz))
pt5 = np.hstack((MT1*MT2*MT3*MT4*MT5*p0, MT1*MT2*MT3*MT4*MT5*px, MT1*MT2*MT3*MT4*MT5*py, MT1*MT2*MT3*MT4*MT5*pz))


# # plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1, projection='3d')
p_list = [p00, pl1, pl2, pl3, pl4, pl5, pl6, pr1, pr2, pr3, pr4, pr5, pm1, pm2, pm3, pm4, pm5, pi1, pi2, pi3, pi4, pi5, pt1, pt2, pt3, pt4, pt5]
Joints = np.hstack([p00[:,0], pl1[:,0], pl2[:,0], pl3[:,0], pl4[:,0], pl5[:,0], pl6[:,0], pr1[:,0], pr2[:,0], pr3[:,0], pr4[:,0], pr5[:,0],
                    pm1[:,0], pm2[:,0], pm3[:,0], pm4[:,0], pm5[:,0], pi1[:,0], pi2[:,0], pi3[:,0], pi4[:,0], pi5[:,0], pt1[:,0], pt2[:,0], pt3[:,0], pt4[:,0], pt5[:,0]])
Joints_T = np.asarray(Joints.T)
ax1.scatter(Joints_T[:,0], Joints_T[:,1], Joints_T[:,2], c=np.array([[0, 0, 0]]), marker='x', linewidths=2)
ax1.scatter([100, 0, 0], [0, 100, 0], [0, 0, 100], c=np.array([[1, 0, 0]]), marker='o', linewidths=2)
lines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [10, 11], [0, 12], [12, 13], [13, 14], [14, 15], [15, 16],
         [0, 17], [17, 18], [18, 19], [19, 20], [20, 21], [0, 22], [22, 23], [23, 24], [24, 25], [25, 26]]
for line in lines:
    x = [Joints[:,line[0]][0,0], Joints[:,line[1]][0,0]]
    y = [Joints[:,line[0]][1,0], Joints[:,line[1]][1,0]]
    z = [Joints[:,line[0]][2,0], Joints[:,line[1]][2,0]]
    ax1.plot(x, y, z, c='r', linewidth=2)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.legend()
plt.axis('off')

ax2 = fig.add_subplot(1,2,2, projection='3d')
r90 = mat4x4_theta_z(90)
ML4 = ML4 * r90
pl1 = np.hstack((ML1*p0, ML1*px, ML1*py, ML1*pz))
pl2 = np.hstack((ML1*ML2*p0, ML1*ML2*px, ML1*ML2*py, ML1*ML2*pz))
pl3 = np.hstack((ML1*ML2*ML3*p0, ML1*ML2*ML3*px, ML1*ML2*ML3*py, ML1*ML2*ML3*pz))
pl4 = np.hstack((ML1*ML2*ML3*ML4*p0, ML1*ML2*ML3*ML4*px, ML1*ML2*ML3*ML4*py, ML1*ML2*ML3*ML4*pz))
pl5 = np.hstack((ML1*ML2*ML3*ML4*ML5*p0, ML1*ML2*ML3*ML4*ML5*px, ML1*ML2*ML3*ML4*ML5*py, ML1*ML2*ML3*ML4*ML5*pz))
pl6 = np.hstack((ML1*ML2*ML3*ML4*ML5*ML6*p0, ML1*ML2*ML3*ML4*ML5*ML6*px, ML1*ML2*ML3*ML4*ML5*ML6*py, ML1*ML2*ML3*ML4*ML5*ML6*pz))
p_list = [p00, pl1, pl2, pl3, pl4, pl5, pl6, pr1, pr2, pr3, pr4, pr5, pm1, pm2, pm3, pm4, pm5, pi1, pi2, pi3, pi4, pi5, pt1, pt2, pt3, pt4, pt5]
Joints = np.hstack([p00[:,0], pl1[:,0], pl2[:,0], pl3[:,0], pl4[:,0], pl5[:,0], pl6[:,0], pr1[:,0], pr2[:,0], pr3[:,0], pr4[:,0], pr5[:,0],
                    pm1[:,0], pm2[:,0], pm3[:,0], pm4[:,0], pm5[:,0], pi1[:,0], pi2[:,0], pi3[:,0], pi4[:,0], pi5[:,0], pt1[:,0], pt2[:,0], pt3[:,0], pt4[:,0], pt5[:,0]])
Joints_T = np.asarray(Joints.T)
ax2.scatter(Joints_T[:,0], Joints_T[:,1], Joints_T[:,2], c=np.array([[0, 0, 0]]), marker='x', linewidths=2)
ax2.scatter([100, 0, 0], [0, 100, 0], [0, 0, 100], c=np.array([[1, 0, 0]]), marker='o', linewidths=2)
lines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [10, 11], [0, 12], [12, 13], [13, 14], [14, 15], [15, 16],
         [0, 17], [17, 18], [18, 19], [19, 20], [20, 21], [0, 22], [22, 23], [23, 24], [24, 25], [25, 26]]
for line in lines:
    x = [Joints[:,line[0]][0,0], Joints[:,line[1]][0,0]]
    y = [Joints[:,line[0]][1,0], Joints[:,line[1]][1,0]]
    z = [Joints[:,line[0]][2,0], Joints[:,line[1]][2,0]]
    ax2.plot(x, y, z, c='r', linewidth=2)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.legend()
plt.axis('off')
plt.show()

M_shadowhand = np.asarray([ML1, ML2, ML3, ML4, ML5, ML6, MR1, MR2, MR3, MR4, MR5, MM1, MM2, MM3, MM4, MM5, MI1, MI2, MI3, MI4, MI5, MT1, MT2, MT3, MT4, MT5])
print(M_shadowhand.shape)
# np.save('./M_shadowhand.npy', M_shadowhand)
M_sh = np.load('../utils/M_shadowhand.npy')
print((M_shadowhand == M_sh).all())


#########################################################################################################################################################################
# # 计算辅助点，各link尺寸由draft_of_shadow2.py计算，各参数对应参考onenote图
M_F_Prox_o = mat4x4_xyz(22.5, 0, 0)
M_F_Prox_up = mat4x4_xyz(0, 9, 0)
M_F_Prox_l = mat4x4_xyz(0, 0, -10)
M_F_Prox_r = mat4x4_xyz(0, 0, 10)

M_F_Middle_o = mat4x4_xyz(12.5, 0, 0)
M_F_Middle_up = mat4x4_xyz(0, 8.5, 0)
M_F_Middle_l = mat4x4_xyz(0, 0, -9)
M_F_Middle_r = mat4x4_xyz(0, 0, 9)

M_F_Distal_o = mat4x4_xyz(15, 0, 0)
M_F_Distal_up = mat4x4_xyz(0, 6.4, 0)
M_F_Distal_l = mat4x4_xyz(0, 0, -8)
M_F_Distal_r = mat4x4_xyz(0, 0, 8)

M_TH_Proximal_o = mat4x4_theta_z(90) * mat4x4_xyz(19, 0, 0) * mat4x4_theta_x(-90)
M_TH_Proximal_up = mat4x4_xyz(0, 13, 0)
M_TH_Proximal_l = mat4x4_xyz(0, 0, -13)
M_TH_Proximal_r = mat4x4_xyz(0, 0, 13)

M_TH_Middle_o = mat4x4_xyz(16, 0, 0)
M_TH_Middle_up = mat4x4_xyz(0, 11, 0)
M_TH_Middle_l = mat4x4_xyz(0, 0, -11)
M_TH_Middle_r = mat4x4_xyz(0, 0, 11)

M_TH_Distal_o = mat4x4_xyz(16.5, 0, 0)
M_TH_Distal_up = mat4x4_xyz(0, 9, 0)
M_TH_Distal_l = mat4x4_xyz(0, 0, -11)
M_TH_Distal_r = mat4x4_xyz(0, 0, 11)

M_ass = np.asarray([[M_F_Prox_o,      M_F_Prox_up,      M_F_Prox_l,      M_F_Prox_r],
                    [M_F_Middle_o,    M_F_Middle_up,    M_F_Middle_l,    M_F_Middle_r],
                    [M_F_Distal_o,    M_F_Distal_up,    M_F_Distal_l,    M_F_Distal_r],
                    [M_TH_Proximal_o, M_TH_Proximal_up, M_TH_Proximal_l, M_TH_Proximal_r],
                    [M_TH_Middle_o,   M_TH_Middle_up,   M_TH_Middle_l,   M_TH_Middle_r],
                    [M_TH_Distal_o,   M_TH_Distal_up,   M_TH_Distal_l,   M_TH_Distal_r]])
print(M_ass.shape)
# np.save('./M_ass.npy', M_ass)
M_asss = np.load('../utils/M_ass.npy')
print((M_ass == M_asss).all())