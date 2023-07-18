import torch
from visdom import Visdom
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
from utils.FK_model import Shadowhand_FK
import time
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from utils.write_xml import write_xml


class IKData(Dataset):
    def __init__(self):
        self.label = np.load('shadow_keypoints_new.npy')
        self.label = torch.tensor(self.label)#.cuda()
        return

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        return self.label[item,:,:]



viz = Visdom()
loss_window = viz.line(
    X = [0],
    Y = [0],
    opts = {'xlabel': 'epochs', 'ylabel': 'loss_value', 'title': 'loss'}
)

class Model(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_channel, 64)
        self.linear2 = nn.Linear(64, 256)
        self.linear3 = nn.Linear(256, output_channel)
        self.last_op = nn.Tanh()

    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        #x = self.last_op(x)

        return x

def transpose(j_p):

    jp = j_p[:, :28, :]
    jp = jp.reshape(28, 3)
    trans1 = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    trans2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    trans3 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    trans = np.dot(trans1, trans2)
    trans = torch.tensor(trans, dtype=torch.float32).cuda()
    jp = torch.matmul(jp,trans)
    t = torch.tensor([[1, 0, 0, -11.0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]],dtype=torch.float32).cuda()
    z = torch.ones(28,1).cuda()
    jp = torch.cat((jp,z),dim=1)
    jp = torch.matmul(t,jp.T)
    jp = jp.T[:, :3]

    return jp

def show_data2(joints):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.scatter(joints[:21, 0], joints[:21, 1], joints[:21, 2], c=np.array([[1, 0, 0]]), marker='o', linewidths=0.2)
    ax1.scatter(joints[21:, 0], joints[21:, 1], joints[21:, 2], c=np.array([[0, 0, 1]]), marker='o', linewidths=0.2)
    lines = [[0, 1], [1, 6], [6, 7], [7, 8],
             [0, 2], [2, 9], [9, 10], [10, 11],
             [0, 3], [3, 12], [12, 13], [13, 14],
             [0, 4], [4, 15], [15, 16], [16, 17],
             [0, 5], [5, 18], [18, 19], [19, 20],
             [0+21, 1+21], [1+21, 2+21], [2+21, 3+21], [3+21, 4+21], [4+21, 5+21], [5+21, 6+21],
             [0+21, 7+21], [7+21, 8+21], [8+21, 9+21], [9+21, 10+21], [10+21, 11+21], [0+21, 12+21],
             [12+21, 13+21], [13+21, 14+21], [14+21, 15+21], [15+21, 16+21],
             [0+21, 17+21], [17+21, 18+21], [18+21, 19+21], [19+21, 20+21], [20+21, 21+21],
             [0+21, 22+21], [22+21, 23+21], [23+21, 24+21], [24+21, 25+21], [25+21, 26+21],[26+21,27+21]
             ]
    i = 0
    for line in lines:
        i += 1
        if i<=20:
            x = [joints[line[0]][0], joints[line[1]][0]]
            y = [joints[line[0]][1], joints[line[1]][1]]
            z = [joints[line[0]][2], joints[line[1]][2]]
            ax1.plot(x, y, z, color='r', linewidth=2)
        if i>20:
            x = [joints[line[0]][0], joints[line[1]][0]]
            y = [joints[line[0]][1], joints[line[1]][1]]
            z = [joints[line[0]][2], joints[line[1]][2]]
            ax1.plot(x, y, z, color='b', linewidth=2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend()
    plt.axis('off')
    plt.show()

def fk_run(outputs_base, outputs_a):
    # # 输入正向运动学层
    # 3 + 4
    #outputs_base = torch.cat((outputs_t / 5.0 * 1000, outputs_r), 1)
    # 17(18) -> 27
    outputs_base = torch.tensor(outputs_base,dtype=torch.float32)
    outputs_rotation = torch.zeros([outputs_a.shape[0], 27]).type_as(outputs_a)  # .cuda()
    outputs_rotation[:, 0:3] = outputs_a[:, 0:3]
    angle_2_pair = torch.ones([2, outputs_a.shape[0]]).cuda()
    angle_1_pair = torch.zeros([2, outputs_a.shape[0]]).cuda()
    angle_2_pair[0] = outputs_a[:, 3]
    angle_1_pair[0] = outputs_a[:, 3] - 1
    outputs_rotation[:, 3] = torch.min(angle_2_pair, 0)[0]
    outputs_rotation[:, 4] = torch.max(angle_1_pair, 0)[0]
    # outputs_rotation[:, 4] = outputs_rotation[:, 3] * 0.8

    outputs_rotation[:, 6:8] = outputs_a[:, 4:6]
    angle_2_pair[0] = outputs_a[:, 6]
    angle_1_pair[0] = outputs_a[:, 6] - 1
    outputs_rotation[:, 8] = torch.min(angle_2_pair, 0)[0]
    outputs_rotation[:, 9] = torch.max(angle_1_pair, 0)[0]
    # outputs_rotation[:, 9] = outputs_rotation[:, 8] * 0.8

    outputs_rotation[:, 11:13] = outputs_a[:, 7:9]
    angle_2_pair[0] = outputs_a[:, 9]
    angle_1_pair[0] = outputs_a[:, 9] - 1
    outputs_rotation[:, 13] = torch.min(angle_2_pair, 0)[0]
    outputs_rotation[:, 14] = torch.max(angle_1_pair, 0)[0]
    # outputs_rotation[:, 14] = outputs_rotation[:, 13] * 0.8

    outputs_rotation[:, 16:18] = outputs_a[:, 10:12]
    angle_2_pair[0] = outputs_a[:, 12]
    angle_1_pair[0] = outputs_a[:, 12] - 1
    outputs_rotation[:, 18] = torch.min(angle_2_pair, 0)[0]
    outputs_rotation[:, 19] = torch.max(angle_1_pair, 0)[0]
    # outputs_rotation[:, 19] = outputs_rotation[:, 18] * 0.8

    outputs_rotation[:, 21:26] = outputs_a[:, 13:]  # all
    fk = Shadowhand_FK()
    outputs_FK = fk.run(outputs_base, outputs_rotation * 1.5708)  # [F, J+1+A, 3]  #原始J+1个关键点，再加上A个辅助点

    return outputs_FK


class IKTrainBlock:
    def __init__(self,on_gpu=True):
        self.batch_size = 1
        self.learning_rate = 1e-3
        self.max_epoch = 300

        if on_gpu and torch.cuda.is_available():
            print('train on gpu')
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            print('warning! train on cpu')


    def train(self):
        train_dataloader = DataLoader(IKData(),num_workers=8, batch_size=self.batch_size,
                                          shuffle=False, drop_last=False)
        device = self.device
        model = Model(63,18).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97)
        steps = 0
        criterion = nn.MSELoss()
        for epoch in range(self.max_epoch):
            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            model.train()
            for i ,batch in loop:
                batchs = batch.to(device).reshape(1,63)
                batchs= torch.tensor(batchs,dtype=torch.float32)
                output = model(batchs)
                ###angle loss###
                angle_lower = torch.tensor(
                    [0., -20., 0., 0., -20., 0., 0., -20., 0., 0., -20., 0., 0., -60., 0., -12, -30.,
                     0.]).cuda() / 90.0  # 90.0  # * 1.5708 / 1.5708
                angle_upper = torch.tensor(
                    [0., 20., 90., 90., 20., 90., 90., 20., 90., 90., 20., 90., 90., 60., 70., 12., 30.,
                     90.]).cuda() / 90.0  # 90.0  # * 1.5708 / 1.5708

                # 更改了角度范围
                predict_j = output.unsqueeze(0)
                angle_lower_pair = torch.zeros([2, predict_j.reshape(-1).shape[0]]).cuda()  # 全0向量
                angle_upper_pair = torch.zeros([2, predict_j.reshape(-1).shape[0]]).cuda()
                angle_lower_pair[0] = angle_lower.repeat(predict_j.shape[0]) - predict_j.reshape(-1)  # 最小值减去角度
                angle_upper_pair[0] = predict_j.reshape(-1) - angle_upper.repeat(predict_j.shape[0])  # 最大值减去角度
                loss_angles = (torch.max(angle_lower_pair, 0)[0] + torch.max(angle_upper_pair, 0)[0]).sum()

                output_base = torch.tensor([[0,0,0,1,0,0,0]],dtype=torch.float32)
                output_FK = fk_run(output_base, output)
                output_points = transpose(output_FK)
                shadow_points = np.load('shadow_keypoints_new.npy')[i]
                shadow_points = torch.tensor(shadow_points,dtype=torch.float32).to(device)


                output_five = torch.zeros((11,3)).type_as(output_points)
                output_five[0] = output_points[6]
                output_five[1] = output_points[11]
                output_five[2] = output_points[16]
                output_five[3] = output_points[21]
                output_five[4] = output_points[27]

                output_five[5] = output_points[4]
                output_five[6] = output_points[9]
                output_five[7] = output_points[14]
                output_five[8] = output_points[19]
                output_five[9] = output_points[24]
                output_five[10] = output_points[22]

                shadow_five = torch.zeros((11,3)).type_as(shadow_points)
                shadow_five[0] = shadow_points[20]
                shadow_five[1] = shadow_points[17]
                shadow_five[2] = shadow_points[14]
                shadow_five[3] = shadow_points[11]
                shadow_five[4] = shadow_points[8]

                shadow_five[5] = shadow_points[18]
                shadow_five[6] = shadow_points[15]
                shadow_five[7] = shadow_points[12]
                shadow_five[8] = shadow_points[9]
                shadow_five[9] = shadow_points[6]
                output_five[10] = shadow_points[1]


                loss_a = torch.zeros(1).to(device)
                for i in range(output_five.shape[0]):
                    loss_a += torch.norm(output_five[i] - shadow_five[i])
                loss = loss_a + 80.0 * loss_angles

                time.sleep(0.1)
                viz.line(
                    X=[steps],
                    Y=[loss.item()],
                    win=loss_window,
                    update='append'
                )

                # 梯度清0
                optimizer.zero_grad()
                # 梯度反传
                #loss = loss.requires_grad_()
                loss.backward()
                # 保留梯度
                optimizer.step()

                loop.set_description(f'Epoch [{epoch}/{self.max_epoch}]')
                loop.set_postfix(loss=loss.item(),loss_angles=loss_angles, lr=optimizer.param_groups[0]['lr'])  # ,losses=losses)
                steps += 1
                # visiualize
                # if epoch == 40 or epoch == 80 or epoch == 120:
                #     shadows = shadow_points.cpu().detach().numpy()
                #     outputs = output_points.cpu().detach().numpy()
                #     points = np.vstack((shadows,outputs))
                #
                #     show_data2(points)

            scheduler.step()
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, 'checkpoint/model_ik_solver_new.pth')


    def test(self):
        train_dataloader = DataLoader(IKData(), num_workers=8, batch_size=self.batch_size,
                                      shuffle=False, drop_last=False)
        device = self.device
        model = Model(63, 18).to(device)
        cudnn.benchmark = True
        modelpath = 'checkpoint/model_ik_solver_new.pth'
        checkpoint = torch.load(modelpath)  # modelpath是你要加载训练好的模型文件地址
        model.load_state_dict(checkpoint['net'])  # ,False
        model.eval().to(device)
        with torch.no_grad():
            grasp_33_angles = []
            for i, batch in enumerate(train_dataloader):
                batchs = batch.to(device).reshape(1, 63)
                batchs = torch.tensor(batchs, dtype=torch.float32)
                output = model(batchs)
                output_base = torch.tensor([[0, 0, 0, 1, 0, 0, 0]], dtype=torch.float32)
                output_FK = fk_run(output_base, output)
                output_points = transpose(output_FK)
                # output_points = torch.tensor(output_points).to(device)
                # output_points = output_points.clone().detach().requires_grad_(True)
                shadow_points = np.load('shadow_keypoints_new.npy')[i]
                shadow_points = torch.tensor(shadow_points, dtype=torch.float32).to(device)
                shadows = shadow_points.cpu().detach().numpy()
                outputs = output_points.cpu().detach().numpy()
                points = np.vstack((shadows,outputs))
                show_data2(points)

                output = output.cpu().detach().numpy()
                grasp_33_angles.append(output)
            # grasp_33_angles = np.array(grasp_33_angles)
            # np.save('grasp_33_angles_new',grasp_33_angles)


                results_xml_path = '../result_xml'
                obj_names = '/good_shapes/' + str('gd_pliers_poisson_010') + '_scaled.obj.smoothed' + '.xml'
                # print(obj_names)

                grip_r = np.array([[1,0,0,0]])
                grip_t = np.array([[0,0,0]])
                grip_a = output * 1.5708


                write_xml(obj_name=obj_names, r=grip_r[0], t=grip_t[0], a=grip_a[0],
                        path=results_xml_path + '/3grasptype/epoch{}.xml'.format(i),
                        mode='train', rs=(21, 'real'))

if __name__ == '__main__':
    trainer = IKTrainBlock(on_gpu=True)
    trainer.test()
    trainer.train()


