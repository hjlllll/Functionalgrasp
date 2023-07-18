#手型的transformer结构demo
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder

class Local_feature2(nn.Module):#pointnet  # 调用需要输入 输入、输出参数，部位总个数、normal_channel
    def __init__(self, num_classes, normal_channel=False):  # numberclasses 物体种类/手指的数量
        super(Local_feature2, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = 0
        self.feat = PointNetEncoder(global_feat=True, feature_transform=False, channel=3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 33)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        x, trans, trans_feat = self.feat(l0_points)

        out = x.view(B, 1024)

        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.dropout(self.fc2(out))))
        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out

if __name__ == '__main__':
    x = torch.randn(2,3,20000)
    model = Local_feature2(33)
    x = model(x)
    print(x)
