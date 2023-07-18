import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler
from model_classify_grasp import Local_feature2
from dataloader_grasptype import RTJlabel,GraspNetCollate
from tqdm import tqdm
import time
import torch.nn.functional as F
import copy
import torch.backends.cudnn as cudnn

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=33, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重. 当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.255
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]        [B*N个标签(假设框中有目标)]，[B个标签]
        :return:
        """

        # 固定类别维度，其余合并(总检测框数或总批次数)，preds.size(-1)是最后一个维度
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)

        # 使用log_softmax解决溢出问题，方便交叉熵计算而不用考虑值域
        preds_logsoft = F.log_softmax(preds, dim=1)

        # log_softmax是softmax+log运算，那再exp就算回去了变成softmax
        preds_softmax = torch.exp(preds_logsoft)

        # 这部分实现nll_loss ( crossentropy = log_softmax + nll)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))

        self.alpha = self.alpha.gather(0, labels.view(-1))

        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        # torch.mul 矩阵对应位置相乘，大小一致
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        # torch.t()求转置
        loss = torch.mul(self.alpha, loss.t())
        # print(loss.size()) [1,5]

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class RefinementBlock:
    def __init__(self,on_gpu=True):

        self.epoch = 0
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.max_epoch = 800

        if on_gpu and torch.cuda.is_available():
            print('train on gpu')
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            print('warning! train on cpu')

    def train(self):
        train_dataloader = DataLoader(RTJlabel(mode='train'), num_workers=8, batch_size=self.batch_size,
                                      shuffle=True, drop_last=False)#, collate_fn=GraspNetCollate)
        device = self.device
        model = Local_feature2(33).to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)

        steps = 0
        criterion = nn.BCELoss()
        train_loss = 0.0
        total_time = 0.0
        batch_size = self.batch_size
        for epoch in range(self.max_epoch):

            count = 0.0
            model.train()
            id = 0
            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))


            accs = 0.0
            for i,batch in loop:
                steps += 1
                label = batch[5].to(device)
                output = model(batch[0].permute(0, 2, 1).to(device)).to(device)
                outputs = torch.zeros(self.batch_size,33).to(device)
                for j in range(output.shape[0]):
                    for i in range(output.shape[1]):
                        if output[j][i].item() >= 0.5:
                            outputs[j][i] = torch.tensor(1.0)
                        else:
                            outputs[j][i] = torch.tensor(0.0)

                loss = criterion(output, label)
                time.sleep(0.1)
                start_time = time.time()

                optimizer.zero_grad()
                # 梯度反传
                loss = loss.requires_grad_()
                loss.backward()
                # 保留梯度
                optimizer.step()

                end_time = time.time()
                total_time += (end_time - start_time)

                count += batch_size
                train_loss += loss.item() * batch_size
                id += 1

                acc = 0
                for i in range(label.shape[0]):
                    for j in range(label.shape[1]):
                        if outputs[i][j] == label[i][j]:
                            acc += 1
                acc = acc / (33.0 * self.batch_size)
                loop.set_description(f'Epoch [{epoch}/{self.max_epoch}]')
                loop.set_postfix(loss=loss.item(),acc=acc,lr=optimizer.param_groups[0]['lr'])#,losses=losses)
                accs += acc

            if epoch % 10== 0:
                print((accs/10400.0)*self.batch_size)
                best_model = copy.deepcopy(model)
                trainer.val(best_model)
                #

            if epoch < 100:
                scheduler.step()

        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, 'checkpoint/model.2022-1-099.pth')

    def val(self, models):
        val_dataloader = DataLoader(RTJlabel(mode='test'), num_workers=8, batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        device = self.device
        model = models
        accs = 0
        for i, batch in enumerate(val_dataloader):
            output = model(batch[0].permute(0, 2, 1).to(device)).to(device)
            label = batch[5].to(device)

            outputs = torch.zeros(self.batch_size, 33).to(device)
            for j in range(output.shape[0]):
                for i in range(output.shape[1]):
                    if output[j][i].item() >= 0.5:
                        outputs[j][i] = torch.tensor(1.0)
                    else:
                        outputs[j][i] = torch.tensor(0.0)
            acc = 0
            for i in range(label.shape[0]):
                for j in range(label.shape[1]):
                    if outputs[i][j] == label[i][j]:
                        acc += 1
            acc = acc / (33.0 * self.batch_size)
            accs += acc
        print('val_acc=', (accs / 2500.0)*self.batch_size )


    def test(self):
        val_dataloader = DataLoader(RTJlabel(mode='test'), num_workers=8, batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        device = self.device
        model = Local_feature2(33).to(device)
        cudnn.benchmark = True
        modelpath = 'checkpoint/model.2022-1-099.pth'
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint['net'])  # ,False
        model.eval().to(device)
        accs = 0
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                output = model(batch[0].permute(0, 2, 1).to(device)).to(device)
                label = batch[5].to(device)

                outputs = torch.zeros(self.batch_size, 33).to(device)
                for j in range(output.shape[0]):
                    for i in range(output.shape[1]):
                        if output[j][i].item() >= 0.5:
                            outputs[j][i] = torch.tensor(1.0)
                        else:
                            outputs[j][i] = torch.tensor(0.0)
                acc = 0
                for i in range(label.shape[0]):
                    print(outputs, label)
                    for j in range(label.shape[1]):
                        if outputs[i][j] == label[i][j]:
                            acc += 1
                acc = acc / (33.0 * self.batch_size)
                accs += acc
            print('test_acc=', (accs / 2500.0)*self.batch_size )

if __name__ == '__main__':
    trainer = RefinementBlock(on_gpu=True)
    trainer.train()
    trainer.test()

