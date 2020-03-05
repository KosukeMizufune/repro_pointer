import logging

import torch
from torch import nn
from torch.nn import functional as F


class STNkd(nn.Module):
    def __init__(self, k=3):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k
        self.iden = torch.eye(self.k).flatten()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x + torch.eye(self.k, device=x.device).flatten()
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STNkd(3)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        batchsize, dim, p_num = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if dim > 3:
            x, feature = x.split(3, dim=2)
        x = torch.bmm(x, trans)
        if dim > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, p_num)
            return torch.cat([x, pointfeat], 1), trans_feat


class PointNetClassifier(nn.Module):
    def __init__(self, n_class=40, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.feat = PointNetEncoder(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_feat
    
    def predict(self, x):
        return self.forward(x)[0]


def _frobenius_loss(trans):
    d = trans.size()[1]
    iden = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        iden = iden.cuda()
    norm = \
        torch.norm(torch.bmm(trans, trans.transpose(2, 1) - iden), dim=(1, 2))
    loss = torch.mean(norm)
    return loss


class PointNetLoss(nn.Module):
    def __init__(self, reg_scale=0.001, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.reg_scale = reg_scale

    def forward(self, pred, target):
        pred_classifier, trans = pred
        label_loss = F.cross_entropy(pred_classifier, target)
        frobenius_loss = _frobenius_loss(trans)
        return label_loss + self.reg_scale * frobenius_loss


class PointNetSegmentation(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x, trans_feat
