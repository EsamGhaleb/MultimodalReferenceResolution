import torch.nn as nn
import torch.nn.functional as F


class ObjectHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=256, num_classes=16, num_joints=27, hidden_dim=1024):
        super(ObjectHeadClassification, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
  
    def forward(self, feat):
        '''
            Input: (N, T, J, C)
        '''
        N, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 2, 3, 1)      # (N, T, J, C) -> (N, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, -1)           # (N, J*C)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat = self.fc2(feat)
        return feat


class ObjectHeadEmbed(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=256, num_joints=27, hidden_dim=1024):
        super(ObjectHeadEmbed, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)

    def forward(self, feat, gestures_similarity=False):
        '''
            Input: (N, T, J, C)
        '''
        N, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 2, 3, 1)      # (N, T, J, C) -> (N, J, C, T)
        feat = feat.mean(dim=-1)
        if gestures_similarity:
            return feat.mean(dim=1)
        feat = feat.reshape(N, -1)           # (N, J*C)
        feat = self.fc1(feat)
        # feat = feat.mean(dim=1)
        feat = F.normalize(feat, dim=-1)
        return feat


class ObjectNet(nn.Module):
    def __init__(
            self,
            backbone,
            dim_rep=512,
            num_classes=16,
            dropout_ratio=0.,
            version='class',
            hidden_dim=768,
            num_joints=27
    ):
        super(ObjectNet, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        if version == 'class':
            self.head = ObjectHeadClassification(
                dropout_ratio=dropout_ratio,
                dim_rep=dim_rep,
                num_classes=num_classes,
                num_joints=num_joints
            )
        elif version == 'embed':
            self.head = ObjectHeadEmbed(
                dropout_ratio=dropout_ratio,
                dim_rep=dim_rep,
                hidden_dim=hidden_dim,
                num_joints=num_joints
            )
        else:
            raise ValueError('Unknown ObjectNet version.')

    def forward(self, x, get_rep=True, gestures_similarity=False):
        '''
            Input: (N x T x 27 x 3)
        '''
        N, T, J, C = x.shape
        if get_rep:
            feat = self.backbone.get_representation(x)
        else:
            return self.backbone(x)
        feat = feat.reshape([N, T, self.feat_J, -1])      # (N, T, J, C)
        out = self.head(feat, gestures_similarity=gestures_similarity)
        return out
