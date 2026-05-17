import torch as t
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import FPNBackbone
from config import config


class CTCModel(nn.Module):
    def __init__(self, num_classes=11):
        super(CTCModel, self).__init__()
        self.backbone = FPNBackbone()
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm = nn.LSTM(config.multiscale_feat_dim, 256, num_layers=2, bidirectional=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.height_pool(feat)
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C, W).permute(2, 0, 1)
        feat, _ = self.lstm(feat)
        logit = self.fc(feat)
        return F.log_softmax(logit, dim=2)
