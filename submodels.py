import torch
import torch.nn as nn
import torch.nn.functional as F


eiip_values = torch.tensor(
    [[0.1260],   # A
     [0.1340],   # C
     [0.0806],   # G
     [0.1335]],  # U/T
    dtype=torch.float32,
)
ncp_values = torch.tensor(
            [[1., 1., 1.],  # A
             [0., 1., 0.],  # C
             [1., 0., 0.],  # G
             [0., 0., 1.]], # U/T
            dtype=torch.float32,
        )


class Conv_layers(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 64,
            out_channels: int = 64,
            kernel1: int = 5,
            kernel2: int = 5,

    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel1,
            padding=kernel1 // 2,  # 保持长度 L 不变
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel2,
            padding=kernel2 // 2,  # 保持长度 L 不变
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        return x


class EIIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.eiip_embedding = nn.Embedding.from_pretrained(
            eiip_values,
            freeze=False,  # False -> 不更新; True -> 可以学
        )

        self.eiip_cnn = Conv_layers(
            in_channels=1,
            hidden_channels=32,
            out_channels=128,
        )

    def forward(self, num_seq: torch.Tensor) -> torch.Tensor:
        eiip = self.eiip_embedding(num_seq)

        return self.eiip_cnn(eiip)

class ONEHOT(nn.Module):
    def __init__(self):
        super().__init__()
        self.onehot_cnn = Conv_layers(
            in_channels=4,
            hidden_channels=128,
            out_channels=128,
        )

    def forward(self, num_seq: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(num_seq, num_classes=4).float()

        return self.onehot_cnn(onehot)

class NCP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ncp_embedding = nn.Embedding.from_pretrained(
            ncp_values,
            freeze=False,
        )
        self.ncp_cnn = Conv_layers(
            in_channels=3,
            hidden_channels=128,
            out_channels=128,
        )

    def forward(self, num_seq: torch.Tensor) -> torch.Tensor:
        ncp = self.ncp_embedding(num_seq)
        return self.ncp_cnn(ncp)

class ENAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.enac_cnn = Conv_layers(
            in_channels=4,
            hidden_channels=128,
            out_channels=128,
        )

    def forward(self, num_seq: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(num_seq, num_classes=4).float()  # use onehot to calculate enac value
        onehot_t = onehot.transpose(1, 2)

        x = F.avg_pool1d(onehot_t, kernel_size=5, stride=1, padding=2)  #The sliding window equals kernel_size

        enac = x.transpose(1, 2)
        return self.enac_cnn(enac)


