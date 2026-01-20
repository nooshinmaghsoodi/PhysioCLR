import torch.nn as nn

class PhysioCLRDecoder(nn.Module):
    """
    Lightweight 3-layer 1-D conv → MLP decoder (TBME paper §III-D)
    """

    def __init__(self, d_model: int, out_channels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(d_model, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(128, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):               # x (B,T,D)
        return self.net(x.transpose(1, 2))    # (B,C,T)
