

import torch.nn as nn
import torch.nn.functional as F
import math

import torch


def get_sinusoidal_embedding(max_steps, embedding_dim):
    """
    预先生成 sinusoidal 时间步 embedding 表，输出 shape: [max_steps, embedding_dim]
    """
    emb = torch.zeros(max_steps, embedding_dim)
    position = torch.arange(0, max_steps, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embedding_dim))
    emb[:, 0::2] = torch.sin(position * div_term)
    emb[:, 1::2] = torch.cos(position * div_term)
    return emb

class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderReconstructive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        self.mp1 = nn.MaxPool2d(2)
        
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )
        self.mp2 = nn.MaxPool2d(2)
        
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=1, padding=0)
        )
        
    
    def forward(self, x):
        x1 = self.block1(x)             # 32x32
        x2 = self.block2(self.mp1(x1))  # 16x16
        x3 = self.block3(self.mp2(x2))  # 8x8
        return x3

class DecoderReconstructive(nn.Module):
    def __init__(self, base_width, out_channels):
        super(DecoderReconstructive, self).__init__()
        self.f_conv = nn.Conv2d(base_width * 4, base_width * 4, kernel_size=1, padding=0)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(base_width * 4 + 1, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_width * 2 + 1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(base_width + 1, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, mask):
        """
        x: shape [B, base_width*4, 8, 8]
        mask: shape [B, 1, 32, 32]
        """
        x = self.f_conv(x)
        x = self.up1(x)  # [B, base_width*4, 16, 16]
        mask1 = F.interpolate(mask, size=x.shape[2:], mode='nearest')  # [B, 1, 16, 16]
        x = torch.cat([x, mask1], dim=1)  # [B, base_width*4+1, 16, 16]
        x = self.conv1(x)
        
        x = self.up2(x)  # [B, base_width*2, 32, 32]
        mask2 = F.interpolate(mask, size=x.shape[2:], mode='nearest')
        x = torch.cat([x, mask2], dim=1)  # [B, base_width*2+1, 32, 32]
        x = self.conv2(x)
        
        mask3 = F.interpolate(mask, size=x.shape[2:], mode='nearest')
        x = torch.cat([x, mask3], dim=1)  # [B, base_width+1, 32, 32]
        x = self.final_conv(x)           # [B, out_channels, 32, 32]
        return x



# ========== 5) 最终网络 ReconstructiveSubNetwork ==========
class FARM(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_width=64, t_emb_dim=128, max_timesteps=1000):
       
        super(FARM, self).__init__()
        
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channels)
        
        
        sinus_emb = get_sinusoidal_embedding(max_timesteps, t_emb_dim)
        
        self.time_embedding = nn.Embedding(max_timesteps, t_emb_dim)
        self.time_embedding.weight.data.copy_(sinus_emb)
       
        self.t_proj = nn.Linear(t_emb_dim, base_width * 4)
        
        self.bg_weight_map_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, 256),   # 先到256维
            nn.ReLU(inplace=True),
            nn.Linear(256, 8 * 8)          # 输出 8x8 个数
        )



    def forward(self, x, mask, t):
       
        encoded = self.encoder(x)  
        _, _, h, w = encoded.shape  # 假设 h=w=4 或 8
        
        
       
        t_emb = self.time_embedding(t)               # [B, t_emb_dim]

        
        t_proj = self.t_proj(t_emb)  #

        t_proj = t_proj.unsqueeze(-1).unsqueeze(-1)  # -> [B, base_width*4, 1, 1]
    
        
        bg_map = self.bg_weight_map_mlp(t_emb)
        bg_map = bg_map.view(-1, 1, 8, 8)          # reshape 到 [B, 1, 8, 8]
        bg_map = torch.sigmoid(bg_map)             # 映射到 (0,1)，初始值约为 0.2
        
        if mask.dim() == 3:
            tmp_mask = mask.unsqueeze(1)
        mask_down = F.interpolate(mask, size=(h, w), mode='nearest')  # [B,1,h,w]
        

        effective_mask = mask_down + (1 - mask_down) * bg_map
        encoded = encoded * effective_mask  # 对 encoder 的特征做位置级别乘法

        encoded = encoded + t_proj
        
        # ============ Decoder部分 ============
        recon = self.decoder(encoded, mask)
        return recon