import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.models.layers import trunc_normal_
import math
from einops import rearrange, repeat



import numbers

from einops import rearrange

class BasicBlock(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, dim, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=1, bias=False, LayerNorm_type='BiasFree'):
        super(RestormerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    

class SpectralBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=1, bias=False, LayerNorm_type='BiasFree'):
        super(SpectralBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, q,k,v):
        x = self.attn(q,k,v) + q
        x = x + self.ffn(self.norm2(x))
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by num_heads"
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        res = query
        batch_size, channels, height, width = query.shape
        query = query.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height*width, channels)
        key = key.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height*width, channels)
        value = value.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height*width, channels)
        values = self.values(value)
        keys = self.keys(key)
        queries = self.queries(query)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (batch, num_heads, query_len, key_len)
        # print(energy.shape) 
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.fc_out(out)
        out = out.view(batch_size, channels, height, width)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 2 // reduction, self.dim * 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg_v = self.avg_pool(x).view(B, self.dim * 2) #B  2C
        max_v = self.max_pool(x).view(B, self.dim * 2)
        avg_se = self.mlp(avg_v).view(B, self.dim * 2, 1)
        max_se = self.mlp(max_v).view(B, self.dim * 2, 1)
        Stat_out = self.sigmoid(avg_se+max_se).view(B, self.dim * 2, 1)
        channel_weights = Stat_out.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SpatialAttention, self).__init__()    
        self.max_mlp = nn.Sequential(
            nn.Conv2d(2,1,3,1,1),
            nn.Sigmoid()    
        )
        self.mean_mlp = nn.Sequential(
            nn.Conv2d(2,1,3,1,1),
            nn.Sigmoid()    
        )
        self.offsef_01 = nn.Parameter(torch.zeros(1))
        self.offsef_02 = nn.Parameter(torch.zeros(1))
        
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x1_mean_out = torch.mean(x1, dim=1, keepdim=True) #B  1  H  W
        x1_max_out, _ = torch.max(x1, dim=1, keepdim=True)  #B  1  H  W
        x2_mean_out = torch.mean(x2, dim=1, keepdim=True) #B  1  H  W
        x2_max_out, _ = torch.max(x2, dim=1, keepdim=True)  #B  1  H  W    
        mean_feat = torch.cat((x1_mean_out, x2_mean_out), dim=1)
        max_feat = torch.cat((x1_max_out, x2_max_out), dim=1)
        mean_spatial_weights = self.mean_mlp(mean_feat).reshape(B, 1, H, W)
        max_spatial_weights = self.max_mlp(max_feat).reshape(B, 1, H, W)
        spatial_weights = self.offsef_01 * mean_spatial_weights + self.offsef_02 * max_spatial_weights
        return spatial_weights

def retain_top_k_values(tensor, k):
    flattened_tensor = tensor.view(tensor.shape[0], tensor.shape[1], -1)
    top_k_values, indices = torch.topk(flattened_tensor, k, dim=-1)
    retained_tensor = torch.zeros_like(flattened_tensor)
    retained_tensor.scatter_(-1, indices, top_k_values)
    return retained_tensor.view_as(tensor)


    

class DSpeFB(nn.Module):
    # (a) Deformable Spectral Feature Block (DSpeFB) 
    def __init__(self,dim):
        super(DSpeFB, self).__init__()
        self.conv_spectral = nn.Conv2d(dim,dim,1,1)
        self.conv1 = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.conv2 = nn.Conv2d(dim,dim,1,1)
        
    def forward(self, x):
        res = x
        score = F.adaptive_max_pool2d(x,(1,1))
        score = F.relu6(self.conv_spectral(score))
        x = F.relu(self.conv1(x))
        x = x + torch.clip(score, -1,1)
        x = F.relu(self.conv2(x))
        return x + res

class DSpaFB(nn.Module):
    def __init__(self,dim):
        super(DSpaFB, self).__init__()
        self.conv = nn.Conv2d(dim,1,3,1,1,)
        self.conv1 = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.conv2 = nn.Conv2d(dim,dim,1,1)
        
    def forward(self, x):
        res = x
        spatial_score = self.conv(x)
        x =self.conv1(x)
        x = x + torch.clip(spatial_score, -1,1)
        x = self.conv2(x)
        return x + res


class SSFAB(nn.Module):
    def __init__(self, dim, reduction=4):
        super(SSFAB, self).__init__()
        self.dim = dim
        # self.block = BasicBlock(dim) 
        
        self.ca_gate = ChannelAttention(self.dim) 
        self.sa_gate = SpatialAttention(reduction=reduction)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, 1),  # 降低特征维度
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 1, 1),  # 恢复特征维度
            nn.Sigmoid()  # 使用 Sigmoid 激活函数来生成门控权重
        )
        self.convs = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, 1, 1)  # 对拼接后的特征进行卷积降维
        )
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1)  # 最终卷积层，进一步优化输出
        )
    
    def forward(self, in_x, hsi_feat, msi_feat):
        # x = self.block(in_x+ hsi_feat+ msi_feat)
        B1, C1, H1, W1 = hsi_feat.shape
        B2, C2, H2, W2 = msi_feat.shape
        assert B1 == B2 and C1 == C2 and H1 == H2 and W1 == W2, "x1 and x2 should have the same dimensions"
        gated_weight = self.gate(in_x)  # B C H W
        ca_out = self.ca_gate(in_x, msi_feat)  # 2 B C 1 1
        sa_out = self.sa_gate(hsi_feat, in_x)  # 2 B 1 H W
        
        
        mixatt_out = ca_out + sa_out  # 2 B C H W
        Gated_attention_x1 = gated_weight * mixatt_out[0]
        
        
        Gated_attention_x2 = (1 - gated_weight) * mixatt_out[1]
        
        
        out_x1 = hsi_feat + Gated_attention_x2 * msi_feat  # B C H W
        out_x2 = msi_feat + Gated_attention_x1 * hsi_feat  # B C H W
        x = torch.cat([out_x1, out_x2], dim=1)
        x = self.convs(x)
        x = self.tail(x)
        return x
    
    
class FuseNet(nn.Module):
    "Spatial-Spectral Rectification Module"
    def __init__(self,dim):
        super(FuseNet, self).__init__()
        self.dim = dim
        # self.block = BasicBlock(dim)
        self.spatial_conv1 = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.spatial_conv2 = nn.Conv2d(dim,dim,1,1)
        self.spatial_mlp = nn.Conv2d(1,dim,1,1)
        
        self.spectral_conv1 = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.spectral_conv2 = nn.Conv2d(dim,dim,1,1)
        self.spectral_mlp = nn.Conv2d(dim,dim,1,1)
        
        self.spectral_attn = SpectralBlock(dim, 4)
        self.spatial_attn = RestormerBlock(dim, 4)
        
    def forward(self, hsi_f0, msi_f0, hsi_fi, msi_fi):
        # x = self.block(hsi_f0+msi_f0 + hsi_fi +msi_fi)  
        spectral_score = F.adaptive_max_pool2d(hsi_fi,(1,1))
        spectral_score = F.relu6(self.spectral_mlp(spectral_score))
        
        spatial_score =  torch.mean(msi_fi, dim=1, keepdim=True)
        spatial_score = F.relu6(self.spatial_mlp(spatial_score))
        
        hsi_f0 = self.spectral_conv1(hsi_f0)
        hsi_f0 = hsi_f0 + torch.clip(spectral_score, -1,1)
        hsi_f0 = self.spectral_conv2(hsi_f0)
        
        msi_f0 = self.spatial_conv1(msi_f0)
        msi_f0 = msi_f0 + torch.clip(spatial_score, -1,1)
        msi_f0 = self.spatial_conv2(msi_f0)
        
        
        
        x1 = self.spectral_attn(msi_f0, hsi_f0, hsi_f0)
        x2 = self.spatial_attn(msi_f0 + hsi_f0)
        x = x1 + x2
        return x
    
    
class SSAANet(nn.Module):
    def __init__(self, hsi_bands, msi_bands, dim=64,upscale=4):
        super(SSAANet, self).__init__()
        # hsi_bands = args.hsi_bands
        # msi_bands = args.msi_bands
        self.scale = upscale
        self.hsi_head = nn.Sequential(
            nn.Conv2d(hsi_bands,dim,1,1),

        )
        self.msi_head = nn.Sequential(
            nn.Conv2d(msi_bands,dim,1,1),

            )
        self.l = 12
        self.spectral_encoders = nn.ModuleList()
        self.spatial_encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(  self.l):
            self.spectral_encoders.append(DSpeFB(dim))
            self.spatial_encoders.append(DSpaFB(dim))
            self.decoders.append(SSFAB(dim))
        self.spectral_encoder_01 = DSpeFB(dim)
        self.spectral_encoder_02 = DSpeFB(dim)
        self.spectral_encoder_03 = DSpeFB(dim)
        self.spectral_encoder_04 = DSpeFB(dim)
        # self.spectral_encoder_05 = DSpeFB(dim)  
        
        self.spatial_encoder_01 = DSpaFB(dim)
        self.spatial_encoder_02 = DSpaFB(dim)
        self.spatial_encoder_03 = DSpaFB(dim)
        self.spatial_encoder_04 = DSpaFB(dim)
        self.fus_net = FuseNet(dim)
        
        self.decoder_01 = SSFAB(dim)
        self.decoder_02 = SSFAB(dim)
        self.decoder_03 = SSFAB(dim)
        self.decoder_04 = SSFAB(dim)
        
        self.tail = nn.Sequential(
            nn.Conv2d(dim,dim,3,1,1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(dim,hsi_bands,3,1,1)
        )
        
        
    def forward(self, hsi, msi):
        _,_,H,W = msi.shape
        up_hsi = F.interpolate(hsi,scale_factor=self.scale,mode='bilinear',align_corners=True)
        hsi_feat = self.hsi_head(up_hsi)
        msi_feat = self.msi_head(msi)
        hsi_feats = []
        msi_feats = []
        decoder_feats = []
        for i in range( self.l):
            hsi_feats.append(self.spectral_encoders[i](hsi_feat))
            msi_feats.append(self.spatial_encoders[i](msi_feat))
        x = self.fus_net(hsi_feat,msi_feat,hsi_feats[-1],msi_feats[-1])
        for i in range(  self.l):
            x = self.decoders[i](x,hsi_feats[-i],msi_feats[-i])
        hsi_feat_e1 = self.spectral_encoder_01(hsi_feat)
        hsi_feat_e2 = self.spectral_encoder_02(hsi_feat_e1)
        hsi_feat_e3 = self.spectral_encoder_03(hsi_feat_e2)
        hsi_feat_e4 = self.spectral_encoder_04(hsi_feat_e3)    
        # hsi_feat_e5 = self.spectral_encoder_05(hsi_feat_e4)   
          
        msi_feat_e1 = self.spatial_encoder_01(msi_feat)
        msi_feat_e2 = self.spatial_encoder_02(msi_feat_e1)
        msi_feat_e3 = self.spatial_encoder_03(msi_feat_e2)
        msi_feat_e4 = self.spatial_encoder_04(msi_feat_e3)
        # msi_feat_e5 = self.spatial_encoder_05(msi_feat_e4) 
        
        
        
        
        x = self.decoder_01(x,hsi_feat_e4,msi_feat_e4)
        x = self.decoder_02(x,hsi_feat_e3,msi_feat_e3)
        x = self.decoder_03(x,hsi_feat_e2,msi_feat_e2)
        x = self.decoder_04(x,hsi_feat_e1,msi_feat_e1)
        # x = self.decoder_05(x,hsi_feat_e1,msi_feat_e1)
        
        x = self.tail(x) + up_hsi
        return x