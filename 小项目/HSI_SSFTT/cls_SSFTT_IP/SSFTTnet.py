import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from fusion_strategy import AxialAttentionTransformer



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    # 输入x: 形状为(batch_size, seq_length, dim)
    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        # print('qkv shape:', [t.shape for t in qkv])  # Debugging line to check shapes of q, k, v
        # qkv = [b, n, dim] -> [b, n, h * dim // h] for each of q, k, v
        # rearrange: [b, n, h * dim // h] -> [b, n, h, dim // h]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions
        #  print('q shape:', q.shape, 'k shape:', k.shape, 'v shape:', v.shape)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # 输出形状：(batch_size, heads, seq_length, seq_length)
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            # 若提供 mask（如处理填充符[PAD]），将无效位置设为负无穷
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(5, AxialAttentionTransformer(
            dim=64,                # 与conv2d_features输出通道一致
            depth=2,               # 可根据需要调整
            heads=heads,
            dim_index=1,           # 通道维
            reversible=True,
            axial_pos_emb_shape=(1, 5)  # 空间尺寸，需与输入一致
        )))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = rearrange(x, 'b n d -> b d 1 n')  # (
            # print(x.shape)  # Debugging line to check shape after attention
            x = mlp(x)  # go to MLP_Block
            x = rearrange(x, 'b d 1 n -> b n d')
        return x

NUM_CLASS = 16

class SSFTTnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(SSFTTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 融合AxialAttentionTransformer
        self.axial_attention = AxialAttentionTransformer(
            dim=64,                # 与conv2d_features输出通道一致
            depth=1,               # 可根据需要调整
            heads=heads,
            dim_index=1,           # 通道维
            reversible=True,
            axial_pos_emb_shape=(13, 13)  # 空间尺寸，需与输入一致
        )
        
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):    # shape:  (1024, 1, 30, 13, 13)
        # print("Input shape:", x.shape)
        x = self.conv3d_features(x)     # (1024, 8, 28, 11, 11)
        # print("After conv3d shape:", x.shape)
        x = rearrange(x, 'b c h w y -> b (c h) w y')    # (1024, 8*28, 11, 11)
        # print("After rearrange shape:", x.shape)
        x = self.conv2d_features(x)     # (1024, 64,9, 9)
        # print("After conv2d shape:", x.shape)
        
        # x = self.axial_attention(x)
        # print("After final rearrange shape:", x.shape)
        # print(x.shape)
        # AxialAttentionTransformer
        # y = self.axial_attention(x)  # (B, 64, 11, 11)
        # print(x.shape)
        x = rearrange(x,'b c h w -> b (h w) c') # (1024,9*9, 64)
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose (1,64,4)
        A = torch.einsum('bij,bjk->bik', x, wa)         # (1024, 9*9, 4)
        A = rearrange(A, 'b h w -> b w h')  # Transpose (1024, 4, 9*9)
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV) # (1024, 9*9, 64)
        T = torch.einsum('bij,bjk->bik', A, VV) # (1024, 4, 64)
        # print("After tokenization shape:", T.shape)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # print("After cls_token expansion shape:", cls_tokens.shape)
        # 将原始形状 (1, 1, d_model) 扩展为 (1024, 1, 64)
        x = torch.cat((cls_tokens, T), dim=1)   #
        # print("After cls_token shape:", x.shape)
        x += self.pos_embedding
        x  = self.dropout(x)

        x = self.transformer(x, mask)  # main game (batch_size, seq_length, dim)
        # y = rearrange(x, 'b n d -> b d 1 n')  # (batch, dim, 1, seq_len)
        # y = self.axial_attention(y)           # (batch, dim, 1, seq_len)
        # y = rearrange(x, 'b d 1 n -> b n d')  # (batch, seq_len, dim)
        # print(y.shape)
        # print("After transformer shape:", x.shape)  # (1024, 5, 64)
        x = self.to_cls_token(x[:, 0])  # (1024, 64)  # cls_token + axial_attention output
        # print("After to_cls_token shape:", x.shape) # (1024, 64)
        x = self.nn1(x) # (b,64)->(b,num_classes)   # (1024, 64) -> (1024, 16)
        # print("Final output shape:", x.shape)
        return x


if __name__ == '__main__':
    model = SSFTTnet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 13, 13)
    y = model(input)
    print(y.size())
