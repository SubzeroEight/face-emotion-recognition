import torch
import torch.nn as nn
from einops import rearrange
from collections import OrderedDict

__all__ = ["coatnet_1"]


def conv_3x3_bn(in_c, out_c, image_size, downsample=False):
    stride = 2 if downsample else 1
    layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU()
    )
    return layer


class SE(nn.Module):
    def __init__(self, in_c, out_c, expansion=0.25):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 将每个样本的每个通道压缩为一个值
        self.fc = nn.Sequential(
            nn.Linear(out_c, int(in_c * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(in_c * expansion), out_c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # view(b, c)：将形状从 (b, c, 1, 1) 展平为 (b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):
    def __init__(self, in_c, out_c, image_size, downsample=False, expansion=4):
        super(MBConv, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        hidden_dim = int(in_c * expansion)

        if self.downsample:
            # 只有第一层的时候，进行下采样
            # self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

        layers = OrderedDict()
        # expand
        expand_conv = nn.Sequential(
            nn.Conv2d(in_c, hidden_dim, 1, stride, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        layers.update({"expand_conv": expand_conv})

        # Depwise Conv
        dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        layers.update({"dw_conv": dw_conv})

        # se
        layers.update({"se": SE(in_c, hidden_dim)})

        # project
        pro_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c)
        )
        layers.update({"pro_conv": pro_conv})
        self.block = nn.Sequential(layers)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.block(x)
        else:
            return x + self.block(x)


class Attention(nn.Module):  # 结合了多头自注意力与相对位置偏置。设计上参考了 Swin Transformer 的思路，同时适配了标准Transformer的结构
    def __init__(self, in_c, out_c, image_size, heads=8, dim_head=32, dropout=0.):  # 256, 512, 28x28, 8, 32, 0.
        super(Attention, self).__init__()
        inner_dim = dim_head * heads  # 32*8
        project_out = not (heads == 1 and dim_head == in_c)  # True

        self.ih, self.iw = image_size if len(image_size) == 2 else (image_size, image_size)

        self.heads = heads
        self.scale = dim_head ** -0.5  # 1/根号(32)

        # parameter table of relative position bias  # [(2*ih-1)*(2*iw-1), heads] 这里面才是实际需要训练的参数量
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads)  # (2*ih-1)*(2*iw-1)行，heads列，27*27行，8列
        )

        # 生成坐标
        coords = torch.meshgrid(torch.arange(self.ih), torch.arange(self.iw), indexing='ij')
        # 得到两个坐标矩阵 row_coords，col_coords。  2个（self.ih行，self.iw列）
        coords = torch.flatten(torch.stack(coords), 1)
        """
        shape：2个（17，17）-> (2,17,17)->（2，17*17）
        这个 17*17 是用来装下坐标差的
        """
        # 将两个坐标矩阵拼接，生成一个网格坐标，表示图像每个像素的位置。

        # 生成相对坐标
        relative_coords = coords[:, :, None] - coords[:, None, :]  # (2, 289, 1) - (2, 1, 289) → (2, 289, 289)
        """   ！！！！  (2, 289, 289)
        relative_coords[0, i, j] = row_i - row_j   (i,j点的行坐标差)
        relative_coords[1, i, j] = col_i - col_j   (i,j点的列坐标差)
        """

        # 平移到左下角，全部变成正数
        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        # 将行和列的相对坐标差从 [-ih+1, ih-1] 和 [-iw+1, iw-1] 平移至非负范围 [0, 2*ih-2] 和 [0, 2*iw-2]
        relative_coords[0] *= 2 * self.iw - 1  # 避免相加后出现相同的。对行坐标按 2*self.iw-1 拉伸

        # 生成相对位置坐标的索引
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')  # (289, 289，2)
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        # sum(-1): 在最后一个维度上求和。  行差 + 列差。  从 (289, 289, 2) 变为 (289, 289)（所有人对所有人的行插值与列插值之和）
        # flatten()：按行展开。                       从 (289, 289) 变为 (289 * 289 = 83521)
        # unsqueeze(1)：在第1维度增加一个维度。         从 (83521,) 变为 (83521, 1)   83521=17^4

        """
        PyTorch中定义模型时，self.register_buffer('name', Tensor)，
        该方法的作用是定义一组参数，该组参数的特别之处在于：
        模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），
        但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        """
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(in_c, inner_dim * 3, bias=False)  # 256 --> (32*8)*3
        self.proj = nn.Sequential(
            nn.Linear(inner_dim, out_c),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x.shape= [b, n, c] = [b, (h w), c] = [b, (14 14), 256]
        # [q,k,v]  q:[b, n, new_c] = [b, (h w), new_c] = [b, (14 14), 32*8]
        qkv = self.qkv(x).chunk(3, dim=-1)
        # q,k,v:[b, h, n, d] = [b, self.heads, (h*w), dim_head] = [b, 8, (14*14), 32]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # 对qkv元组中的每个张量（q、k、v）应用相同的 rearrange 变换操作。指定h。 （batch_size, 8, (14*14), 32）

        # [batch_size, num_heads, ih*iw, ih*iw]
        # 时间复杂度：O(图片边长的平方)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        # repeat(1, self.heads): 在第1维重复 heads 次。形状从 (83521, 1) 到 (83521, 8)
        # gather(0, ...)：在 self.relative_bias_table 的第0维执行索引查找
        """
        # gather 伪代码
        for i in range(83521):
            for j in range(heads):
                relative_bias[i, j] = relative_bias_table[relative_index[i, j], j]
        """

        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw
        )
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class Transformer(nn.Module):
    def __init__(self, in_c, out_c, image_size, heads=8, dim_head=32, downsample=False, dropout=0., expansion=4,
                 norm_layer=nn.LayerNorm):
        super(Transformer, self).__init__()
        self.downsample = downsample
        hidden_dim = int(in_c * expansion)
        self.ih, self.iw = image_size

        if self.downsample:
            # 第一层进行下采样
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

        self.attn = Attention(in_c, out_c, image_size, heads, dim_head, dropout)  # 256, 512, 28x28, 8, 32, 0.
        self.ffn = FFN(out_c, hidden_dim)
        self.norm1 = norm_layer(in_c)
        self.norm2 = norm_layer(out_c)

    def forward(self, x):
        x1 = self.pool1(x) if self.downsample else x
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x1 = self.attn(self.norm1(x1))
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=self.ih, w=self.iw)
        x2 = self.proj((self.pool2(x))) if self.downsample else x

        x3 = x1 + x2
        x4 = rearrange(x3, 'b c h w -> b (h w) c')
        x4 = self.ffn(self.norm2(x4))
        x4 = rearrange(x4, 'b (h w) c -> b c h w', h=self.ih, w=self.iw)
        out = x3 + x4
        return out


class CoatNet(nn.Module):
    def __init__(self,
                 image_size=(224, 224),
                 in_channels: int = 3,
                 num_blocks: list = [2, 2, 3, 5, 2],  # L  [2, 2, 6, 14, 2]
                 channels: list = [64, 96, 192, 384, 768],  # D  [96, 128, 256, 512, 1024]
                 num_classes: int = 1000,
                 block_types=['C', 'C', 'T', 'T']):
        super(CoatNet, self).__init__()

        assert len(image_size) == 2, "image size must be: {H,W}"
        assert len(channels) == 5
        assert len(block_types) == 4

        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2)
            # conv_3x3_bn，3，96，2，112x112
        )
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4)
            # MBConv，96，128，2，56x56
        )
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8)
            # MBConv，128，256，6，28x28
        )
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16)
            # Transformer，256，512，6，14x14
        )
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32)
            # Transformer，512，1024，6，7x7
        )

        # 总共下采样32倍 2^5=32
        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)  # 1024 --> 7

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)  # 将池化后的特征图展平，从 (b, c, 1, 1) 变为 (b, c)
        x = self.fc(x)
        return x

    def _make_layer(self, block, in_c, out_c, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(in_c, out_c, image_size, downsample=True))
            else:
                layers.append(block(out_c, out_c, image_size, downsample=False))
        return nn.Sequential(*layers)


def coatnet_1(num_classes=7):
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [96, 128, 256, 512, 1024]   # D
    return CoatNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.randn(1, 3, 224, 224).to(device)
    model = coatnet_1().to(device)
    out = model(img)
    summary(model, input_size=(3, 224, 224))
    print(out.shape, count_parameters(model))

    # net = coatnet_1()
    # out = net(img)
    # print(out.shape, count_parameters(net))
