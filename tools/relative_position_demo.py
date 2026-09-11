import torch
import torch.nn as nn
from einops import rearrange

print(torch.__version__)               # 应输出 2.3.0
print(torch.cuda.is_available())       # 应输出 True
print(torch.version.cuda)


ih, iw = 2, 2

# 生成坐标
coords = torch.meshgrid(torch.arange(ih), torch.arange(iw), indexing='ij')  # 2个（self.ih行，self.iw列）
coords = torch.flatten(torch.stack(coords), 1)  # shape：2个（7，7）-> (2,7,7)->（2，7*7）
print("\ncoords:\n", coords)
"""
coords:
 tensor([[0, 0, 1, 1],  row_coords
        [0, 1, 0, 1]])  col_coords
"""

# 生成相对坐标
relative_coords = coords[:, :, None] - coords[:, None, :]  # (2, 49, 1) - (2, 1, 49) → (2, 49, 49)
print("\ncoords[:, :, None]:\n", coords[:, :, None])
"""
coords[:, :, None]:
 tensor([[[0],
         [0],
         [1],
         [1]],

        [[0],
         [1],
         [0],
         [1]]])
"""
print("\ncoords[:, None, :]\n", coords[:, None, :])
"""
coords[:, None, :]
 tensor([[[0, 0, 1, 1]],

        [[0, 1, 0, 1]]])
"""
print("\nrelative_coords:\n", relative_coords)
"""
relative_coords:
 tensor([[[ 0,  0, -1, -1],
         [ 0,  0, -1, -1],
         [ 1,  1,  0,  0],
         [ 1,  1,  0,  0]],

        [[ 0, -1,  0, -1],
         [ 1,  0,  1,  0],
         [ 0, -1,  0, -1],
         [ 1,  0,  1,  0]]])
"""

relative_coords[0] += ih - 1
relative_coords[1] += iw - 1
relative_coords[0] *= 2 * iw - 1  # 3
print("\n修改后的relative_coords:\n", relative_coords)
"""
修改后的relative_coords:
 tensor([[[3, 3, 0, 0],
         [3, 3, 0, 0],
         [6, 6, 3, 3],
         [6, 6, 3, 3]],

        [[1, 0, 1, 0],
         [2, 1, 2, 1],
         [1, 0, 1, 0],
         [2, 1, 2, 1]]])
"""

# 生成相对位置坐标的索引
relative_coords = rearrange(relative_coords, 'c h w -> h w c')  # (289, 289，2)
relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
print(print("\nrelative_index:\n", relative_index))
"""
relative_index:
 tensor([[4],
        [3],
        [1],
        [0],
        [5],
        [4],
        [2],
        [1],
        [7],
        [6],
        [4],
        [3],
        [8],
        [7],
        [5],
        [4]])
None
"""