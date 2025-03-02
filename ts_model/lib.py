from torch import nn
import torch
from e3nn.o3 import spherical_harmonics, FullyConnectedTensorProduct
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v):
    """
    Scaled Dot-Product Attention 計算
    :param q: クエリテンソル (N, N, C)
    :param k: キーテンソル (N, N, C)
    :param v: バリューテンソル (N, N, C)
    :return: アテンション結果

    Examples:
        # 例: N=4, C=3 の場合
        N = 4
        C = 3
        q = torch.randn(N, N, C)  # クエリテンソル
        k = torch.randn(N, N, C)  # キーテンソル
        v = torch.randn(N, N, C)  # バリューテンソル
        
        # アテンション計算
        output = scaled_dot_product_attention(q, k, v)
        
        # 結果
        print("Output Shape:", output.shape)  # 出力の形状 (N, N, C)
        print(output)
    """
    N, _, C = q.shape
    
    # 1. クエリとキーの内積を計算 (QK^T)
    scores = torch.einsum("nmc,nmc->nm", q, k)  
    
    # 2. スケーリング
    scores = scores / torch.sqrt(torch.tensor(C, dtype=torch.float32))  # スケーリング
    
    # 3. ソフトマックスで重みを計算
    attention_weights = F.softmax(scores, dim=-1)  
    
    # 4. 重みをバリューに掛けて最終結果を計算
    output = torch.einsum("ni,nic->nc", attention_weights, v)  
    
    return output


def normalize(n_i):
    """
    Nは原子数, Cは埋め込み次元
    Args:
       n_i (torch.tensor): (N, C)
    """
    norms = torch.norm(n_i, dim=1, keepdim=True)
    normalized_n_i = n_i / norms
    return normalized_n_i


def compute_spherical_harmonics(norm_rel_vec, l_max):
    """
    Spherical Harmonics を計算する

    Args:
        norm_rel_vec (torch.Tensor): (N, N, 3) の規格化相対ベクトル
        l_max (int): 球面調和関数の最大次数

    Returns:
        torch.Tensor: (N, N, L) の球面調和埋め込み
    """
    sh = spherical_harmonics(l_max, norm_rel_vec, normalize=True)  # (N, N, L)
    return sh

def sinusoidal_embedding(x, dim):
    """
    Sinusoidal Embedding を計算する

    Args:
        x (torch.Tensor): (N, N) の相対距離行列
        dim (int): 埋め込み次元（偶数）

    Returns:
        torch.Tensor: (N, N, dim) の埋め込みベクトル
    """
    assert dim % 2 == 0, "埋め込み次元は偶数にしてください"

    # 周波数スケール
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))  # (dim/2,)

    # Sin/Cos の計算
    x_expanded = x.unsqueeze(-1)  # (N, N, 1)
    pe = torch.zeros(*x.shape, dim)  # (N, N, dim)
    pe[:, :, 0::2] = torch.sin(x_expanded * div_term)  # 偶数次元
    pe[:, :, 1::2] = torch.cos(x_expanded * div_term)  # 奇数次元

    return pe


def compute_relative_vectors(coords):
    """
    座標集合から相対距離と規格化された相対ベクトルを計算する

    Args:
        coords (torch.Tensor): (N, 3) の座標テンソル

    Returns:
        rel_dist (torch.Tensor): (N, N) の相対距離行列
        norm_rel_vec (torch.Tensor): (N, N, 3) の規格化された相対ベクトル
    
    Examples:
        coords = torch.tensor([[0.0, 0.0, 0.0], 
                               [1.0, 0.0, 0.0], 
                               [0.0, 1.0, 0.0]])
        
        rel_dist, norm_rel_vec = compute_relative_vectors(coords)
        
        print("相対距離行列:")
        print(rel_dist)
        
        print("\n規格化された相対ベクトル:")
        print(norm_rel_vec)

    """
    # すべてのペアの座標の差分を計算（ブロードキャストを利用）
    rel_vec = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 3)

    # 各ベクトルのノルム（距離）を計算
    rel_dist = torch.norm(rel_vec, dim=-1)  # (N, N)

    # 規格化: ゼロ除算を防ぐために小さな値を足す
    norm_rel_vec = rel_vec / (rel_dist.unsqueeze(-1) + 1e-8)  # (N, N, 3)

    return rel_dist, norm_rel_vec


def concat(n_i, n_j, s_ij):
    """
    原子数はN
    Args:
        n_i (torch.Tensor): (N, C) 
        n_j (torch.Tensor): (N, C) 
        s_ij (torch.Tensor): (N, N, C) 
    """
    e_ij = torch.einsum("nc,mc,nmc->nmc", n_i, n_j, s_ij)
    return e_ij
    

class Interaction(nn.Module):
    def __init__(self):
        self.edge_feat_q = EdgeFeat()
        self.edge_feat_k = EdgeFeat()
        self.edge_feat_v = EdgeFeat()

    def forward(self, n, s, v):
        q = self.edge_feat_q(n, s, v)
        k = self.edge_feat_k(n, s, v)
        v = self.edge_feat_v(n, s, v)
        x = scaled_dot_product_attention(q, k, v)
        x += n
        o = normalize(x)
        return o



class EdgeFeat(nn.Module):
    def __init__(self, dim=256):
        self.edge_index = EdgeIndex()
        self.fnn = nn.Linear(dim, dim)
        self.o3tr = None

    def forward(self, n, s, v):
        n1, n2 = self.edge_index(n)
        N, C = n1.shape
        e = concat(n1, n2, s)
        e = e.reshape((N*N, C))
        e = self.fnn(e).reshape((N, N, C))
        e *= v
        return e
    

class EdgeIndex(nn.Module):
    def __init__(self):
        pass

    def forward(self, n):
        return n, n


class ReadOut(nn.Module):
    def __init__(self):
        pass

    def forward(self, o):
        """
        Args:
            o (torch.tensor): (N, C)
        """


class OneHotEncoder(nn.Module):
    def __init__(self, num_classes):
        super(OneHotEncoder, self).__init__()
        self.num_classes = num_classes

    def forward(self, labels):
        return torch.nn.functional.one_hot(labels, num_classes=self.num_classes)