from torch import nn
from ts_model.lib import (
    Interaction,
    ReadOut,
    OneHotEncoder,
    compute_relative_vectors,
    compute_spherical_harmonics,
    sinusoidal_embedding,
    normalize
)


class TSGenModel(nn.Module):
    def __init__(
        self,
        elem_embeddim=256,
        coord_embeddim_d=256,
        coord_embeddim_r=256,
    ):
        self.elem_embeddim = elem_embeddim ## C
        self.coord_embeddim_d = coord_embeddim_d ## D1
        self.coord_embeddim_r = coord_embeddim_r ## D2
        self.one_hot = OneHotEncoder(self.elem_embeddim)

        self.z_fnn = nn.Linear(self.elem_embeddim, self.elem_embeddim)
        self.r_fnn = nn.Linear(self.elem_embeddim_d, self.elem_embeddim_d)
        self.interaction = Interaction()
        self.readout = ReadOut()

    def forward(self, z_i, rij, cij=None):
        """
        原子数をNとする
        Args:
            z_i (torch.Tensor): (N,)
            r_ij (torch.Tensor): (N, N)
            c_ij (torch.Tensor): (N, N)

        """
        N = z_i.shape[0]
        C = self.elem_embeddim
        z_embed = self.one_hot(z_i) ## (N, C)
        rel_dist, rel_vec = compute_relative_vectors(rij) 
        d_embed = sinusoidal_embedding(rel_dist, dim=self.coord_embeddim_d) ## (N, N, D1)
        r_embed = compute_spherical_harmonics(rel_vec, l_max=self.coord_embeddim_r) ## (N, N, D2)

        z_embed = self.z_fnn(z_embed)
        n = normalize(z_embed) # (N, C)

        s = self.r_fnn(d_embed.reshape(N**2, C)).reshape(N, N, C) # (N, N, C)
        v = r_embed

        o = self.interaction(n, s, v)
        y_fin, y_std, y_tan, y_prp = self.readout(o)
        return y_fin, y_std, y_tan, y_prp
