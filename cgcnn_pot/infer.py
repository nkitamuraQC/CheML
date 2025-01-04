from __future__ import print_function, division

import csv
import functools
import json
import os
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from cgcnn_pot.data import *
from ase import Atoms
from ase.io import write
import pathlib
from cgcnn_pot.model import CrystalGraphConvNet
from torch.autograd import Variable

class CrysGNNPot(torch.nn.Module):
    def __init__(self, checkpoint_file_path="", atom_fea_len=64, nbr_fea_len=128):

        crysgnn_model = CrystalGraphConvNet(orig_atom_fea_len=64, nbr_fea_len=nbr_fea_len, atom_fea_len=atom_fea_len, n_conv=5)
        checkpoint = torch.load(checkpoint_file_path)
        crysgnn_model.load_state_dict(checkpoint['state_dict'])

    def forward(self, atoms, cif_id=0):
        self.atoms = atoms
        self.root_dir = pathlib.Path("./root")
        self.cif = str(self.root_dir.joinpath(f"{cif_id}.cif"))
        write(self.cif, self.atoms)

        st = CIFData(str(self.root_dir))
        info, _ = st[0]

        input_var = (Variable(info[0]),
                     Variable(info[1]),
                     info[2],
                     info[3])
        
        output = self.model(*input_var)
        return output