# -*- coding: utf-8 -*-
import torch.nn as nn

# Impairment learning parameters
class PertNet(nn.Module):
    def __init__(self,init_vector):
        super(PertNet, self).__init__()
        #Define vector to optimize
        self.pos = nn.Parameter(init_vector)

# Dictionary learning parameters
class MatrixNet(nn.Module):
    def __init__(self,init_matrix):
        super(MatrixNet, self).__init__()
        #Define matrix to optimize
        self.A = nn.Parameter(init_matrix)