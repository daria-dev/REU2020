# Handy loss functions

import torch

def L2Reg (y, y_hat, l, weights):
    criterion = torch.nn.MSELoss()
    loss = criterion(y_hat, y)
    reg = torch.tensor(0.0)

    for w in weights:
        reg += w.norm()
    
    loss += l*reg

    return loss

def L1Reg (y, y_hat, l, weights):
    criterion = torch.nn.MSELoss()
    loss = criterion(y_hat, y)
    reg = torch.tensor(0.0)

    for w in weights:
        reg += w.abs()
    
    loss += l*reg

    return loss