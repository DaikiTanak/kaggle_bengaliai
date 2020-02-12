import numpy as np

import torch
import torch.nn as nn


# cut mix regularization

class CutMixLoss(nn.Module):
    # https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    def __init__(self, device="cpu"):

        super().__init__()

        self.device = device
        pass


    def forward(self, batch_X, label1, label2, label3, beta=1):
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(batch_X.size()[0]).to(self.device)
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(batch_X.size(), lam)
        batch_X[:, :, bbx1:bbx2, bby1:bby2] = batch_X[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_X.size()[-1] * batch_X.size()[-2]))
        # compute output
        out1, out2, out3 = model(batch_X)

        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

        return loss

    


def test():
    batch_x = torch.rand(10,3,128,128)
    batch_y = torch.rand(10,)

    cutmix = CutMixLoss()

    print(cutmix(batch_x, batch_y))


if __name__ == "__main__":
    test()
