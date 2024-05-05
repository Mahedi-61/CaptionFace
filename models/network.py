import sys
import iresnet
import os
import torch.nn.functional as F
import torch.nn as nn
import torch


def load_features(arch):
    if arch == 'iresnet34':
        features = iresnet.iresnet34(pretrained=False, progress=True)

    elif arch == 'iresnet18':
        features = iresnet.iresnet18(pretrained=False, progress=True)

    elif arch == 'iresnet50':
        features = iresnet.iresnet50(pretrained=False, progress=True)

    elif arch == 'iresnet100':
        features = iresnet.iresnet101(pretrained=False, progress=True)

    else:
        raise ValueError()
    return features


class NetworkBuilder(nn.Module):
    def __init__(self, arch):
        super(NetworkBuilder, self).__init__()
        self.features = load_features(arch)

    def forward(self, input):
        # add Fp, a pose feature
        gl_featus, lc_feats = self.features(input)
        return gl_featus, lc_feats 


if __name__ == "__main__":
    arch = "iresnet50"
    resnet = iresnet.iresnet101(pretrained=False, progress=True)

    x = torch.randn(32, 3, 112, 112).cuda()
    #resnet = torch.nn.DataParallel(resnet, device_ids=[0]).cuda()
    #print(resnet.state_dict().keys())

    mag_dict = torch.load("magface_ir101_ms1mv2.pth")
   
    del mag_dict["state_dict"]["fc.weight"]

    model = {}
    for k, v in mag_dict["state_dict"].items():
        model[k[16:]] = v

    del mag_dict
    resnet.load_state_dict(model)
