import copy

import torch
import torch.nn as nn


class lora_Conv2D(nn.Module):
    def __init__(self, r, alpha, layer):
        super().__init__()
        self.W = layer
        if self.W.bias is not None:
            self.bias = copy.deepcopy(self.W.bias).reshape(1, -1, 1, 1)
        else:
            self.bias = torch.zeros(1, self.W.out_channels, 1, 1, device=layer.weight.device)
        # self.bias.requires_grad = False
        self.W.bias = None
        # for p in self.W.parameters():
        #     p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.A = nn.Conv2d(layer.in_channels, r, kernel_size=layer.kernel_size, padding=layer.padding,
                           stride=layer.stride, bias=False).to(layer.weight.device)
        # for p in self.A.parameters():
        #     p.requires_grad = True
        #     p.data.requires_grad = True
        self.B = nn.Conv2d(r, out_channels=layer.out_channels, kernel_size=1,
                           padding=0,
                           stride=1, bias=False).to(layer.weight.device)
        # for p in self.B.parameters():
        #     p.requires_grad = True
        #     p.data.requires_grad = True
        self.requires_grad_(True)
        for p in self.W.parameters():
            p.requires_grad = False
        self.bias.requires_grad = False
        self.reset()

    def reset(self):
        nn.init.normal_(self.B.weight, std=1 / self.r)
        nn.init.zeros_(self.A.weight)

    def forward(self, x):
        # with torch.no_grad():
        org = self.W(x)
        return org + (self.alpha / self.r) * self.B(self.A(x)) + self.bias


class lora_Conv1D(nn.Module):
    def __init__(self, r, alpha, layer):
        super().__init__()
        self.W = layer
        if self.W.bias is not None:
            self.bias = copy.deepcopy(self.W.bias).reshape(1, -1, 1)
        else:
            self.bias = torch.zeros(1, self.W.out_channels, 1, device=layer.weight.device)
        # self.bias.requires_grad = False
        self.W.bias = None
        # for p in self.W.parameters():
        #     p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.A = nn.Conv1d(layer.in_channels, r, kernel_size=layer.kernel_size, padding=layer.padding,
                           stride=layer.stride, bias=False).to(layer.weight.device)
        # for p in self.A.parameters():
        #     p.requires_grad = True
        #     p.data.requires_grad = True
        self.B = nn.Conv1d(r, out_channels=layer.out_channels, kernel_size=1,
                           padding=0,
                           stride=1, bias=False).to(layer.weight.device)
        # for p in self.B.parameters():
        #     p.requires_grad = True
        #     p.data.requires_grad = True
        self.requires_grad_(True)
        for p in self.W.parameters():
            p.requires_grad = False
        self.bias.requires_grad = False
        self.reset()

    def reset(self):
        nn.init.normal_(self.B.weight, std=1 / self.r)
        nn.init.zeros_(self.A.weight)

    def forward(self, x):
        org = self.W(x)
        return org + (self.alpha / self.r) * self.B(self.A(x)) + self.bias

def inject_lora(model, r, alpha, conv=True, linear=False, attention=False):
    model_ = model#copy.deepcopy(model)
    if conv:
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                setattr(model_.get_submodule('.'.join(name.split('.')[:-1])), name.split('.')[-1], lora_Conv2D(layer=layer, r=r, alpha=alpha))

    if attention:
        for name, layer in model.named_modules():
            if 'attention' in name.lower() or 'qkv' in name.lower():
                if isinstance(layer, torch.nn.Conv1d):
                    setattr(model_.get_submodule('.'.join(name.split('.')[:-1])), name.split('.')[-1], lora_Conv1D(layer=layer, r=r, alpha=alpha))

    if linear:
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Linear):
                setattr(model_, name, lora_Linear(layer=layer, r=r))

    # del model
    # torch.cuda.empty_cache()
    model = model_
    return model


def reset_lora(model):
    for name, layer in model.named_modules():
        if isinstance(layer, lora_Conv2D):
            # model.get_submodule('.'.join(name.split('.')[:-1]))[-1].reset()
            model.get_submodule('.'.join(name.split('.'))).reset()
