"""Inspectable PyTorch LoRA: frozen linear weight plus trainable low-rank delta."""
import math
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        if rank < 1: raise ValueError('rank must be positive')
        self.base=base; self.base.requires_grad_(False); self.scale=alpha/rank
        self.dropout=nn.Dropout(dropout)
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):
        delta=nn.functional.linear(nn.functional.linear(self.dropout(x.float()),self.a),self.b)
        return self.base(x)+delta.to(x.dtype)*self.scale

def configure_lora(model,rank=8,alpha=16,dropout=0.05,train_projector=True):
    model.requires_grad_(False);targets=[]
    for name,module in list(model.named_modules()):
        if 'language_model' in name and name.rsplit('.',1)[-1] in ['q_proj','v_proj'] and isinstance(module,nn.Linear):
            parent_name,attribute=name.rsplit('.',1)
            setattr(model.get_submodule(parent_name),attribute,LoRALinear(module,rank,alpha,dropout));targets.append(name)
    if not targets:raise ValueError('No language q/v projections found')
    if train_projector:
        for name,parameter in model.named_parameters():
            if 'multi_modal_projector' in name:parameter.requires_grad_(True)
    return targets

def adapter_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def restore_adapter(model,state):
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    if expected!=set(state):raise ValueError('Adapter keys/config do not match')
    parameters=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in state.items():parameters[name].copy_(value.to(parameters[name]))
