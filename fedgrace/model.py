"""FedGRACE on three FedRec backbones.

Every model maps item ids to the two DSA components of the item representation,
z_align (aligned to the server anchor) and z_pers (kept free of the alignment), and scores the
concatenation with the backbone's client-side prediction head:

  FedNCF  : item table E -> heads f_align / f_pers; an MLP over [user vector; z_align; z_pers]
  PFedRec : item table E -> heads f_align / f_pers; an affine layer over [z_align; z_pers]
  FedRAP  : FedRAP's global item table C gives z_align, its private item table D gives z_pers;
            an affine layer over [z_align; z_pers]

`global_modules` are uploaded and averaged by the server every round; all other parameters stay
on the client. The frozen anchors are held by the engine, not by the model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _head(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())


class FedNCF(nn.Module):
    global_modules = ('item_embedding', 'f_align')

    def __init__(self, num_items, cf_dim, latent_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, cf_dim)
        self.f_align = _head(cf_dim, latent_dim)
        self.f_pers = _head(cf_dim, latent_dim)
        self.user_embedding = nn.Embedding(1, latent_dim)  # one client = one user
        self.mlp = nn.Sequential(
            nn.Linear(3 * latent_dim, 2 * latent_dim), nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 1))

    def representations(self, items):
        e = self.item_embedding(items)
        return F.normalize(self.f_align(e), dim=-1), F.normalize(self.f_pers(e), dim=-1)

    def forward(self, items):
        z_align, z_pers = self.representations(items)
        user = self.user_embedding(torch.tensor([0], device=items.device)).repeat(items.size(0), 1)
        return self.mlp(torch.cat([user, z_align, z_pers], dim=-1))  # logits

    def param_groups(self, lr, lr_item, lr_personal=None):
        """(local groups updated in step 1, global groups updated in step 2) for torch.optim.SGD."""
        local = [{'params': list(self.user_embedding.parameters()) + list(self.mlp.parameters()), 'lr': lr},
                 {'params': list(self.f_pers.parameters()), 'lr': lr}]
        shared = [{'params': list(self.f_align.parameters()), 'lr': lr},
                  {'params': list(self.item_embedding.parameters()), 'lr': lr_item}]
        return local, shared


class PFedRec(nn.Module):
    global_modules = ('item_embedding', 'f_align')

    def __init__(self, num_items, cf_dim, latent_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, cf_dim)
        self.f_align = _head(cf_dim, latent_dim)
        self.f_pers = _head(cf_dim, latent_dim)
        self.affine_output = nn.Linear(2 * latent_dim, 1)

    def representations(self, items):
        e = self.item_embedding(items)
        return F.normalize(self.f_align(e), dim=-1), F.normalize(self.f_pers(e), dim=-1)

    def forward(self, items):
        return self.affine_output(torch.cat(self.representations(items), dim=-1))  # logits

    def param_groups(self, lr, lr_item, lr_personal=None):
        local = [{'params': list(self.affine_output.parameters()), 'lr': lr},
                 {'params': list(self.f_pers.parameters()), 'lr': lr}]
        shared = [{'params': list(self.f_align.parameters()), 'lr': lr},
                  {'params': list(self.item_embedding.parameters()), 'lr': lr_item}]
        return local, shared


class FedRAP(nn.Module):
    global_modules = ('item_commonality',)

    def __init__(self, num_items, latent_dim):
        super().__init__()
        self.item_commonality = nn.Embedding(num_items, latent_dim)  # C, global
        self.item_personality = nn.Embedding(num_items, latent_dim)  # D, private
        self.affine_output = nn.Linear(2 * latent_dim, 1)

    def representations(self, items):
        return (F.normalize(self.item_commonality(items), dim=-1),
                F.normalize(self.item_personality(items), dim=-1))

    def forward(self, items):
        return self.affine_output(torch.cat(self.representations(items), dim=-1))  # logits

    def param_groups(self, lr, lr_item, lr_personal=None):
        # D is never aggregated; --lr_personal gives it its own learning rate (default: the item-table lr).
        local = [{'params': list(self.item_personality.parameters()),
                  'lr': lr_item if lr_personal is None else lr_personal},
                 {'params': list(self.affine_output.parameters()), 'lr': lr}]
        shared = [{'params': list(self.item_commonality.parameters()), 'lr': lr_item}]
        return local, shared


def build_model(config):
    if config['backbone'] == 'fedncf':
        return FedNCF(config['num_items'], config['cf_dim'], config['latent_dim'])
    if config['backbone'] == 'pfedrec':
        return PFedRec(config['num_items'], config['cf_dim'], config['latent_dim'])
    if config['backbone'] == 'fedrap':
        return FedRAP(config['num_items'], config['latent_dim'])
    raise ValueError(f"unknown backbone {config['backbone']!r}")
