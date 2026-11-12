"""Federated training and full-ranking evaluation of FedGRACE.

Per round, every sampled client starts from its own parameters with the latest global parameters
(FedAvg of the previous round) and runs one or more local epochs. Each mini-batch is a two-step update:

  step 1  personalized parameters (f_pers / FedRAP's D and the prediction head):
          L_rec + lambda_orth * L_orth, with z_align detached;
  step 2  global parameters (item table and f_align / FedRAP's C):
          L_rec + lambda_orth * L_orth + lambda_align * L_align, gradients clipped to norm 1.

The server then averages the global parameters of the participants uniformly.
"""
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import UserItemRatingDataset

TOP_KS = (10, 20, 50)


def orthogonality_loss(z_align, z_pers):
    """L_orth: mean squared cosine between the two components of each item."""
    return (z_align * z_pers).sum(dim=-1).pow(2).mean()


def alignment_loss(z_align, anchor):
    """L_align: mean of 1 - cos(z_align, a_i)."""
    return (1 - F.cosine_similarity(z_align, anchor, dim=-1)).mean()


class Engine(object):
    def __init__(self, config, model, anchor):
        self.config = config
        self.device = torch.device(config['device'])
        self.model = model.to(self.device)            # server copy (global parameters)
        self.anchor = anchor.to(self.device)          # frozen server anchors a_i
        self.server_params = {}                       # FedAvg of the last round
        self.client_params = {}                       # each client's own parameters (CPU)
        self.crit = torch.nn.BCEWithLogitsLoss()
        # The global item table gets lr x num_items x lr_eta.
        self.lr_item = config['lr'] * config['num_items'] * config['lr_eta']

    def is_global(self, key):
        return key.split('.', 1)[0] in self.model.global_modules

    # ------------------------------------------------------------------ training
    def train_batch(self, model, items, ratings, optimizer_local, optimizer_global):
        lambda_orth, lambda_align = self.config['lambda_orth'], self.config['lambda_align']
        unique_items = torch.unique(items)

        # step 1: personalized parameters
        optimizer_local.zero_grad()
        loss_local = self.crit(model(items).view(-1), ratings)
        z_align, z_pers = model.representations(unique_items)
        loss_local = loss_local + lambda_orth * orthogonality_loss(z_align.detach(), z_pers)
        loss_local.backward()
        optimizer_local.step()

        # step 2: global parameters
        optimizer_global.zero_grad()
        z_align, z_pers = model.representations(unique_items)
        anchor = self.anchor[unique_items]
        bce = self.crit(model(items).view(-1), ratings)
        orth = orthogonality_loss(z_align, z_pers)
        align = alignment_loss(z_align, anchor)
        loss = bce + lambda_orth * orth + lambda_align * align
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for group in optimizer_global.param_groups for p in group['params']],
                                       max_norm=1.0)
        optimizer_global.step()

        with torch.no_grad():
            cos = (z_align * z_pers).sum(-1).abs().mean().item()
        return {'bce': bce.item(), 'orth': orth.item(), 'align': align.item(), 'cos_align_pers': cos}

    def train_round(self, train_data):
        """One communication round. train_data = [users, items, ratings], one list per user."""
        num_users = self.config['num_users']
        participants = random.sample(range(num_users), int(num_users * self.config['clients_sample_ratio']))
        model = copy.deepcopy(self.model)
        uploads, diags = {}, []

        for user in participants:
            # the client's own parameters, overwritten by the latest global parameters
            state = dict(self.model.state_dict())
            for key, value in self.client_params.get(user, {}).items():
                state[key] = value.to(self.device)
            for key, value in self.server_params.items():
                state[key] = value.to(self.device)
            model.load_state_dict(state)

            local_groups, global_groups = model.param_groups(
                self.config['lr'], self.lr_item, self.config.get('lr_personal'))
            optimizer_local = torch.optim.SGD(local_groups)
            optimizer_global = torch.optim.SGD(global_groups)

            dataset = UserItemRatingDataset(torch.LongTensor(train_data[0][user]),
                                            torch.LongTensor(train_data[1][user]),
                                            torch.FloatTensor(train_data[2][user]))
            loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
            model.train()
            for _ in range(self.config['local_epoch']):
                for _, items, ratings in loader:
                    diags.append(self.train_batch(model, items.to(self.device), ratings.float().to(self.device),
                                                  optimizer_local, optimizer_global))

            self.client_params[user] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            uploads[user] = {k: v for k, v in self.client_params[user].items() if self.is_global(k)}

        self.aggregate(uploads)
        return {key: float(np.mean([d[key] for d in diags])) for key in diags[0]}

    def aggregate(self, uploads):
        """FedAvg: uniform mean of the participants' global parameters."""
        server = None
        for params in uploads.values():
            if server is None:
                server = {k: v.clone() for k, v in params.items()}
            else:
                for key in params:
                    server[key] += params[key]
        for key in server:
            server[key] /= len(uploads)
        self.server_params = server
        self.model.load_state_dict(server, strict=False)

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, eval_data, history):
        """Full ranking: every user's held-out item is ranked against all items, the user's other observed
        items excluded, with the user's own model. Returns {'hr@k', 'ndcg@k'} for k in TOP_KS."""
        users, targets = eval_data[0].to(self.device), eval_data[1].to(self.device)
        all_items = torch.arange(self.config['num_items'], device=self.device)
        model = copy.deepcopy(self.model)
        model.eval()
        hits = {k: [] for k in TOP_KS}
        ndcgs = {k: [] for k in TOP_KS}
        for user in range(self.config['num_users']):
            params = self.client_params.get(user)
            if params is None:
                model.load_state_dict(self.model.state_dict())
            else:
                model.load_state_dict({k: v.to(self.device) for k, v in params.items()})
            scores = model(all_items).view(-1)
            target = targets[user].item()
            seen = set(history.get(users[user].item(), ()))
            seen.discard(target)
            scores[list(seen)] = -float('inf')
            top = torch.topk(scores, k=min(max(TOP_KS), scores.size(0))).indices.cpu().tolist()
            for k in TOP_KS:
                if target in top[:k]:
                    hits[k].append(1.0)
                    ndcgs[k].append(1.0 / np.log2(top[:k].index(target) + 2))
                else:
                    hits[k].append(0.0)
                    ndcgs[k].append(0.0)
        metrics = {}
        for k in TOP_KS:
            metrics[f'hr@{k}'] = float(np.mean(hits[k]))
            metrics[f'ndcg@{k}'] = float(np.mean(ndcgs[k]))
        return metrics
