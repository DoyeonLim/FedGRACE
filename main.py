"""Train and evaluate FedGRACE on one backbone and dataset.

    python main.py --backbone fedncf --dataset amazon-office --lambda_orth 5 --lambda_align 5 --beta 2

Each round trains every sampled client, aggregates the global parameters, and evaluates all users on the
validation and test splits (full ranking). The reported result is the test performance at the round with the
best validation NDCG@10; training stops early after --early_stop_patience rounds without improvement.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # deterministic cuBLAS; must be set before torch loads

import argparse
import csv
import json
import logging
import time
from datetime import datetime

import torch

from fedgrace.anchor import build_anchor
from fedgrace.data import DATASETS, SampleGenerator, load_item_embeddings, load_ratings
from fedgrace.engine import TOP_KS, Engine
from fedgrace.model import build_model
from fedgrace.utils import EarlyStopping, ensure_dir, format_metrics, set_seed, setup_logging


def parse_args():
    p = argparse.ArgumentParser(description='FedGRACE federated training', allow_abbrev=False,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--backbone', choices=['fedncf', 'pfedrec', 'fedrap'], default='fedncf')
    p.add_argument('--dataset', choices=DATASETS, default='amazon-office')

    g = p.add_argument_group('FedGRACE')
    g.add_argument('--lambda_orth', '--lambda1', dest='lambda_orth', type=float, default=1.0,
                   help='weight of the orthogonality loss L_orth')
    g.add_argument('--lambda_align', '--lambda2', dest='lambda_align', type=float, default=1.0,
                   help='weight of the alignment loss L_align')
    g.add_argument('--beta', type=float, default=1.0,
                   help='anchor offset scale: 0 collapses every item onto its cluster vertex')
    g.add_argument('--num_clusters', type=int, default=None,
                   help='K, number of k-means clusters = ETF vertices, 2 .. latent_dim + 1 (None: latent_dim + 1)')
    g.add_argument('--anchor_seed', type=int, default=42, help='random state of the server k-means')

    g = p.add_argument_group('model')
    g.add_argument('--latent_dim', type=int, default=32,
                   help='d, dimension of z_align, z_pers and the anchors (FedRAP: of its item tables)')
    g.add_argument('--cf_dim', type=int, default=64, help='item-embedding dimension (FedNCF, PFedRec)')

    g = p.add_argument_group('optimization')
    g.add_argument('--lr', type=float, default=0.5, help='learning rate of heads and prediction layers')
    g.add_argument('--lr_eta', type=float, default=80,
                   help='the global item table is trained with lr x num_items x lr_eta')
    g.add_argument('--lr_personal', type=float, default=None,
                   help="FedRAP only: lr of the private item table D (None: the global item table's lr)")
    g.add_argument('--batch_size', type=int, default=256)
    g.add_argument('--local_epoch', type=int, default=1)
    g.add_argument('--num_negative', type=int, default=4, help='sampled negatives per training positive')

    g = p.add_argument_group('federated training')
    g.add_argument('--num_round', type=int, default=300)
    g.add_argument('--clients_sample_ratio', type=float, default=1.0, help='fraction of clients per round')
    g.add_argument('--early_stop_patience', type=int, default=50)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--device', default='cuda', help="'cuda', 'cuda:<id>' or 'cpu'")

    g = p.add_argument_group('output')
    g.add_argument('--out_dir', default='results', help='runs are written to <out_dir>/<backbone>/<dataset>/<run>')
    g.add_argument('--use_wandb', type=int, default=0, choices=[0, 1], help='log to Weights & Biases')
    g.add_argument('--wandb_project', default='FedGRACE')

    args = p.parse_args()
    if args.num_clusters is not None and not 2 <= args.num_clusters <= args.latent_dim + 1:
        p.error('--num_clusters must lie in [2, latent_dim + 1]')
    if not 0.0 < args.clients_sample_ratio <= 1.0:
        p.error('--clients_sample_ratio must lie in (0, 1]')
    if args.lr_personal is not None and args.backbone != 'fedrap':
        p.error('--lr_personal applies to --backbone fedrap only')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA is not available; pass --device cpu')
    return args


def run_name(args):
    name = f'lr{args.lr}_orth{args.lambda_orth}_align{args.lambda_align}_beta{args.beta}'
    if args.lr_personal is not None:
        name += f'_lrp{args.lr_personal}'
    if args.num_clusters is not None:
        name += f'_K{args.num_clusters}'
    return f'{name}_seed{args.seed}'


def main():
    args = parse_args()
    set_seed(args.seed)
    config = vars(args)
    cli_config = dict(config)
    run_dir = ensure_dir(os.path.join(args.out_dir, args.backbone, args.dataset,
                                      f'{run_name(args)}_{datetime.now():%Y%m%d-%H%M%S}'))
    setup_logging(os.path.join(run_dir, 'train.log'))
    logging.info('config %s', json.dumps(cli_config))
    device = torch.device(args.device)
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device)
    wandb = None
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=f'{args.backbone}_{args.dataset}_{run_name(args)}',
                   config=cli_config, dir=run_dir)

    # data and the server-side anchor (built once, before training)
    ratings, config['num_users'], config['num_items'], item_list = load_ratings(args.dataset)
    embeddings = load_item_embeddings(args.dataset, item_list)
    anchor = build_anchor(embeddings, args.latent_dim, args.beta, num_clusters=args.num_clusters,
                          seed=args.anchor_seed)
    del embeddings
    engine = Engine(config, build_model(config), anchor)
    logging.info('%s', engine.model)

    data = SampleGenerator(ratings)
    val_data, test_data = data.validate_data, data.test_data
    val_history, test_history = data.get_user_history('validation'), data.get_user_history('test')

    metric_names = [f'{m}@{k}' for k in TOP_KS for m in ('hr', 'ndcg')]
    diag_names = ['bce', 'orth', 'align', 'cos_align_pers']
    history_file = open(os.path.join(run_dir, 'history.csv'), 'w', newline='')
    writer = csv.writer(history_file)
    writer.writerow(['round', 'train_seconds'] + diag_names
                    + [f'val_{m}' for m in metric_names] + [f'test_{m}' for m in metric_names])

    stopper = EarlyStopping(args.early_stop_patience)
    best = {'round': -1, 'val': {'ndcg@10': -float('inf')}, 'test': None}
    for round_id in range(args.num_round):
        start = time.perf_counter()
        diag = engine.train_round(data.store_all_train_data(args.num_negative))
        train_seconds = time.perf_counter() - start
        test_metrics = engine.evaluate(test_data, test_history)
        val_metrics = engine.evaluate(val_data, val_history)

        if val_metrics['ndcg@10'] > best['val']['ndcg@10']:
            best = {'round': round_id, 'val': val_metrics, 'test': test_metrics}
        logging.info('[round %d] train %.1fs  ' + '  '.join(f'{k}={v:.4f}' for k, v in diag.items()),
                     round_id, train_seconds)
        logging.info('[round %d] val   %s', round_id, format_metrics(val_metrics))
        logging.info('[round %d] test  %s', round_id, format_metrics(test_metrics))
        logging.info('[best round %d] test HR@10=%.4f NDCG@10=%.4f (val NDCG@10=%.4f)', best['round'],
                     best['test']['hr@10'], best['test']['ndcg@10'], best['val']['ndcg@10'])
        writer.writerow([round_id, train_seconds] + [diag[n] for n in diag_names]
                        + [val_metrics[m] for m in metric_names] + [test_metrics[m] for m in metric_names])
        history_file.flush()
        if wandb is not None:
            log = {f'diag/{k}': v for k, v in diag.items()}
            log.update({f'val/{k}': v for k, v in val_metrics.items()})
            log.update({f'test/{k}': v for k, v in test_metrics.items()})
            log.update({'best/round': best['round'], 'best/val_ndcg@10': best['val']['ndcg@10'],
                        'best/test_hr@10': best['test']['hr@10'], 'best/test_ndcg@10': best['test']['ndcg@10']})
            wandb.log(log, step=round_id)

        stopper(val_metrics['ndcg@10'])
        if stopper.stop:
            logging.info('early stopping: no validation improvement for %d rounds', args.early_stop_patience)
            break
    history_file.close()

    result = {'config': cli_config, 'num_users': config['num_users'], 'num_items': config['num_items'],
              'rounds_run': round_id + 1, 'best_round': best['round'],
              'best_val': best['val'], 'best_test': best['test']}
    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    logging.info('best round %d | test %s', best['round'], format_metrics(best['test']))
    logging.info('results written to %s', run_dir)
    if wandb is not None:
        wandb.summary.update({f'best/test_{k}': v for k, v in best['test'].items()})
        wandb.finish()


if __name__ == '__main__':
    main()
