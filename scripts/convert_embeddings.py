"""Convert LLM item embeddings from json ({raw item id: [float, ...]}) to the npz file(s) read by FedGRACE.

    python scripts/convert_embeddings.py --json item_embedding.json \
        --ratings dataset/<name>/ratings.csv --out_dir dataset/<name> [--shards 3]

Writes <out_dir>/item_embedding.npz, or item_embedding.part<i>.npz with --shards > 1 (GitHub rejects files
above 100 MB). Items that do not occur in the ratings file are dropped. Values are stored as float32, the
precision the training code uses, so the conversion does not change any result.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedgrace.data import EMBEDDING_GLOB, read_interactions  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json', required=True, help='{raw item id: embedding} json')
    p.add_argument('--ratings', required=True, help='interaction csv: user, item, rating, timestamp (no header)')
    p.add_argument('--out_dir', required=True, help='dataset folder')
    p.add_argument('--shards', type=int, default=1, help='number of npz files to split the items into')
    p.add_argument('--overwrite', action='store_true', help='replace existing item_embedding*.npz files')
    args = p.parse_args()

    existing = glob.glob(os.path.join(args.out_dir, EMBEDDING_GLOB))
    if existing and not args.overwrite:
        p.error(f'{existing} already exist (all of them would be loaded); pass --overwrite to replace them')
    items = read_interactions(args.ratings)['mid'].astype(str).drop_duplicates().tolist()
    with open(args.json) as f:
        table = json.load(f)
    kept = [item for item in items if item in table]
    embeddings = np.array([table[item] for item in kept], dtype=np.float32)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError('embeddings must be finite vectors of one dimension')

    for path in existing:
        os.remove(path)
    os.makedirs(args.out_dir, exist_ok=True)
    bounds = np.linspace(0, len(kept), args.shards + 1).round().astype(int)
    for i in range(args.shards):
        name = 'item_embedding.npz' if args.shards == 1 else f'item_embedding.part{i + 1}.npz'
        path = os.path.join(args.out_dir, name)
        lo, hi = bounds[i], bounds[i + 1]
        np.savez_compressed(path, item_ids=np.array(kept[lo:hi]), embeddings=embeddings[lo:hi])
        print(f'{path}: items {lo}-{hi - 1}, {os.path.getsize(path) / 1e6:.1f} MB')
    print(f'{len(kept)} of {len(items)} rated items have an embedding (dim {embeddings.shape[1]})')


if __name__ == '__main__':
    main()
