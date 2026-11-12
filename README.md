# FedGRACE

PyTorch implementation of **FedGRACE: Federated Geometric Representation Anchoring via Controlled-Collapse ETF
for LLM-Enhanced Recommendation**.

FedGRACE brings the semantics of LLM item embeddings into federated recommendation (FedRec) by aligning ID-based
item representations to a geometry built for ranking, instead of to the raw LLM space:

- **Controlled-Collapse Geometric Anchoring (CGA, server).** The server whitens the LLM item embeddings to the ID
  dimension $d$, clusters them into $K = d + 1$ groups, rotates a Simplex ETF onto the cluster centroids (orthogonal
  Procrustes) and places every item at its cluster's vertex plus $\beta$ times its own within-cluster offset.
  The anchors are built once, before training, and broadcast to the clients; the raw LLM embeddings never leave the server.
- **Disentangled Selective Alignment (DSA, client).** Each client splits an item representation into an alignable
  component $z^{\text{align}}$, pulled toward the item's anchor ($\mathcal{L}_{\text{align}}$), and a personalized component
  $z^{\text{pers}}$, kept orthogonal to it ($\mathcal{L}_{\text{orth}}$) and free of the alignment. The prediction head
  scores their concatenation. The client objective is

  $$\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_{\text{orth}}\,\mathcal{L}_{\text{orth}} + \lambda_{\text{align}}\,\mathcal{L}_{\text{align}}$$

One code base runs FedGRACE on three FedRec backbones:

| backbone | item representation | uploaded to the server (FedAvg) | kept on the client |
|---|---|---|---|
| `fedncf`  | item table $E$ → heads $f^{\text{align}}$, $f^{\text{pers}}$ | $E$, $f^{\text{align}}$ | $f^{\text{pers}}$, user vector, MLP scorer |
| `pfedrec` | item table $E$ → heads $f^{\text{align}}$, $f^{\text{pers}}$ | $E$, $f^{\text{align}}$ | $f^{\text{pers}}$, affine scorer |
| `fedrap`  | FedRAP's global table $C$ (→ $z^{\text{align}}$) and private table $D$ (→ $z^{\text{pers}}$) | $C$ | $D$, affine scorer |

## Repository layout

```
main.py                    training and evaluation of one backbone on one dataset
fedgrace/
  data.py                  interactions, leave-two-out split, negative sampling, LLM item embeddings
  anchor.py                CGA: server-side anchor construction
  model.py                 FedGRACE on FedNCF / PFedRec / FedRAP
  engine.py                federated rounds (two-step local update, FedAvg) and full-ranking evaluation
  utils.py                 seeding, logging, early stopping
scripts/
  run_main.sh              tuned configurations of the main results
  convert_embeddings.py    json item embeddings -> npz (for adding a dataset)
dataset/<name>/            ratings.csv and item_embedding*.npz (Last.fm, Office, Music)
```

## Installation

```bash
conda create -n fedgrace python=3.8 -y
conda activate fedgrace
pip install -r requirements.txt
```

`requirements.txt` pins the versions the results were produced with (PyTorch 2.4.1 with CUDA 12.1). For a different
CUDA version, install the matching PyTorch 2.4.1 build first. Weights & Biases logging is optional
(`pip install wandb`, then `--use_wandb 1`).

## Datasets

The paper uses six datasets. Three of them (`lastfm-2k`, `amazon-office`, `amazon-music`) are included in
`dataset/`; the other three are not part of this repository, and their statistics are
listed for reference. Statistics after loading (see the protocol below):

| `--dataset` | source | filtering | users | items | interactions | density | included |
|---|---|---|---:|---:|---:|---:|:---:|
| `lastfm-2k`      | HetRec 2011 Last.fm                  | 5-core                 | 1,397 | 11,981 | 69,453 | 0.41% | ✓ |
| `amazon-office`  | Amazon Office Products (2014)        | 5-core                 | 4,905 | 2,420 | 53,258 | 0.45% | ✓ |
| `amazon-music`   | Amazon Digital Music (2014)          | 5-core                 | 5,541 | 3,568 | 64,706 | 0.33% | ✓ |
| `amazon-science` | Amazon Industrial & Scientific (2018)| 5-core                 | 8,516 | 4,408 | 59,519 | 0.16% | |
| `amazon-video`   | Amazon Video Games (2018)            | rating ≥ 3, 10-core    | 4,061 | 2,774 | 69,850 | 0.62% | |
| `amazon-toys`    | Amazon Toys & Games (2018)           | rating ≥ 3, 10-core    | 8,530 | 6,771 | 152,290 | 0.26% | |

Each folder contains

- `ratings.csv`: `user,item,rating,timestamp`, no header (raw ids). Every interaction is a positive.
- `item_embedding*.npz`: `item_ids` (raw item ids) and `embeddings` (float32, 3072-d). For every item, GPT-4o-mini
  wrote a short profile from its metadata (title, brand, category and description for Amazon; artist name and
  user tags for Last.fm), which was encoded with OpenAI `text-embedding-3-large`. Last.fm is stored as three files to
  stay below GitHub's file-size limit; all `item_embedding*.npz` files of a folder are loaded together.

The interactions come from public research datasets: the Amazon review data (2014: He and McAuley, WWW 2016;
2018: Ni, Li and McAuley, EMNLP 2019) and HetRec 2011 Last.fm (Cantador, Brusilovsky and Kuflik, RecSys 2011).
They are redistributed here for research use only and remain subject to the terms of their original sources.
Please cite the original sources as well when you use them.

To add a dataset, put its `ratings.csv` into `dataset/<name>/`, convert the item embeddings with
`python scripts/convert_embeddings.py --json <file>.json --ratings dataset/<name>/ratings.csv --out_dir dataset/<name>`,
and add `<name>` to `DATASETS` in `fedgrace/data.py`. Another dataset root can be used with `FEDGRACE_DATASET_DIR`.

## Quick start

FedGRACE on FedNCF with Amazon Office, using the tuned configuration of the paper:

```bash
python main.py --backbone fedncf --dataset amazon-office \
    --lr 0.5 --lr_eta 80 --lambda_orth 0.01 --lambda_align 1.0 --beta 2.0 \
    --latent_dim 32 --cf_dim 64 --anchor_seed 42 --seed 42 \
    --num_round 300 --early_stop_patience 50
```

Our run selects round 77 by validation NDCG@10 and stops early after round 127. Its test results are:

| HR@10 | NDCG@10 | HR@20 | NDCG@20 | HR@50 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|
| 0.0279 | 0.0142 | 0.0434 | 0.0181 | 0.0734 | 0.0239 |

To check your setup without the full run, compare the first round in `train.log` (or row 0 of `history.csv`):
`bce=0.5670 orth=0.0309 align=1.0016`, test `HR@10=0.0084 NDCG@10=0.0052`, val `HR@10=0.0039 NDCG@10=0.0022`.
Training takes about 1–2 minutes per round on one GPU.

A fast functional check (2 rounds with 10% of the clients):

```bash
python main.py --backbone pfedrec --dataset amazon-office --num_round 2 --clients_sample_ratio 0.1
```

## Main options

| option | default | meaning |
|---|---|---|
| `--backbone` | `fedncf` | `fedncf`, `pfedrec` or `fedrap` |
| `--dataset` | `amazon-office` | one of the datasets above |
| `--lambda_orth` | 1.0 | $\lambda_{\text{orth}}$, weight of $\mathcal{L}_{\text{orth}}$ (alias `--lambda1`) |
| `--lambda_align` | 1.0 | $\lambda_{\text{align}}$, weight of $\mathcal{L}_{\text{align}}$ (alias `--lambda2`) |
| `--beta` | 1.0 | $\beta$, offset of an item from its ETF vertex (0 = every item collapses onto its vertex) |
| `--num_clusters` | $d+1$ | $K$, number of k-means clusters / ETF vertices ($2 \ldots d+1$) |
| `--anchor_seed` | 42 | random state of the server's k-means |
| `--latent_dim` | 32 | $d$, dimension of $z^{\text{align}}$, $z^{\text{pers}}$ and the anchors (FedRAP: of $C$ and $D$) |
| `--cf_dim` | 64 | dimension of the item table $E$ (FedNCF, PFedRec) |
| `--lr` | 0.5 | learning rate of the heads and prediction layers |
| `--lr_eta` | 80 | the global item table ($E$, or $C$ for FedRAP) is trained with `lr × #items × lr_eta` |
| `--lr_personal` | item-table lr | FedRAP only: learning rate of the private item table $D$ |
| `--num_round` / `--early_stop_patience` | 300 / 50 | communication rounds / rounds without validation improvement before stopping |
| `--clients_sample_ratio` | 1.0 | fraction of clients that train in each round |
| `--num_negative` / `--batch_size` / `--local_epoch` | 4 / 256 / 1 | local training |
| `--seed`, `--device` | 42, `cuda` | `--device cpu` runs without a GPU |

Ablations of the paper map to options: without $\mathcal{L}_{\text{orth}}$ `--lambda_orth 0`, without $\mathcal{L}_{\text{align}}$
`--lambda_align 0`, full collapse `--beta 0`. Run `python main.py -h` for all options.

## Outputs

Each run writes `results/<backbone>/<dataset>/<run name>_<time>/` (change the root with `--out_dir`):

- `train.log`: configuration, anchor statistics and per-round training diagnostics and metrics,
- `history.csv`: per round, the mean losses (`bce`, `orth`, `align`), the mean absolute cosine between $z^{\text{align}}$ and $z^{\text{pers}}$,
  and validation / test HR and NDCG at 10, 20 and 50,
- `result.json`: the configuration, the best round, and its validation and test metrics.

## Protocol

- **Data.** Each (user, item) pair keeps its latest interaction; users with fewer than three items are removed.
  The last item of every user (by time) is the test item and the second last the validation item.
- **Training.** In every round, each participating client trains one local epoch on its training items, each paired
  with 4 negatives drawn from the items it has not interacted with in training (held-out items are never consulted).
  A mini-batch is a two-step SGD update: (1) the client-side parameters on
  $\mathcal{L}_{\text{rec}} + \lambda_{\text{orth}}\mathcal{L}_{\text{orth}}$ with $z^{\text{align}}$ detached; (2) the global parameters on the full
  objective, with gradients clipped to norm 1. The server averages the uploaded global parameters uniformly.
- **Evaluation.** Full ranking: every user's held-out item is ranked against all items with the user's own model,
  excluding the user's training items (validation) or training and validation items (test); HR@K and NDCG@K for
  K = 10, 20, 50. We report the test metrics of the round with the best validation NDCG@10.
- **Cost.** Every client keeps its own copy of the model (on CPU), so memory grows with users × items × dimension:
  about 3 GB for Office. A round takes about 1–4 minutes on one GPU, a full run several hours.

## Citation

The citation will be added upon publication.

## License

The code is released under the [MIT License](LICENSE). The datasets in `dataset/` are not covered by it; they
follow the terms of their original sources (see [Datasets](#datasets)).
