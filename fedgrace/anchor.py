"""Controlled-Collapse Geometric Anchoring (CGA).

The server turns the LLM item embeddings s_i into one frozen anchor a_i per item, once, before training:

  (i)   centering + PCA whitening to d = latent_dim dimensions, then projection onto the unit sphere;
  (ii)  K-means of the whitened directions into K clusters (default K = d + 1, the most an ETF admits in R^d);
  (iii) a Simplex ETF rotated onto the centred cluster centroids by orthogonal Procrustes;
  (iv)  a_i = normalize(v_k(i) + beta * (s~_i - mu_k(i))): beta = 0 collapses every item onto its cluster's
        vertex, a larger beta keeps more of the item's own offset from the cluster centroid.
"""
import logging
import math

import torch
import torch.nn.functional as F


def matching_etf(M_hat):
    """Simplex ETF of C unit vectors in R^d (C - 1 <= d) rotated onto the target directions M_hat [C, d].

    The canonical ETF P is rotated by the orthogonal U that best maps it onto M_hat (Procrustes, one SVD).
    """
    C, d = M_hat.shape
    if C - 1 > d:
        raise ValueError(f'a Simplex ETF requires C - 1 <= d, got C={C}, d={d}')
    P = math.sqrt(C / (C - 1)) * (torch.eye(C, device=M_hat.device, dtype=M_hat.dtype)
                                   - torch.ones(C, C, device=M_hat.device, dtype=M_hat.dtype) / C)
    S = M_hat.T @ P.T                       # [d, C]
    W, _, Vt = torch.linalg.svd(S, full_matrices=False)
    U = W @ Vt                              # [d, C], U^T U = I
    return F.normalize((U @ P).T, dim=1)    # [C, d]


def etf_gram_sanity(H):
    """Off-diagonal Gram entries of a Simplex ETF all equal -1 / (C - 1)."""
    C = H.shape[0]
    off_diag = (H @ H.T)[~torch.eye(C, dtype=torch.bool, device=H.device)]
    target = -1.0 / (C - 1)
    return {'target': target, 'mean': off_diag.mean().item(), 'max_abs_dev': (off_diag - target).abs().max().item()}


def build_anchor(embeddings, latent_dim, beta, num_clusters=None, seed=42):
    """Frozen per-item anchors, float32 tensor [num_items, latent_dim].

    embeddings: float32 array [num_items, llm_dim]; seed: random state of K-means.
    """
    from sklearn.cluster import KMeans

    # (i) centering and PCA whitening
    raw = torch.tensor(embeddings, dtype=torch.float64)
    centered = raw - raw.mean(dim=0, keepdim=True)
    if centered.shape[0] < 2 or latent_dim > centered.shape[1]:
        raise ValueError('PCA needs at least two items and latent_dim <= the embedding dimension')
    covariance = (centered.T @ centered) / (centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    top_vals = eigenvalues[-latent_dim:].flip(0)
    top_vecs = eigenvectors[:, -latent_dim:].flip(1)
    eps = 1e-3 * top_vals[0].item()
    if eps <= 0 or not torch.isfinite(top_vals).all():
        raise ValueError('the embedding covariance has no finite positive leading eigenvalue')
    proj = (centered @ top_vecs).float()
    directions = F.normalize(proj * (1.0 / torch.sqrt(top_vals + eps)).float(), dim=1)  # whitened unit directions
    logging.info('[anchor] whitened PCA to %d dims, eigenvalue ratio %.4g',
                 latent_dim, top_vals[0].item() / max(top_vals[-1].item(), 1e-30))

    # (ii) K-means clusters and their centroids
    cluster_count = latent_dim + 1 if num_clusters is None else num_clusters
    labels = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10).fit_predict(directions.numpy())
    labels = torch.as_tensor(labels, dtype=torch.long)
    means = torch.zeros(cluster_count, latent_dim)
    counts = torch.zeros(cluster_count)
    means.index_add_(0, labels, directions)
    counts.index_add_(0, labels, torch.ones(len(directions)))
    means = means / counts.clamp(min=1).unsqueeze(1)

    # (iii) Procrustes-to-ETF
    vertices = matching_etf(F.normalize(means - means.mean(0), dim=1))
    sanity = etf_gram_sanity(vertices)
    if sanity['max_abs_dev'] >= 1e-5:
        raise ValueError(f'ETF Gram check failed: {sanity}')

    # (iv) controlled-collapse interpolation
    anchor = F.normalize(vertices[labels] + beta * (directions - means[labels]), dim=1)
    logging.info('[anchor] K = %d clusters (sizes %d-%d), beta %g, k-means seed %d',
                 cluster_count, int(counts.min()), int(counts.max()), beta, seed)
    return anchor
