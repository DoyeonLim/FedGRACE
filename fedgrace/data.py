"""Interaction data, leave-two-out splits, training negatives and LLM item embeddings."""
import glob
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Dataset root: $FEDGRACE_DATASET_DIR, or <repo>/dataset.
DATASET_DIR = os.environ.get(
    'FEDGRACE_DATASET_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset'))

DATASETS = ('lastfm-2k', 'amazon-office', 'amazon-music', 'amazon-science', 'amazon-video', 'amazon-toys')
RATINGS_FILE = 'ratings.csv'  # user, item, rating, timestamp (no header)
# item_embedding.npz, or item_embedding.part<i>.npz shards (keeps every file below GitHub's size limit);
# keys: item_ids (raw item id, str) and embeddings (float32).
EMBEDDING_GLOB = 'item_embedding*.npz'


def dataset_file(dataset, filename):
    path = os.path.join(DATASET_DIR, dataset, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist (set FEDGRACE_DATASET_DIR to the dataset root)')
    return path


def read_interactions(path):
    return pd.read_csv(path, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')


def load_ratings(dataset):
    """Read the interactions and map them to contiguous user/item ids.

    Each (user, item) pair keeps its latest event; users with fewer than three distinct items are
    removed (one item each for training, validation and test). Equal timestamps keep the file order.
    Returns (ratings[userId, itemId, rating, timestamp], num_users, num_items, raw item id per itemId).
    """
    rating = read_interactions(dataset_file(dataset, RATINGS_FILE))
    if rating.isna().any().any():
        raise ValueError('Interactions must contain non-missing ids, ratings and timestamps')
    for column in ('rating', 'timestamp'):
        rating[column] = pd.to_numeric(rating[column], errors='raise')
        if not np.isfinite(rating[column].to_numpy()).all():
            raise ValueError(f'{column} must be finite')
    original_rows, original_users = len(rating), rating['uid'].nunique()
    rating = rating.loc[rating['rating'] > 0].copy()
    rating = rating.sort_values(['uid', 'timestamp'], kind='mergesort')
    rating = rating.drop_duplicates(['uid', 'mid'], keep='last')
    counts = rating.groupby('uid')['mid'].transform('size')
    rating = rating.loc[counts >= 3].reset_index(drop=True)
    logging.info('[data] %s: rows %d -> %d, users %d -> %d after de-duplication and the 3-item filter',
                 dataset, original_rows, len(rating), original_users, rating['uid'].nunique())

    user_id = rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    rating = pd.merge(rating, user_id, on=['uid'], how='left')
    item_id = rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    rating = pd.merge(rating, item_id, on=['mid'], how='left')

    item_list = item_id.sort_values(by='itemId')['mid'].astype(str).tolist()
    rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    logging.info('[data] %d users, %d items, %d interactions', len(user_id), len(item_id), len(rating))
    return rating, len(user_id), len(item_id), item_list


def load_item_embeddings(dataset, item_list):
    """LLM embedding of every item in item_list order, float32 [num_items, dim]; missing items get zeros."""
    files = sorted(glob.glob(os.path.join(DATASET_DIR, dataset, EMBEDDING_GLOB)))
    if not files:
        raise FileNotFoundError(f'no {EMBEDDING_GLOB} in {os.path.join(DATASET_DIR, dataset)}')
    item_ids, tables = [], []
    for path in files:
        with np.load(path) as data:
            item_ids.extend(data['item_ids'].tolist())
            tables.append(data['embeddings'])
    table = np.concatenate(tables) if len(tables) > 1 else tables[0]
    row_of = {item: row for row, item in enumerate(item_ids)}
    embedding = np.zeros((len(item_list), table.shape[1]), dtype=np.float32)
    missing = 0
    for row, item in enumerate(item_list):
        source = row_of.get(item)
        if source is None:
            missing += 1
        else:
            embedding[row] = table[source]
    if missing:
        logging.warning('[data] %d items have no LLM embedding (zero vectors)', missing)
    if not np.isfinite(embedding).all():
        raise ValueError('LLM embeddings must be finite')
    logging.info('[data] LLM item embeddings %s', embedding.shape)
    return embedding


def set_sampler(k):
    """random.sample(k) from each set, over a cached sorted copy (a deterministic population order)."""
    as_tuple = {}

    def sample(s):
        t = as_tuple.get(id(s))
        if t is None:
            t = as_tuple[id(s)] = tuple(sorted(s))
        return random.sample(t, k)

    return sample


def split_by_user(train_ratings, num_negatives):
    """Rows [positive, negative_1..negative_k] with labels [1, 0, .., 0], grouped by user.

    Returns (users, items, ratings): one python list per user, in ascending user id.
    """
    order = np.argsort(train_ratings['userId'].to_numpy(), kind='stable')
    uid = train_ratings['userId'].to_numpy()[order]
    pos = train_ratings['itemId'].to_numpy()[order]
    rating = train_ratings['rating'].to_numpy(dtype=float)[order]
    neg = np.array(train_ratings['negatives'].tolist(), dtype=np.int64).reshape(len(order), num_negatives)[order]

    width = 1 + num_negatives
    flat_users = np.repeat(uid, width)
    flat_items = np.concatenate([pos[:, None], neg], axis=1).reshape(-1)
    flat_ratings = np.concatenate([rating[:, None], np.zeros((len(order), num_negatives))], axis=1).reshape(-1)

    cuts = (np.flatnonzero(np.diff(uid)) + 1) * width
    users = [a.tolist() for a in np.split(flat_users, cuts)]
    items = [a.tolist() for a in np.split(flat_items, cuts)]
    ratings = [a.tolist() for a in np.split(flat_ratings, cuts)]
    return users, items, ratings


class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Leave-two-out split by time (the last item is the test item, the second last the validation item)
    and training negatives drawn from the items a user has not interacted with in training."""

    def __init__(self, ratings):
        columns = ['userId', 'itemId', 'rating', 'timestamp']
        ratings = ratings[columns].copy()
        for column in columns:
            ratings[column] = pd.to_numeric(ratings[column], errors='raise')
        for column in ('userId', 'itemId'):
            ratings[column] = ratings[column].astype(np.int64)
            unique = np.sort(ratings[column].unique())
            if not np.array_equal(unique, np.arange(len(unique))):
                raise ValueError(f'{column} must be contiguous; use load_ratings()')
        if ratings.duplicated(['userId', 'itemId']).any():
            raise ValueError('Duplicate user-item pairs; use load_ratings()')
        if (ratings.groupby('userId')['itemId'].size() < 3).any():
            raise ValueError('Every user needs three distinct items; use load_ratings()')

        self.ratings = ratings.sort_values(['userId', 'timestamp'], kind='mergesort').reset_index(drop=True)
        self.preprocess_ratings = self._binarize(self.ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)
        self.negatives = self._sample_negative(self.train_ratings)

    def get_user_history(self, split):
        """Items to exclude from ranking: training items for validation, training + validation items for test."""
        if split == 'validation':
            history = self.train_ratings
        elif split == 'test':
            history = pd.concat([self.train_ratings, self.val_ratings], ignore_index=True)
        else:
            raise ValueError("split must be 'validation' or 'test'")
        return history.groupby('userId')['itemId'].apply(list).to_dict()

    @staticmethod
    def _binarize(ratings):
        ratings = ratings.copy()
        ratings['rating'] = 1.0
        return ratings

    @staticmethod
    def _split_loo(ratings):
        ratings = ratings.sort_values(['userId', 'timestamp'], kind='mergesort').copy()
        ratings['rank_latest'] = ratings.groupby('userId').cumcount(ascending=False) + 1
        test = ratings[ratings['rank_latest'] == 1]
        val = ratings[ratings['rank_latest'] == 2]
        train = ratings[ratings['rank_latest'] > 2]
        keep = ['userId', 'itemId', 'rating']
        return train[keep], val[keep], test[keep]

    def _sample_negative(self, ratings):
        """Negative pool of every user: all items minus the user's training items (held-out items are
        never consulted). The 'negative_samples' column is not used, but the draw is kept so that the
        random stream, and with it every training negative, matches the runs reported in the paper."""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(
            lambda x: random.sample(sorted(x), min(198, len(x))))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def store_all_train_data(self, num_negatives):
        """Training rows of one round: every training positive with num_negatives fresh negatives,
        as [users, items, ratings] with one list per user."""
        if self.negatives['negative_items'].map(len).min() < num_negatives:
            raise ValueError('num_negative exceeds the negative pool of a user')
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(set_sampler(num_negatives))
        users, items, ratings = split_by_user(train_ratings, num_negatives)
        assert len(users) == len(self.user_pool)
        return [users, items, ratings]

    @property
    def validate_data(self):
        """[users, validation items] as LongTensors, one row per user in ascending user id."""
        return self._held_out(self.val_ratings)

    @property
    def test_data(self):
        """[users, test items] as LongTensors, one row per user in ascending user id."""
        return self._held_out(self.test_ratings)

    @staticmethod
    def _held_out(split):
        split = split.sort_values('userId', kind='mergesort')
        users = [int(u) for u in split['userId']]
        items = [int(i) for i in split['itemId']]
        return [torch.LongTensor(users), torch.LongTensor(items)]
