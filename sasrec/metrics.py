import numpy as np
import torch

def ndcg_k(actual, predicted, k=10):
    """
    Computes NDCG at k.
    actual: list of relevant items (usually just one [item_id])
    predicted: list of predicted items (ranked)
    """
    idcg = 1.0
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            dcg += 1.0 / np.log2(i + 2)
    return dcg / idcg

def hit_k(actual, predicted, k=10):
    """
    Computes Hit at k.
    """
    for p in predicted[:k]:
        if p in actual:
            return 1.0
    return 0.0

