import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

def load_movielens(data_path: str):
    """
    Load MovieLens 1M dataset.
    Expects ratings.dat in data_path.
    Returns a DataFrame with [user_id, item_id, rating, timestamp].
    """
    # Check for ratings.dat directly or inside ml-1m subdirectory
    ratings_file = os.path.join(data_path, "ratings.dat")
    if not os.path.exists(ratings_file):
        ratings_file = os.path.join(data_path, "ml-1m", "ratings.dat")
        
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"ratings.dat not found in {data_path} or {data_path}/ml-1m")
    
    print(f"Loading data from {ratings_file}...")
    # ML-1M: UserID::MovieID::Rating::Timestamp
    df = pd.read_csv(
        ratings_file,
        sep="::",
        header=None,
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    return df

def data_partition(df):
    """
    Split data into train/val/test using leave-one-out strategy.
    """
    usernum = df["user_id"].max()
    itemnum = df["item_id"].max()
    
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)
    
    # Group by user
    grouped = df.groupby("user_id")
    
    for user_id, group in grouped:
        # sort by timestamp
        interactions = group.sort_values("timestamp")
        items = interactions["item_id"].tolist()
        
        if len(items) < 3:
            # If less than 3 interactions, put all in train (or ignore)
            user_train[user_id] = items
            continue
            
        # Last item -> test
        user_test[user_id] = [items[-1]]
        # Second last -> valid
        user_valid[user_id] = [items[-2]]
        # Rest -> train
        user_train[user_id] = items[:-2]
        
    return [user_train, user_valid, user_test, usernum, itemnum]

class SASRecDataset(Dataset):
    def __init__(self, user_train, usernum, itemnum, maxlen, train=True):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.train = train
        
        # Filter users who have training data
        self.users = [u for u in range(1, usernum + 1) if len(user_train[u]) > 0]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        train_items = self.user_train[user]
        
        # SASRec Training Strategy:
        # Input sequence: [x1, x2, ..., x_{n-1}]
        # Target sequence: [x2, x3, ..., x_n] (Positive samples)
        # Negative sequence: [rand, rand, ..., rand] (Negative samples)
        
        # If sequence is shorter than maxlen, we pad with 0 from LEFT.
        # If longer, we take the LAST maxlen items.
        
        # However, for training, we usually generate ONE sample per user per epoch?
        # Or we train on the full sequence?
        # Standard SASRec implementation:
        # "pos" is the sequence shifted by 1.
        # The model effectively computes loss for all positions.
        
        # Construct sequence up to maxlen
        seq_len = len(train_items)
        
        if seq_len <= self.maxlen:
            # Pad left
            padding_len = self.maxlen - seq_len
            input_seq = [0] * padding_len + train_items
            
            # Pos is inputs shifted. But we need inputs to predict pos.
            # In typical SASRec code (e.g. pmixer/TiSASRec.pytorch):
            # seq = [0, 0, 1, 2, 3]
            # pos = [0, 1, 2, 3, 4] ?? No.
            # Usually:
            # LogSeq (Input): [0, 0, 1, 2, 3]
            # PosSeq (Label): [0, 1, 2, 3, 4] (where 4 is next item NOT in input?)
            # No, standard is self-supervision on the sequence.
            # Input: [s_1, s_2, ..., s_{n-1}]
            # Target: [s_2, s_3, ..., s_n]
            
            # So if we have [1, 2, 3], input is [1, 2], target is [2, 3]?
            # Usually we use the whole sequence for training causality.
            # Let's follow standard logic:
            # seq = [0, 0, 1, 2, 3]
            # pos = [0, 0, 2, 3, 4] 
            
            # Actually, the dataset usually returns:
            # user, input_ids, target_ids (pos), neg_ids
            
            # Let's take the last `maxlen` items for training.
            # To predict the last item in `train_items`, it must be in `pos`.
            # So `input` shouldn't see the last item?
            # Standard:
            # input: train_items[:-1]
            # pos: train_items[1:]
            pass
        else:
            # Truncate, keep last maxlen
            pass

        # Let's define exactly what goes into the model.
        # We want to predict train_items[1:] given train_items[:-1].
        # Length of prediction is len(train_items) - 1.
        # We pad to maxlen.
        
        # Sequence handling
        if seq_len < 2:
            # Too short to train
            return self.__getitem__((idx + 1) % len(self))

        # Truncate to maxlen + 1 (to have inputs and targets)
        # If we take maxlen+1 items: i0, i1, ... i_maxlen
        # Input: i0...i_{maxlen-1}
        # Target: i1...i_maxlen
        if seq_len > self.maxlen:
            # Take last maxlen items as input.
            # So we need last maxlen+1 items from history to form pairs.
            # Wait, if we take last maxlen items as input, we predict the ones after them?
            # No, we predict the next item at each step.
            # So we typically use the last `maxlen` items as input sequence.
            # And the target is the same sequence shifted?
            
            # Re-reading standard SASRec data loader:
            # It takes the sequence of user u.
            # It constructs pos and neg sequences.
            # ts = set(train_items)
            # for i in train_items:
            #    seq.append(i)
            #    pos.append(next_item)
            #    neg.append(random_neg)
            
            # It seems it iterates over the sequence.
            # To keep it simple and vectorized:
            
            input_ids = train_items[:-1]
            target_ids = train_items[1:]
            
            # Truncate to maxlen
            if len(input_ids) > self.maxlen:
                input_ids = input_ids[-self.maxlen:]
                target_ids = target_ids[-self.maxlen:]
            else:
                # Pad
                pad_len = self.maxlen - len(input_ids)
                input_ids = [0] * pad_len + input_ids
                target_ids = [0] * pad_len + target_ids
                
        else: # seq_len <= maxlen
             # Pad
            input_ids = train_items[:-1]
            target_ids = train_items[1:]
            
            pad_len = self.maxlen - len(input_ids)
            input_ids = [0] * pad_len + input_ids
            target_ids = [0] * pad_len + target_ids

        # Negative sampling
        neg_ids = []
        ts = set(train_items)
        for _ in range(self.maxlen):
            t = np.random.randint(1, self.itemnum + 1)
            while t in ts:
                t = np.random.randint(1, self.itemnum + 1)
            neg_ids.append(t)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(neg_ids, dtype=torch.long)
        )


class SASRecEvalDataset(Dataset):
    """
    Dataset for validation / test evaluation.
    Each item corresponds to one user, its interaction history (from user_train)
    and the target item from user_valid or user_test.
    """

    def __init__(self, user_train, user_split, usernum, itemnum, maxlen):
        self.user_train = user_train
        self.user_split = user_split
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen

        # users that have both train history and at least one item in the split
        self.users = [
            u
            for u in range(1, usernum + 1)
            if len(user_train[u]) > 0 and len(user_split[u]) > 0
        ]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_train[user]
        target_items = self.user_split[user]
        target_item = target_items[0]

        seq_len = len(seq)
        if seq_len >= self.maxlen:
            seq = seq[-self.maxlen :]
        else:
            seq = [0] * (self.maxlen - seq_len) + seq

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(target_item, dtype=torch.long),
        )


