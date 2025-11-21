import torch
from torch.utils.data import DataLoader
from .dataset import load_movielens, data_partition, SASRecDataset, SASRecEvalDataset

def create_dataloaders(config, seed=None):
    """
    Create dataloaders for SASRec training, validation and testing.

    Returns
    -------
    train_loader : DataLoader
        Loader over SASRecDataset built from user_train.
    val_loader : DataLoader
        Loader over SASRecEvalDataset using (user_train, user_valid).
    test_loader : DataLoader
        Loader over SASRecEvalDataset using (user_train, user_test).
    dataset_split : list
        [user_train, user_valid, user_test, usernum, itemnum] for
        compatibility with existing evaluation code.
    """
    if seed is not None:
        torch.manual_seed(seed)

    data_path = config["data"]["path"]
    batch_size = config["training"]["batch_size"]
    eval_batch_size = config["training"].get("eval_batch_size", batch_size)
    maxlen = config["model"]["maxlen"]
    num_workers = config["training"].get("num_workers", 4)

    df = load_movielens(data_path)
    dataset_split = data_partition(df)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset_split

    train_ds = SASRecDataset(user_train, usernum, itemnum, maxlen, train=True)
    val_ds = SASRecEvalDataset(user_train, user_valid, usernum, itemnum, maxlen)
    test_ds = SASRecEvalDataset(user_train, user_test, usernum, itemnum, maxlen)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, dataset_split

