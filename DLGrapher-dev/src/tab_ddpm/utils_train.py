import numpy as np
import os
from typing import Optional
import tab_ddpm.lib as lib
from tab_ddpm.modules import MLPDiffusion, ResNetDiffusion

def get_model(
    model_name,
    model_params,
    # n_num_features,
    # category_sizes
):
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise NotImplementedError("Unknown model!")
    return model

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def make_dataset(
    data_path: str,
    T: lib.Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool
):
    # classification
    if num_classes > 0:
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {}

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t

    info = lib.load_json(os.path.join(data_path, 'info.json'))

    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = lib.change_val(D)

    return lib.transform_dataset(D, T, None)

def make_dataset_from_memory(
    X_cat: Optional[dict[str, np.ndarray]],
    X_num: Optional[dict[str, np.ndarray]],
    y: dict[str, np.ndarray],
    info: dict,
    T: lib.Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool
):
    if num_classes > 0:
        # classification
        X_cat_ = {} if X_cat or not is_y_cond else None
        X_num_ = {} if X_num else None

        for split in ['train', 'val', 'test']:
            X_num_t = X_num[split] if X_num else None
            X_cat_t = X_cat[split] if X_cat else None
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y[split])
            if X_num_ is not None:
                X_num_[split] = X_num_t
            if X_cat_ is not None:
                X_cat_[split] = X_cat_t
    else:
        # regression
        X_cat_ = {} if X_cat else None
        X_num_ = {} if X_num or not is_y_cond else None

        for split in ['train', 'val', 'test']:
            X_num_t = X_num[split] if X_num else None
            X_cat_t = X_cat[split] if X_cat else None
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y[split])
            if X_num_ is not None:
                X_num_[split] = X_num_t
            if X_cat_ is not None:
                X_cat_[split] = X_cat_t

    D = lib.Dataset(
        X_num_,
        X_cat_,
        y,
        y_info={},
        task_type=lib.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = lib.change_val(D)

    return lib.transform_dataset(D, T, None)
