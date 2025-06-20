from math import ceil
from typing import cast

import numpy as np
import pandas as pd
import pandas.api.types as pd_dts
import torch

from src.datasets.abstract_dataset import TableGraphDatasetInfos
from tab_ddpm.sample import sample
from tab_ddpm.train import train

T = {
    # "seed": 0,
    "normalization": "minmax",
    "num_nan_policy": None,
    "cat_nan_policy": None,
    "cat_min_frequency": None,
    "cat_encoding": "one-hot",
    "y_policy": "default"
}

MODEL_RTDL_PARAMS = {
    "d_layers": [
        1024,
        1024,
        1024
    ],
    "dropout": 0.0
}


# Currently assumes that target column y is categorical
class TabDDPM:
    def __init__(
            self,
            dataset_infos: TableGraphDatasetInfos,
            y_col: str,
            epochs=300) -> None:
        self.tti = dataset_infos.tti
        self.y_col = y_col
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if y_col in self.tti.bool_cols:
            self.num_classes = 2
        else:
            self.num_classes = len(self.tti.dtypes[y_col].categories)
        self.cat_feats: list[str] = []
        self.num_feats: list[str] = []

        for col, c_dtype in self.tti.dtypes.items():
            if col == self.y_col:
                continue
            if pd_dts.is_bool_dtype(c_dtype) or pd_dts.is_categorical_dtype(c_dtype):
                self.cat_feats.append(cast(str, col))
            elif pd_dts.is_numeric_dtype(c_dtype) or pd_dts.is_datetime64_dtype(c_dtype):
                self.num_feats.append(cast(str, col))

        self.feats_cat_num = self.cat_feats + self.num_feats
        y_col_dtype = self.tti.dtypes[self.y_col]
        self.y_is_cat = pd_dts.is_categorical_dtype(y_col_dtype)

        if self.y_is_cat and len(y_col_dtype.categories) > 2:
            self.task_type = 'multiclass'
        else:
            self.task_type = 'binclass'

        self.model_params = {
            "num_classes": self.num_classes,
            "is_y_cond": True,
            "rtdl_params": MODEL_RTDL_PARAMS
        }
        self.not_fit = True

    def fit(
            self,
            train_data: pd.DataFrame,
            val_data: pd.DataFrame,
            test_data: pd.DataFrame,
            seed: int,
            batch_size=4096) -> None:
        astype_mapping = {dt: 'int' for dt in self.tti.dt_cols}
        train_data_ = train_data.astype(astype_mapping, copy=True)
        val_data_ = val_data.astype(astype_mapping, copy=True)
        test_data_ = test_data.astype(astype_mapping, copy=True)

        if self.y_is_cat:
            train_data_[self.y_col] = train_data_[self.y_col].cat.codes
            val_data_[self.y_col] = val_data_[self.y_col].cat.codes
            test_data_[self.y_col] = test_data_[self.y_col].cat.codes

        self.X_cat = {
            "train": train_data_[self.cat_feats].to_numpy(),
            "val": val_data_[self.cat_feats].to_numpy(),
            "test": test_data_[self.cat_feats].to_numpy()
        } if len(self.cat_feats) else {}

        self.X_num = {
            "train": train_data_[self.num_feats].to_numpy(),
            "val": val_data_[self.num_feats].to_numpy(),
            "test": test_data_[self.num_feats].to_numpy()
        } if len(self.num_feats) else {}

        self.y = {
            "train": train_data_[self.y_col].to_numpy(),
            "val": val_data_[self.y_col].to_numpy(),
            "test": test_data_[self.y_col].to_numpy()
        }

        self.info = {"task_type": self.task_type, "n_classes": self.num_classes}
        t_dict = T.copy()
        t_dict["seed"] = seed

        self.diffusion_model, _ = train(
            steps=self.epochs * ceil(len(train_data_) / batch_size),
            # lr=0.002,
            # weight_decay=0.0001,
            batch_size=batch_size,

            # num_timesteps=1000,
            # gaussian_loss_type="mse",

            X_cat=self.X_cat,
            X_num=self.X_num,
            y=self.y,
            info=self.info,

            # model_type="mlp",

            model_params=self.model_params,

            T_dict=t_dict,

            num_numerical_features=len(self.num_feats),

            device=self.device,

            # seed=0,

            change_val=False
        )

        self.not_fit = False

    def sample(self, num_rows: int, batch_size=4096) -> pd.DataFrame:
        if self.not_fit:
            raise AssertionError("Can't sample before fitting the model.")

        X_cat_gen, X_num_gen, y_gen = sample(
            num_samples=num_rows,

            batch_size=batch_size,

            # disbalance=None,

            # num_timesteps=1000,
            # gaussian_loss_type="mse",

            X_cat_reals=self.X_cat,
            X_num_reals=self.X_num,
            y_reals=self.y,
            info=self.info,

            model=self.diffusion_model,

            model_params=self.model_params,

            T_dict=T,

            num_numerical_features=len(self.num_feats),

            device=self.device,

            # seed=0,

            change_val=False
        )

        rows = pd.DataFrame(
            np.concatenate((X_cat_gen, X_num_gen), axis=-1),
            columns=self.feats_cat_num)

        if self.y_is_cat:
            rows[self.y_col] = pd.Categorical.from_codes(y_gen, dtype=self.tti.dtypes[self.y_col])
        else:
            rows[self.y_col] = y_gen

        rows = rows[self.tti.dtypes.keys()]
        rows = rows.astype(self.tti.dtypes)

        for dt_col in self.tti.dt_cols:
            rows[dt_col] = rows[dt_col].dt.normalize()

        return rows
