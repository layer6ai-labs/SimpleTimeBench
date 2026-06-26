from collections.abc import Iterable

import pandas as pd
import torch
from utilsforecast.processing import make_future_dataframe


class TimeSeriesDataset:
    def __init__(
        self,
        data: torch.Tensor,
        uids: Iterable,
        last_times: Iterable,
        batch_size: int,
    ):
        self.data = data
        self.uids = uids
        self.last_times = last_times
        self.batch_size = batch_size
        self.n_batches = len(data) // self.batch_size + (
            0 if len(data) % self.batch_size == 0 else 1
        )
        self.current_batch = 0

    @classmethod
    def from_df(cls, df: pd.DataFrame, batch_size: int, target_cols: list[str] = ['y']):
        tensors = []
        df_sorted = df.sort_values(by=["unique_id", "ds"])
        for _, group in df_sorted.groupby("unique_id"):
            tensors.append(torch.tensor(group[target_cols].values, dtype=torch.bfloat16))
        uids = df_sorted["unique_id"].unique()
        last_times = df_sorted.groupby("unique_id")["ds"].tail(1)
        return cls(tensors, uids, last_times, batch_size)

    def __len__(self):
        return self.n_batches

    def make_future_dataframe(self, h: int, freq: str) -> pd.DataFrame:
        n_dims = self.data[0].shape[1]
        if n_dims == 1:
            return make_future_dataframe(
                uids=self.uids,
                last_times=pd.to_datetime(self.last_times),
                h=h,
                freq=freq,
            )  # type: ignore
        return make_future_dataframe( # TODO check
            uids=[f"{uid}_dim{dim}" for uid in self.uids for dim in range(n_dims)],
            last_times=pd.to_datetime(self.last_times).repeat(n_dims),
            h=h,
            freq=freq,
        )  # type: ignore

    def __iter__(self):
        self.current_batch = 0  # Reset for new iteration
        return self

    def __next__(self):
        if self.current_batch < self.n_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            self.current_batch += 1
            return self.data[start_idx:end_idx]
        else:
            raise StopIteration
