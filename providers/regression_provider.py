from .provider import Provider

import numpy as np


class RegressionProvider(Provider):
    def __init__(self, df, batch_size, shuffle=False):
        super().__init__()
        self._df = df
        self._batch_size = batch_size
        self._pos = 0
        self._shuffle = True 

    def __iter__(self):
        self._pos = 0
        if self._shuffle:
            self._df = self._df.sample(frac=1).reset_index(drop=True)
        return self

    def __next__(self):
        samples = len(self._df)
        if self._pos >= samples:
            raise StopIteration

        current_pos = self._pos
        next_pos = min(current_pos + self._batch_size, samples)
        self._pos = next_pos
        return (self._df['x'].iloc[current_pos:next_pos].values, 
            np.expand_dims(self._df['y'].iloc[current_pos:next_pos].values, axis=1))
