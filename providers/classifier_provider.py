from .provider import Provider

import numpy as np


class ClassifierProvider(Provider):
    def __init__(self, df, batch_size, shuffle=True):
        super().__init__()
        self._df = df
        self._classes = self._df['cls'].nunique()
        self._batch_size = batch_size
        self._pos = 0
        self._shuffle = shuffle

        self.number_of_inputs = 2
        self.number_of_outputs = self._classes

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
        x = self._df[['x', 'y']].iloc[current_pos:next_pos].values
        labels = self._df['cls'].iloc[current_pos:next_pos]
        y = self.one_hot_encoding(labels)
        return x, y

    def __len__(self):
        return len(self._df)

    def one_hot_encoding(self, labels):
        labels = labels.apply(lambda x: x-1).values
        y = np.zeros(shape=(len(labels), self._classes))
        y[np.arange(len(labels)), labels] = 1
        return y

