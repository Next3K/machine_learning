from Model import Model
import numpy as np
import pandas as pd
from Tree import Tree
from typing import List
from collections import Counter
import random

class Forrest(Model):
    def __init__(self, num_of_trees: int, data: pd.DataFrame, bootstrap_percent: float):
        assert 0 < (bootstrap_percent / 100) < 1
        self.bootstrap_percent = bootstrap_percent
        self.num_of_trees = num_of_trees
        self.data = data
        self.bootstrap_number: int = int((bootstrap_percent / 100) * len(data))
        self.trees: [Tree] = self.create_trees()

    def predict(self, x: pd.DataFrame) -> int:
        return Counter(map(lambda some_tree: some_tree.predict(x), self.trees)).most_common(1)[0][0]

    def create_trees(self):
        trees = []
        for _ in range(self.num_of_trees):
            def generate_bit_mask(length=6, num_ones=3):
                bit_mask = [0] * length
                ones_positions = random.sample(range(length), num_ones)
                for pos in ones_positions:
                    bit_mask[pos] = 1
                return bit_mask

            mask: [int] = [1] + generate_bit_mask(6, 3)
            columns_to_select = [i for i, m in enumerate(mask) if m == 1]
            if len(columns_to_select) == 1:
                exit(-1)
            data_less_cols = self.data.iloc[:, columns_to_select]
            data_subset = data_less_cols.sample(n=(len(self.data) + self.bootstrap_number), replace=True)
            trees.append(Tree(5, 7, 3, 'entropy', data_subset))
        return trees
