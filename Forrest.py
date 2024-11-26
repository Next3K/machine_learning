import random
from collections import Counter

import pandas as pd

from Tree import Tree
from Model import Model


class Forrest(Model):
    def __init__(self, num_of_trees: int, data: pd.DataFrame, bootstrap_percent: float):
        assert 0 < bootstrap_percent < 1
        self.num_of_trees = num_of_trees
        self.data = data
        self.bootstrap_number: int = int(bootstrap_percent * len(data))
        self.trees: [Tree] = self.create_trees()

    def predict(self, x: pd.DataFrame) -> int:
        return Counter(map(lambda some_tree: some_tree.predict(x), self.trees)).most_common(1)[0][0]

    def create_trees(self):
        trees = []
        for _ in range(self.num_of_trees):
            mask = [1 if i in random.sample(range(7), random.randint(2, 5)) else 0 for i in range(7)]
            columns_to_select = [i for i, m in enumerate(mask) if m == 1]
            data_less_cols = self.data.iloc[:, columns_to_select]
            data_subset = data_less_cols.sample(n=self.bootstrap_number, replace=True)
            trees.append(Tree(5, 7, 10, 'entropy', data_subset))
        return trees
