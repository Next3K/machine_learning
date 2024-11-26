from typing import List

import numpy as np
import pandas as pd

from fun import get_measure
from main import NUM_OF_PARTITIONS
from Model import Model


class Tree(Model):
    def __init__(self,
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 criterion: str,
                 dataframe: pd.DataFrame,
                 depth=0):

        # node data
        self.column: str = ""
        self.value: float = 0
        self.left: Tree = None
        self.right: Tree = None
        self.predicted_class: int = None
        self.is_leaf: bool = True

        # construct the tree
        if depth <= max_depth and len(dataframe) >= min_samples_split:
            columns_min_max_map: {str, (float, float)} = {}
            for column in dataframe.columns:
                min_val = dataframe[column].min()
                max_val = dataframe[column].max()
                columns_min_max_map[column] = (min_val, max_val)

            all_decision_variants: {str, [float]} = {}
            for column, (min_val, max_val) in columns_min_max_map.items():
                evenly_spaced_values = np.linspace(min_val, max_val, NUM_OF_PARTITIONS).tolist()
                all_decision_variants[column] = evenly_spaced_values

            all_combinations: (str, float) = [(column, value) for column, values in all_decision_variants.items() for value in values]

            all_combinations_with_measure: List[((str, float), float)] = \
                [((col_name, decision_value),
                  (get_measure(dataframe=dataframe, column=col_name, decision=decision_value, criterion=criterion)))
                 for (col_name, decision_value) in all_combinations]
            decision_column, decision_value = min(all_combinations_with_measure, key=lambda x: x[1])[0]
            right_dataframe = dataframe[dataframe[decision_column] >= decision_value]
            left_dataframe = dataframe[dataframe[decision_column] < decision_value]

            if len(left_dataframe) >= min_samples_leaf and len(right_dataframe) >= min_samples_leaf:
                self.is_leaf = False
                self.value, self.column = decision_value, decision_column
                self.left = Tree(max_depth=max_depth, min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf, criterion=criterion,
                                 dataframe=left_dataframe, depth=depth + 1)
                self.right = Tree(max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf, criterion=criterion,
                                  dataframe=right_dataframe, depth=depth + 1)
                self.caption = f'{self.column} < {self.value}>'

        if self.is_leaf:
            self.predicted_class = dataframe['grade'].value_counts().idxmax()
            self.caption = f'grade: {self.predicted_class}'

    def predict(self, x: pd.DataFrame) -> int:
        if len(x) != 1:
            raise ValueError("Input DataFrame x must have exactly one row")
        if self.is_leaf:
            return self.predicted_class
        return self.left.predict(x) if x[self.column].iloc[0] < self.value else self.right.predict(x)
