import random
import numpy as np
from pandas import Series
from pandas.core.interchange.dataframe_protocol import DataFrame
from fun import calculate_total_distances, get_average, get_label


class Knn:
    def __init__(self,
                 k: int,
                 mask: [int],
                 use_average: bool,
                 dataset: DataFrame):
        self.mask = mask
        self.k = k
        self.use_average = use_average
        # first column is a grade, the next columns are (12) features of the movie
        # grade, budget, popularity, release_date, revenue, runtime, vote_average, vote_count, votes, genres, production_companies, production_countries, overview
        self.dataset = dataset

    # series is just a single row from dataframe. The columns (12) are:
    # budget, popularity, release_date, revenue, runtime, vote_average, vote_count, votes, genres, production_companies, production_countries, overview
    # return expected grade
    def predict(self, x: Series) -> int:
        distances = calculate_total_distances(x, self.dataset, self.mask)
        idx = np.argsort(distances)[:self.k]
        print(idx)
        if self.use_average:
            print("Approximating Average Value:")
            return get_average(idx, self.dataset)
        print("Approximating Label:")
        return get_label(idx, self.dataset)
