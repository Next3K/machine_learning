import itertools
import random
import time

import numpy as np
import pandas as pd
from numpy import random, nan
from pandas.core.interchange.dataframe_protocol import DataFrame

from fun import evaluate_predictions

MAX_EPOCHS = 50
BATCH_SIZE = 512


class CrossFilter:
    def __init__(self, train_data: pd.DataFrame, num_features: int, learning_rate: float, lambda_reg: float):
        self.train_data: pd.DataFrame = train_data
        self.F: pd.Dataframe = pd.DataFrame(np.random.normal(0, scale=0.1, size=(train_data.shape[0], num_features)),
                                            columns=[f'f{i}' for i in range(num_features)])
        self.P: pd.Dataframe = pd.DataFrame(
            np.random.normal(0, scale=0.1, size=(num_features + 1, train_data.shape[1])),
            columns=[f'f{i}' for i in range(train_data.shape[1])])
        self.train(lambda_reg, learning_rate, num_features)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def train(self, lambda_reg, learning_rate, num_features):
        for epoch in range(MAX_EPOCHS):
            pass


def grid_search_hyperparams(dataset: pd.DataFrame, hyperparameter_grid) -> (int, float, float):
    train_set, validation_set = split_set(dataset, seed=42, split=0.2)
    best_params, best_acc = None, float('inf')
    i = 0
    I = len(hyperparameter_grid)

    all_expected: [int] = []
    all_predicted: [int] = []

    for params in hyperparameter_grid:
        num_features, learning_rate, lambda_reg = params
        model = CrossFilter(train_set, num_features, learning_rate, lambda_reg)
        predicted: pd.DataFrame = model.predict(validation_set)
        mask = ~validation_set.isna()
        matches = (validation_set[mask] == predicted[mask])
        acc = matches.mean()

        if acc > best_acc:
            all_expected = validation_set[mask].values
            all_predicted = predicted[mask].values
            best_params = params
            best_acc = acc
        i += 1
        print(f"Done testing {i}/{I} -- num_features={num_features}, learning_rate={learning_rate}, reg={lambda_reg}")
    evaluate_predictions(all_expected, all_predicted)

    return best_params


def fill_task_csv(model: CrossFilter):
    task = pd.read_csv("task.csv", sep=';', header=None)
    task.columns = ['id', 'user_id', 'movie_id', "grade"]
    task_copy = task.copy()
    task_copy.drop(['id'], inplace=True)
    task_matrix: DataFrame = task_copy.pivot(index='movie_id', columns='user_id', values='grade')
    predicted: pd.DataFrame = model.predict(task_matrix)
    for i in range(len(task)):
        user_id = str(int(task.iloc[i]['user_id']))
        movie_id = int(task.iloc[i]['movie_id'])
        grade = int(predicted.loc[user_id][movie_id])
        task.at[i, 'grade'] = grade
    task = task.astype(int)
    task.to_csv("submission.csv", index=False, header=False, sep=";")


def split_set(src: pd.DataFrame, seed: int, split: float) -> (pd.DataFrame, pd.DataFrame):
    random.seed(seed)
    src_copy: pd.DataFrame = src.copy()

    subset = pd.DataFrame(index=src_copy.index, columns=src_copy.columns)
    non_nan_indices = src_copy.stack().index
    sampled_indices = random.choice(len(non_nan_indices),
                                    size=int(split * len(non_nan_indices)),
                                    replace=False)
    for idx in sampled_indices:
        row, col = non_nan_indices[idx]
        subset.at[row, col] = src_copy.at[row, col]
        src_copy.at[row, col] = nan
    return src_copy, subset


if __name__ == '__main__':
    start_time = time.time()

    train = pd.read_csv("train.csv", sep=';', header=None)
    train.columns = ['id', 'user_id', 'movie_id', "grade"]
    train = train.drop('id', axis=1)
    train = train.pivot(index='movie_id', columns='user_id', values='grade')

    # Define hyperparameters
    num_features = [8, 11, 19, 21]
    learning_rate = [0.01, 0.02, 0.03, 0.05]
    lambda_reg = [0.01, 0.02, 0.05, 0.1]
    possible_hyperparameters: list[tuple[int, float, float]] = (
        list(itertools.product(num_features, learning_rate, lambda_reg)))

    # Create and train model with best hyperparameters
    num_features, learning_rate, lambda_reg = grid_search_hyperparams(train, possible_hyperparameters)

    model = CrossFilter(train, lambda_reg, learning_rate, num_features)

    fill_task_csv(model)

    end_time = time.time()

    print(f"Task finished in: {end_time - start_time:.2f} seconds")
