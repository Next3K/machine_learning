import itertools
import random
import time

import numpy as np
import pandas as pd
from numpy import random, nan
from pandas.core.interchange.dataframe_protocol import DataFrame

from fun import evaluate_predictions

MAX_EPOCHS = 200


class CrossFilter:
    def __init__(self, train_data: pd.DataFrame, num_features: int, learning_rate: float, lambda_reg: float):
        self.train_data: pd.DataFrame = train_data
        self.Q: pd.Dataframe = pd.DataFrame(np.random.normal(0, size=(train_data.shape[1], num_features)))
        self.P: pd.Dataframe = pd.DataFrame(np.random.normal(0, size=(train_data.shape[0], num_features)))
        self.train(learning_rate=learning_rate, lambda_reg=lambda_reg)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        mask = data.isna()
        data[mask] = self.train_data[mask]
        return data.astype(int)

    def train(self, learning_rate, lambda_reg):
        assert self.Q.shape[1] == self.P.shape[1]
        R = self.train_data.fillna(-1).to_numpy()
        P = self.P.to_numpy()
        Q = self.Q.to_numpy()
        K = self.Q.shape[1]
        Q = Q.T

        for _ in range(MAX_EPOCHS):
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > -1:
                        eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                        for k in range(K):
                            P[i][k] = P[i][k] + learning_rate * (2 * eij * Q[k][j] - lambda_reg * P[i][k])
                            Q[k][j] = Q[k][j] + learning_rate * (2 * eij * P[i][k] - lambda_reg * Q[k][j])

        nP, nQ = P, Q.T
        nR = np.dot(nP, nQ.T)
        rounded = np.round(nR)
        clipped = np.clip(rounded, 0, 5)
        df = pd.DataFrame(clipped, index=self.train_data.index, columns=self.train_data.columns)
        mask = self.train_data.isna()
        self.train_data[mask] = df[mask]


def grid_search_hyperparams(dataset: pd.DataFrame, hyperparameter_grid) -> (int, float, float):
    train_set, validation_set = split_set(dataset, seed=42, split=0.2)
    best_params, best_acc = None, 0
    i = 0
    I = len(hyperparameter_grid)

    all_expected: [int] = []
    all_predicted: [int] = []

    for params in hyperparameter_grid:
        num_features, learning_rate, lambda_reg = params
        model = CrossFilter(train_set, num_features, learning_rate, lambda_reg)
        mask = ~validation_set.isna()
        empty_spots = validation_set.copy()
        empty_spots[:] = pd.NA
        predicted: pd.DataFrame = model.predict(empty_spots)
        matches = (validation_set[mask] == predicted[mask])
        acc = matches.sum().sum() / mask.sum().sum()

        if acc > best_acc:
            for _, row in validation_set[mask].iterrows():
                for value in row.dropna():
                    all_expected.append(value)
            for _, row in predicted[mask].iterrows():
                for value in row.dropna():
                    all_predicted.append(value)
            best_params = params
            best_acc = acc
        i += 1
        print(
            f"Done testing {i}/{I} -- acc={best_acc} -- num_features={num_features}, learning_rate={learning_rate}, reg={lambda_reg}")
    print(evaluate_predictions(all_expected, all_predicted))

    return best_params


def fill_task_csv(model: CrossFilter):
    task = pd.read_csv("task.csv", sep=';', header=None)
    task_copy = task.copy()
    task_copy.columns = ['id', 'user_id', 'movie_id', "grade"]
    task.columns = ['id', 'user_id', 'movie_id', "grade"]
    task_matrix: DataFrame = task.pivot(index='user_id', columns='movie_id', values='grade')
    predicted: pd.DataFrame = model.predict(task_matrix)
    for i in range(len(task)):
        user_id = int(task.iloc[i]['user_id'])
        movie_id = int(task.iloc[i]['movie_id'])
        grade = int(predicted.at[user_id, movie_id])
        task_copy.at[i, 'grade'] = grade
    task_copy = task_copy.astype(int)
    task_copy.to_csv("submission.csv", index=False, header=False, sep=";")


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
    train = train.pivot(index='user_id', columns='movie_id', values='grade')

    # Define hyperparameters
    num_features = [3, 8, 19]
    learning_rate = [0.0001, 0.0003, 0.001]
    lambda_reg = [0.01, 0.05, 0.1]
    possible_hyperparameters: list[tuple[int, float, float]] = (
        list(itertools.product(num_features, learning_rate, lambda_reg)))

    # Create and train model with best hyperparameters
    num_features, learning_rate, lambda_reg = grid_search_hyperparams(train, possible_hyperparameters)
    print(f"Best hyperparameters: num_features={num_features}, learning_rate={learning_rate}, reg={lambda_reg}")

    model = CrossFilter(train, num_features, learning_rate, lambda_reg)

    fill_task_csv(model)

    end_time = time.time()

    print(f"Task finished in: {end_time - start_time:.2f} seconds")
