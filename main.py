import concurrent.futures
import heapq
import os
import random
import pandas as pd
import numpy as np


def fill_task_csv(df: pd.DataFrame, hyperparams_per_user: {str: (int, str, str)}):
    task = pd.read_csv("task.csv", sep=';', header=None)
    task.columns = ['id', 'user_id', 'movie_id', "grade"]
    for i in range(len(task)):
        user_id = str(int(task.iloc[i]['user_id']))
        movie_id = str(int(task.iloc[i]['movie_id']))
        k, measure, strategy = hyperparams_per_user[user_id]
        predicted: int = grade(user_id, movie_id, df, k, measure, strategy)
        task.loc[i, "grade"] = predicted
    task = task.astype(int)
    task.to_csv("submission.csv", index=False, header=False, sep=";")


def choice(grades: [int], strategy: str) -> int:
    if strategy == 'mean':
        return np.mean(grades)
    elif strategy == 'dominant':
        values, counts = np.unique(grades, return_counts=True)
        return values[np.argmax(counts)]
    elif strategy == 'random':
        return random.choice(grades)
    else:
        raise ValueError(f"Bad input: f{strategy} is not a valid strategy.")


def euclidean(row1, row2):
    assert len(row1) == len(row2)
    dimensions = len(row1)
    valid_indices = ~np.isnan(row1) & ~np.isnan(row2)
    valid_row1 = row1[valid_indices]
    valid_row2 = row2[valid_indices]
    distance = np.sqrt(np.sum((valid_row1 - valid_row2) ** 2))
    normalization = np.sqrt(dimensions / len(valid_row1))
    return distance * normalization


def manhattan(row1, row2):
    assert len(row1) == len(row2)
    dimensions = len(row1)
    valid_indices = ~np.isnan(row1) & ~np.isnan(row2)
    valid_row1 = row1[valid_indices]
    valid_row2 = row2[valid_indices]
    distance = np.sum(np.abs(valid_row1 - valid_row2))
    normalization = np.sqrt(dimensions / len(valid_row1))
    return distance * normalization


def grade(user_id: str, movie_id: str, df: pd.DataFrame, k: int, measure: str, strategy: str) -> int:
    df.columns = df.columns.astype(str)
    df.index = df.index.astype(str)
    df = df.T
    my_row = df.loc[user_id]
    df = df[(df[movie_id].notna())]
    distances = []
    for idx, row in df.iterrows():
        if idx != user_id:
            distances.append((euclidean(my_row, row) if measure == 'euclidean' else manhattan(my_row, row), idx))
    closest_rows = heapq.nsmallest(k, distances, key=lambda x: x[0])
    closest_indices = [idx for _, idx in closest_rows]
    grades = df.loc[closest_indices][movie_id].tolist()
    return choice(grades, strategy)


def get_score(user_id: str, train_data: pd.DataFrame, test_data: pd.DataFrame, k: int, measure: str,
              strategy: str) -> int:
    total_elems, correct = len(test_data), 0
    for i in range(total_elems):
        should_be: int = test_data.iloc[i][user_id]
        movie_id: str = test_data.index[i]
        if grade(user_id, movie_id, train_data, k, measure, strategy) == should_be:
            correct += 1
    return correct / total_elems


def hyperparams_for_user(user_id: str, dataframe: pd.DataFrame) -> (int, str, str):
    best_k, best_metric, best_strategy = 1, None, None
    best_score: float = 0.0

    for k in [1, 7, 37, 97]:
        for measure in ["manhattan", "euclidean"]:
            for strategy in ["mean", "dominant", "random"]:
                current_score = get_score(user_id, dataframe, dataframe.dropna(subset=[user_id]), k, measure, strategy)
                if current_score > best_score:
                    best_score = current_score
                    best_k, best_metric, best_strategy = k, measure, strategy
    print(f"User {user_id} finished: k={k} -- measure={measure} -- strategy={strategy} -- score={best_score}")
    return best_k, best_metric, best_strategy


if __name__ == '__main__':
    train = pd.read_csv("train.csv", sep=';', header=None)
    train.columns = ['id', 'user_id', 'movie_id', "grade"]
    train = train.drop('id', axis=1)
    train = train.pivot(index='movie_id', columns='user_id', values='grade')
    train.index = train.index.astype(str)
    train.columns = train.columns.astype(str)

    futures = {}
    ids = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for user_id in train.columns:
            futures[user_id] = executor.submit(hyperparams_for_user, user_id, train)

        hyperparams_per_user = {}
        for future in concurrent.futures.as_completed(futures.values()):
            user_id = next(key for key, value in futures.items() if value == future)
            hyperparams_per_user[user_id] = future.result()

    fill_task_csv(train, hyperparams_per_user)
