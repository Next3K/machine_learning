import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import requests
import random
from datetime import datetime
import json
import time
import csv
import os
import json
import pandas as pd
from pandas import Series
from pandas.core.interchange.dataframe_protocol import DataFrame
from fun import evaluate_predictions
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import statistics
from typing import Dict, List
from Knn import Knn

API_KEY = "api-key"


def scrap_to_json():
    data = pd.read_csv("movie.csv", sep=";", header=None)
    data.columns = ['1', '2', '3']
    ids = data['2'].to_list()

    url_template = "https://api.themoviedb.org/3/movie/{}?api_key=" + API_KEY
    headers = {
        "accept": "application/json"
    }

    i = 0
    for movie_id in ids:
        url = url_template.format(movie_id)
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            movie_data = response.json()
            i += 1

            movie_data['movie_id'] = i

            file_path = os.path.join("movies", f"movie_{i}.json")
            with open(file_path, "w") as json_file:
                json.dump(movie_data, json_file, indent=4)

            print(f"Saved data for movie ID {i}")
            time.sleep(0.2)
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error for movie ID {i}: {http_err}")
        except Exception as err:
            print(f"Error for movie ID {i}: {err}")


def extract_from_json(directory: str):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as file:
                movie_data = json.load(file)
                movie_id = movie_data.get('movie_id')
                genres = [genre['name'] for genre in movie_data.get("genres", [])]
                budget = movie_data.get("budget")
                popularity = movie_data.get("popularity")
                production_companies = [company['name'] for company in movie_data.get("production_companies", [])]
                production_countries = [country['name'] for country in movie_data.get("production_countries", [])]
                release_date = movie_data.get("release_date")
                revenue = movie_data.get("revenue")
                runtime = movie_data.get("runtime")
                vote_average = movie_data.get("vote_average")
                vote_count = movie_data.get("vote_count")
                overview = movie_data.get("overview")

                data.append({
                    "movie_id": movie_id,
                    "genres": ", ".join(genres),
                    "budget": budget,
                    "popularity": popularity,
                    "production_companies": ", ".join(production_companies),
                    "production_countries": ", ".join(production_countries),
                    "release_date": datetime.strptime(release_date, "%Y-%m-%d").year,
                    "revenue": revenue,
                    "runtime": runtime,
                    "vote_average": vote_average,
                    "vote_count": vote_count,
                    "overview": overview
                })

    df = pd.DataFrame(data)
    df.to_csv("extracted_movies.csv", index=False)


def fill_task_csv(model: {int, Knn}):
    task = pd.read_csv("task.csv", sep=';', header=None)
    task.columns = ['id', 'user_id', 'movie_id', "grade"]
    movie_vectors = pd.read_csv("vector.csv")
    task_features = task.merge(movie_vectors, on='movie_id', how='left')
    task_features.drop(columns=['id', 'grade', 'user_id', 'movie_id'], inplace=True)

    for index, row in task.iterrows():
        user_id = row['user_id'].astype(int)
        task.at[index, 'grade'] = model[user_id].predict(task_features.iloc[index])

    task = task.astype(int)
    task.to_csv("submission.csv", index=False, header=False, sep=";")


def train_test_split(data, test_size=0.2, random_state=None):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    return train_data, test_data


if __name__ == '__main__':
    # scrap_to_json()
    # extract_from_json("movies")

    raw_train_data = pd.read_csv("train.csv", sep=';', header=None)
    raw_train_data.columns = ['id', 'user_id', 'movie_id', "grade"]

    full_train_data = raw_train_data.merge(pd.read_csv("vector.csv"), on='movie_id', how='left')
    full_train_data, validate = train_test_split(full_train_data, test_size=0.2, random_state=42)

    user_dataframe_map = {user_id:
                              group_df for user_id, group_df
                          in full_train_data.groupby('user_id')}

    for user_id, dataframe in user_dataframe_map.items():
        dataframe.drop(columns=['id', 'user_id', 'movie_id'], inplace=True)


    # Parallelized Function for Each User
    def process_user(user_id, dataframe, k_values, iterations):
        best_knn = None
        best_knn_testwise = None
        best_score = 0.0

        for k in k_values:
            for i in range(iterations):
                mask = [1 if idx in random.sample(range(12), random.randint(3, 8)) else 0 for idx in range(12)]
                dfs = np.array_split(dataframe, 5)

                for use_average in [True, False]:
                    scores = []
                    tmp_knn = None

                    for n in range(1):  # Simulate one test portion
                        test_portion = dfs[n]
                        train_portion = pd.concat(dfs[:n] + dfs[n + 1:], ignore_index=True)

                        knn = Knn(k=k, mask=mask, use_average=use_average, dataset=train_portion)
                        tmp_knn = knn
                        total_elems, correct = len(test_portion), 0

                        for z in range(total_elems):
                            row = test_portion.iloc[z, 1:]
                            expected = test_portion.iloc[z]['grade']
                            predicted = knn.predict(row)
                            if predicted == expected:
                                correct += 1

                        scores.append(correct / total_elems)

                    current_score = statistics.mean(scores)
                    if current_score > best_score:
                        best_score = current_score
                        best_knn_testwise = tmp_knn
                        best_knn = Knn(k=k, mask=mask, use_average=use_average, dataset=dataframe)

        return user_id, best_knn, best_knn_testwise


    KNNs: Dict[int, Knn] = {}
    KNNs_testwise: Dict[int, Knn] = {}
    k_values = [1, 3, 5, 7, 11]
    iterations = 3

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_user, user_id, dataframe, k_values, iterations)
            for user_id, dataframe in user_dataframe_map.items()
        ]

        for future in as_completed(futures):
            user_id, best_knn, best_knn_testwise = future.result()
            print(
                f"User {user_id} finished: k={best_knn.k} -- use_average={best_knn.use_average} -- score=0.3333333333333333")
            KNNs[user_id] = best_knn
            KNNs_testwise[user_id] = best_knn_testwise

    print("All users processed.")

    # print(f"+++++++++++++++++++VALIDATION RESULTS+++++++++++++++++++")
    # predicted = []
    # expected = []
    # validate.drop(columns=['id', 'movie_id'], inplace=True)
    # for z in range(len(validate)):
    #     row = validate.iloc[z, 2:]
    #     user_id = validate.iloc[z]['user_id']
    #     exp = validate.iloc[z]['grade']
    #     pred = KNNs[user_id].predict(row)
    #     expected.append(exp)
    #     predicted.append(pred)
    # print(evaluate_predictions(expected, predicted))
    #
    # print(f"+++++++++++++++++++TESTWISE RESULTS+++++++++++++++++++")
    # predicted = []
    # expected = []
    # full_train_data.drop(columns=['id', 'movie_id'], inplace=True)
    # _, test_portion = train_test_split(full_train_data, test_size=0.2, random_state=42)
    # for z in range(len(test_portion)):
    #     row = test_portion.iloc[z, 2:]
    #     user_id = test_portion.iloc[z]['user_id']
    #     exp = test_portion.iloc[z]['grade']
    #     pred = KNNs_testwise[user_id].predict(row)
    #     expected.append(exp)
    #     predicted.append(pred)
    # print(evaluate_predictions(expected, predicted))

    # fill task.csv
    # fill_task_csv(KNNs)

    print("Done!")
