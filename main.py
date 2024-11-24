import math
import statistics
from collections import Counter

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
from Knn import Knn
from abc import ABC, abstractmethod

API_KEY = "api-key"

class Model(ABC):
    @abstractmethod
    def predict(self, data: Series) -> int:
        pass


class Tree(Model):
    def __init__(self,
                 max_tree_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 criterion: str,
                 data: DataFrame):
        self.max_tree_depth = max_tree_depth
        self.min_samples_split = min_samples_split
        self.data = data
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

    def predict(self, x: Series) -> int:
        return 2


class Forrest(Model):
    def __init__(self, num_of_trees: int, data: DataFrame):
        self.num_of_trees = num_of_trees
        self.data = data
        self.trees: [Tree] = []

    def predict(self, x: Series) -> int:
        return Counter(map(lambda some_tree: some_tree.predict(x), self.trees)).most_common(1)[0][0]


def scrap_to_json():
    data = pd.read_csv("movie.csv", sep=";", header=None)
    data.columns = ['1', '2', '3']
    ids = data['2'].to_list()

    url_template = "https://api.themoviedb.org/3/movie/?api_key=" + API_KEY
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


def fill_csv(model_per_user: {int, Model}, out_name: str):
    task = pd.read_csv("task.csv", sep=';', header=None)
    task.columns = ['id', 'user_id', 'movie_id', "grade"]
    movie_vectors = pd.read_csv("vector.csv")
    task_features = task.merge(movie_vectors, on='movie_id', how='left')
    task_features.drop(columns=['id', 'grade', 'user_id', 'movie_id'], inplace=True)

    for index, row in task.iterrows():
        user_id = row['user_id'].astype(int)
        task.at[index, 'grade'] = model_per_user[user_id].predict(task_features.iloc[index])

    task = task.astype(int)
    task.to_csv(out_name, index=False, header=False, sep=";")


if __name__ == '__main__':
    # scrap_to_json()
    # extract_from_json("movies")

    raw_train_data = pd.read_csv("train.csv", sep=';', header=None)
    raw_train_data.columns = ['id', 'user_id', 'movie_id', "grade"]

    full_train_data = raw_train_data.merge(pd.read_csv("vector.csv"), on='movie_id', how='left')

    user_dataframe_map = {user_id:
                              group_df for user_id, group_df
                          in full_train_data.groupby('user_id')}

    for user_id, dataframe in user_dataframe_map.items():
        dataframe.drop(columns=['id', 'user_id', 'movie_id'], inplace=True)

    # find the best possible knn for every user
    best_tree_per_user: {int, Tree} = {}
    for num, (user_id, dataframe) in enumerate(user_dataframe_map.items(), 1):
        print(f"Training tree: {num}/358")
        best_tree: Tree = None
        best_score: float = .0
        for max_tree_depth in [5, 7, 10, 13]:
            for min_samples_split in [5, 7, 10, 13]:
                for min_samples_leaf in [5, 7, 10, 13]:
                    for criterion in ["gemini", "entropy"]:
                        dfs: [DataFrame] = np.array_split(dataframe, 5)
                        scores: [float] = []
                        for n in range(5):
                            test_portion = dfs[n]
                            train_portion = dfs[:n] + dfs[n + 1:]
                            tree = Tree(max_tree_depth, min_samples_split, min_samples_leaf, criterion, train_portion)
                            total_elems, correct = len(test_portion), 0
                            for z in range(total_elems):
                                row = test_portion.iloc[z, 1:]
                                predicted = test_portion.iloc[z]['grade']
                                if tree.predict(row) == predicted:
                                    correct += 1
                            scores.append(correct / total_elems)

                        current_score = statistics.mean(scores)
                        if current_score > best_score:
                            best_score = current_score
                            best_tree = Tree(max_tree_depth,
                                             min_samples_split,
                                             min_samples_leaf,
                                             criterion,
                                             train_portion)

        best_tree_per_user[user_id] = best_tree

    fill_csv(best_tree_per_user, "submission_tree.csv")

    # find the best possible forrest for every user
    best_forrest_per_user: {int, Tree} = {}
    for num, (user_id, dataframe) in enumerate(user_dataframe_map.items(), 1):
        print(f"Training forrest: {num}/358")
        best_forrest: Forrest = None
        best_score: float = .0
        for number_of_trees in [3, 5, 7, 9]:
                dfs: [DataFrame] = np.array_split(dataframe, 5)
                scores: [float] = []
                for n in range(5):
                    test_portion = dfs[n]
                    train_portion = dfs[:n] + dfs[n + 1:]
                    forrest = Forrest(num_of_trees=number_of_trees)
                    total_elems, correct = len(test_portion), 0
                    for z in range(total_elems):
                        row = test_portion.iloc[z, 1:]
                        predicted = test_portion.iloc[z]['grade']
                        if forrest.predict(row) == predicted:
                            correct += 1
                    scores.append(correct / total_elems)

                current_score = statistics.mean(scores)
                if current_score > best_score:
                    best_score = current_score
                    best_forrest = best_forrest

        best_forrest_per_user[user_id] = best_forrest
    fill_csv(best_forrest_per_user, "submission_forrest.csv")

    print("Done!")
