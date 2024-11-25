import math
import statistics
from collections import Counter
from graphviz import Digraph
import numpy as np
from typing import List
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
from abc import ABC, abstractmethod

API_KEY = "api-key"

class Model(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> int:
        pass

class Tree(Model):
    def __init__(self,
                 max_tree_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 criterion: str,
                 data: pd.DataFrame):
        self.max_tree_depth = max_tree_depth
        self.min_samples_split = min_samples_split
        self.data = data
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        # TODO create actuall Tree
        self.value: str = "test value"
        self.children: [Tree] = []

    def predict(self, x: pd.DataFrame) -> int:
        return 2


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


def print_tree(root: Tree, filename):
    graph = Digraph(format='png')

    def add_nodes_edges(node):
        if not node:
            return
        for child in node.children:
            graph.edge(str(node.value), str(child.value))
            add_nodes_edges(child)

    graph.node(str(root.value))
    add_nodes_edges(root)
    graph.render(filename, cleanup=True)


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
        task.at[index, 'grade'] = model_per_user[user_id].predict(pd.DataFrame([task_features.iloc[index]]))

    task = task.astype(int)
    task.to_csv(out_name, index=False, header=False, sep=";")


def get_score(test_data: [Series], model: Model) -> float:
    total_elems, correct = len(test_data), 0
    for k in range(total_elems):
        row_dataframe: pd.DataFrame = pd.DataFrame([test_data.iloc[k, 1:]])
        predicted = test_data.iloc[k]['grade']
        if model.predict(row_dataframe) == predicted:
            correct += 1
    return correct / total_elems


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
        dataframe.drop(columns=['id', 'user_id', 'release_date', 'movie_id', 'genres', 'production_companies',
                                'production_countries', 'overview'], inplace=True)

    # find the best possible knn for every user
    best_tree_per_user: {int, Tree} = {}
    for num, (user_id, dataframe) in enumerate(user_dataframe_map.items(), 1):
        print(f"Training tree: {num}/358")
        best_tree: Tree = None
        best_score: float = .0
        for max_tree_depth in [3, 7, 10, 13]:
            for min_samples_split in [5, 7, 10, 13]:
                for min_samples_leaf in [5, 7, 10, 13]:
                    for criterion in ["gemini", "entropy"]:
                        n = len(dataframe) // 5
                        dfs: List[pd.DataFrame] = [
                            dataframe[:n],
                            dataframe[n: 2 * n],
                            dataframe[2 * n: 3 * n],
                            dataframe[3 * n: 4 * n],
                            dataframe[4 * n:],
                        ]
                        scores: [float] = []
                        for n in range(5):
                            test_portion = dfs[n]
                            train_portion = pd.concat(dfs[:n] + dfs[n + 1:], ignore_index=True)
                            tree = Tree(max_tree_depth, min_samples_split, min_samples_leaf, criterion, train_portion)
                            scores.append(get_score(test_portion, tree))
                        current_score = statistics.mean(scores)

                        if current_score > best_score:
                            best_score, best_tree = current_score, Tree(max_tree_depth,
                                                                        min_samples_split,
                                                                        min_samples_leaf,
                                                                        criterion,
                                                                        dataframe)

        best_tree_per_user[user_id] = best_tree
    fill_csv(best_tree_per_user, "submission_tree.csv")
    print("Trees done!")
    print_tree(best_tree_per_user[10], "tree.png")

    # find the best possible forrest for every user
    best_forrest_per_user: {int, Model} = {}
    for num, (user_id, dataframe) in enumerate(user_dataframe_map.items(), 1):
        print(f"Training forrest: {num}/358")
        best_forrest: Forrest = None
        best_score: float = .0
        # hyperparam tuning
        for number_of_trees in [7, 11, 19, 31]:
            for bootstrap_percent in [20, 30, 40, 50, 60, 80]:
                n = len(dataframe) // 5
                dfs: List[pd.DataFrame] = [
                    dataframe[:n],
                    dataframe[n: 2 * n],
                    dataframe[2 * n: 3 * n],
                    dataframe[3 * n: 4 * n],
                    dataframe[4 * n:],
                ]
                scores: [float] = []
                for n in range(5):
                    test_portion: pd.DataFrame = dfs[n]
                    train_portion: pd.DataFrame = pd.concat(dfs[:n] + dfs[n + 1:], ignore_index=True)
                    forrest = Forrest(
                        num_of_trees=number_of_trees,
                        data=train_portion,
                        bootstrap_percent=bootstrap_percent)
                    scores.append(get_score(test_portion, forrest))

                current_score = statistics.mean(scores)
                if current_score > best_score:
                    best_score, best_forrest = current_score, Forrest(num_of_trees=number_of_trees, data=dataframe)

        best_forrest_per_user[user_id] = best_forrest
    fill_csv(best_forrest_per_user, "submission_forrest.csv")
    print("Forrests done!")
