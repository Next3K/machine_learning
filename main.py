import statistics
from typing import List
import json
import os
import time
from datetime import datetime
import requests
from graphviz import Digraph
import random
from collections import Counter
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import concurrent.futures
import os

NUM_OF_PARTITIONS: int = 8
API_KEY = "api-key"
NUM_OF_WORKERS = os.cpu_count() or 10


def get_measure(dataframe: pd.DataFrame, column: str, decision: float, criterion: str) -> float:
    measure: float = 0
    left = dataframe[dataframe[column] < decision]
    right = dataframe[dataframe[column] >= decision]
    if criterion == 'entropy':
        measure = (
                (len(left) / len(dataframe)) * entropy(left) +
                (len(right) / len(dataframe)) * entropy(right)
        )
    elif criterion == 'gini':
        measure = (
                (len(left) / len(dataframe)) * gini(left) +
                (len(right) / len(dataframe)) * gini(right)
        )
    return measure


def entropy(sub_df):
    value_counts = sub_df['grade'].value_counts()
    total = len(sub_df)
    if total == 0:
        return 0
    probabilities = value_counts / total
    return -np.sum(probabilities * np.log2(probabilities))


def gini(sub_df):
    value_counts = sub_df['grade'].value_counts()
    total = len(sub_df)
    if total == 0:
        return 0
    probabilities = value_counts / total
    return 1 - np.sum(probabilities ** 2)


class Model(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> int:
        pass


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
        if dataframe.shape[1] == 1:
            print("shape")
        if depth <= max_depth and len(dataframe) >= min_samples_split:
            columns_min_max_map: {str, (float, float)} = {}
            for column in dataframe.columns:
                if column != 'grade':
                    min_val = dataframe[column].min()
                    max_val = dataframe[column].max()
                    columns_min_max_map[column] = (min_val, max_val)

            all_decision_variants: {str, [float]} = {}
            for column, (min_val, max_val) in columns_min_max_map.items():
                evenly_spaced_values = np.linspace(min_val, max_val, NUM_OF_PARTITIONS).tolist()[1:-1]
                all_decision_variants[column] = evenly_spaced_values

            all_combinations: (str, float) = [(column, value) for column, values in all_decision_variants.items() for
                                              value in values]

            all_combinations_with_measure: List[((str, float), float)] = \
                [((col_name, decision_value),
                  (get_measure(dataframe=dataframe, column=col_name, decision=decision_value, criterion=criterion)))
                 for (col_name, decision_value) in all_combinations]
            if len(all_combinations_with_measure) == 0:
                print("all comb")
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
                self.caption = f'{self.column} < {self.value}'

        if self.is_leaf:
            self.predicted_class = dataframe['grade'].value_counts().idxmax()
            self.caption = f'grade: {self.predicted_class}'

    def predict(self, x: pd.DataFrame) -> int:
        if len(x) != 1:
            raise ValueError("Input DataFrame x must have exactly one row")
        if self.is_leaf:
            return self.predicted_class
        return self.left.predict(x) if x[self.column].iloc[0] < self.value else self.right.predict(x)


class Forrest(Model):
    def __init__(self, num_of_trees: int, data: pd.DataFrame, bootstrap_percent: float):
        assert 0 < (bootstrap_percent / 100) < 1
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


def print_tree(root: Tree, filename):
    graph = Digraph(format='png')

    def grade_to_color(grade):
        min_grade, max_grade = 1, 5
        normalized_grade = (grade - min_grade) / (max_grade - min_grade)
        green_intensity = int(255 * normalized_grade)
        return f"#00{green_intensity:02x}00"

    def add_nodes_edges(node: Tree, parent_id=None):
        unique_id = str(id(node))
        if node.is_leaf:
            display_label = str(node.predicted_class)
            color = grade_to_color(int(node.predicted_class))
            graph.node(unique_id, label=display_label, shape='ellipse', color=color, style='filled')
        else:
            display_label = str(node.caption)
            color = 'lightgray'
            graph.node(unique_id, label=display_label, shape='ellipse', color=color, style='filled')

        if parent_id:
            graph.edge(parent_id, unique_id)

        if not node.is_leaf:
            if node.left:
                add_nodes_edges(node.left, unique_id)
            if node.right:
                add_nodes_edges(node.right, unique_id)

    root_id = str(id(root))
    graph.node(root_id, label=str(root.caption), shape='ellipse', color='lightgray')
    add_nodes_edges(root)
    graph.render(filename, cleanup=True)


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


def get_score(test_data: [pd.Series], model: Model) -> float:
    total_elems, correct = len(test_data), 0
    for k in range(total_elems):
        row_dataframe: pd.DataFrame = pd.DataFrame([test_data.iloc[k, 0:]])
        predicted = test_data.iloc[k]['grade']
        if model.predict(row_dataframe) == predicted:
            correct += 1
    return correct / total_elems


def parallel_forrest(user_id, dataframe) -> (int, "Forrest"):
    best_forrest: Forrest = None
    best_score: float = 0.0

    for number_of_trees in [3, 5, 7]:
        for bootstrap_percent in [20, 30, 50]:
            print(f"User {user_id}: {number_of_trees} - {bootstrap_percent}")
            n = len(dataframe) // 5
            dfs: List[pd.DataFrame] = [
                dataframe[:n],
                dataframe[n: 2 * n],
                dataframe[2 * n: 3 * n],
                dataframe[3 * n: 4 * n],
                dataframe[4 * n:],
            ]
            scores: List[float] = []
            for i in range(5):
                test_portion: pd.DataFrame = dfs[i]
                train_portion: pd.DataFrame = pd.concat(dfs[:i] + dfs[i + 1:], ignore_index=True)
                print(f"User {user_id}: New forest with {number_of_trees} trees, {bootstrap_percent}% bootstrap")
                forrest = Forrest(
                    num_of_trees=number_of_trees,
                    data=train_portion,
                    bootstrap_percent=bootstrap_percent)
                scores.append(get_score(test_portion, forrest))

            current_score = statistics.mean(scores)
            if current_score > best_score:
                best_score, best_forrest = current_score, Forrest(
                    num_of_trees=number_of_trees,
                    data=dataframe,
                    bootstrap_percent=bootstrap_percent)

    return user_id, best_forrest


def parallel_trees(user_id, dataframe):
    best_tree = None
    best_score = 0.0

    for max_tree_depth in [3, 7, 10]:
        for min_samples_split in [5, 7, 10, 13]:
            for min_samples_leaf in [5, 7, 10, 13]:
                for criterion in ["gini", "entropy"]:
                    n = len(dataframe) // 5
                    dfs = [
                        dataframe[:n],
                        dataframe[n: 2 * n],
                        dataframe[2 * n: 3 * n],
                        dataframe[3 * n: 4 * n],
                        dataframe[4 * n:],
                    ]
                    scores = []
                    for n in range(5):
                        test_portion = dfs[n]
                        train_portion = pd.concat(dfs[:n] + dfs[n + 1:], ignore_index=True)
                        tree = Tree(max_tree_depth, min_samples_split, min_samples_leaf, criterion, train_portion)
                        scores.append(get_score(test_portion, tree))

                    current_score = statistics.mean(scores)
                    if current_score > best_score:
                        best_score, best_tree = current_score, Tree(max_tree_depth, min_samples_split,
                                                                    min_samples_leaf, criterion, dataframe)

    return user_id, best_tree


if __name__ == '__main__':
    # scrap_to_json()
    # extract_from_json("movies")
    print(f"Workers: {NUM_OF_WORKERS}")

    raw_train_data = pd.read_csv("train.csv", sep=';', header=None)
    raw_train_data.columns = ['id', 'user_id', 'movie_id', "grade"]

    full_train_data = raw_train_data.merge(pd.read_csv("extracted_movies.csv"), on='movie_id', how='left')

    user_dataframe_map = {user_id:
                              group_df for user_id, group_df
                          in full_train_data.groupby('user_id')}

    for user_id, dataframe in user_dataframe_map.items():
        dataframe.drop(columns=['id', 'user_id', 'release_date', 'movie_id', 'genres', 'production_companies',
                                'production_countries', 'overview'], inplace=True)

    # sequential trees
    # for user_id, dataframe in user_dataframe_map.items():
    #     parallel_trees(user_id=user_id, dataframe=dataframe)

    # sequential forest
    # for user_id, dataframe in user_dataframe_map.items():
    #     parallel_forrest(user_id=user_id, dataframe=dataframe)

    best_tree_per_user = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_OF_WORKERS) as executor:
        futures = []
        for user_id, dataframe in user_dataframe_map.items():
            futures.append(executor.submit(parallel_trees, user_id, dataframe))

        for num, future in enumerate(concurrent.futures.as_completed(futures), 1):
            user_id, best_tree = future.result()
            best_tree_per_user[user_id] = best_tree
            print(f"Finished training forest for user: {num}/358")

    fill_csv(best_tree_per_user, "submission_tree.csv")

    print_tree(best_tree_per_user[13], "tree13")
    print_tree(best_tree_per_user[17], "tree17")
    print_tree(best_tree_per_user[19], "tree19")
    print_tree(best_tree_per_user[21], "tree21")

    best_forrest_per_user: {int, Model} = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_OF_WORKERS) as executor:
        futures = [
            executor.submit(parallel_forrest, user_id, dataframe)
            for user_id, dataframe in user_dataframe_map.items()
        ]

        for num, future in enumerate(concurrent.futures.as_completed(futures), 1):
            user_id, best_forrest = future.result()
            best_forrest_per_user[user_id] = best_forrest
            print(f"Finished training forest for user: {num}/358")

    fill_csv(best_forrest_per_user, "submission_forrest.csv")
