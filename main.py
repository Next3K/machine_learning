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
from sklearn.model_selection import train_test_split

from Model import Model
from typing import List
import numpy as np
import concurrent.futures
import os
from Tree import Tree
from Forrest import Forrest
from fun import evaluate_predictions

NUM_OF_WORKERS = os.cpu_count()


def print_tree(root: Tree, filename):
    graph = Digraph(format='png')

    def grade_to_color(grade):
        min_grade, max_grade = 0, 5
        normalized_grade = (grade - min_grade) / (max_grade - min_grade)
        green_intensity = 55 + int(200 * normalized_grade)
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


def parallel_forrest(user_id, dataframe) -> (int, Forrest):
    best_forrest: Forrest = None
    best_score: float = 0.0

    for number_of_trees in [5]:
        for bootstrap_percent in [30]:
    # for number_of_trees in [3, 5, 7, 11]:
    #     for bootstrap_percent in [20, 30, 50, 60]:
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


def parallel_trees(user_id, dataframe) -> (int, Tree):
    best_tree, best_score = None, 0.0


    for max_tree_depth in [3]:
        for min_samples_split in [7]:
            for min_samples_leaf in [3]:
                for criterion in ["entropy"]:

    # for max_tree_depth in [3, 5, 7, 10]:
    #     for min_samples_split in [2, 3, 5, 7, 11, 13]:
    #         for min_samples_leaf in [1, 2, 3, 5, 7, 11, 13]:
    #             for criterion in ["gini", "entropy"]:
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

    raw_train_data = pd.read_csv("train.csv", sep=';', header=None)
    raw_train_data.columns = ['id', 'user_id', 'movie_id', "grade"]
    full_train_data = raw_train_data.merge(pd.read_csv("extracted_movies.csv"), on='movie_id', how='left')
    full_train_data, validate = train_test_split(full_train_data, test_size=0.2, random_state=42)

    user_dataframe_map = {user_id:
                              group_df for user_id, group_df
                          in full_train_data.groupby('user_id')}
    for user_id, dataframe in user_dataframe_map.items():
        dataframe.drop(columns=['id', 'user_id', 'release_date', 'movie_id', 'genres', 'production_companies',
                                'production_countries', 'overview'], inplace=True)

    # Tree experiments
    best_tree_per_user = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_OF_WORKERS) as executor:
        futures = []
        for user_id, dataframe in user_dataframe_map.items():
            futures.append(executor.submit(parallel_trees, user_id, dataframe))

        for num, future in enumerate(concurrent.futures.as_completed(futures), 1):
            user_id, best_tree = future.result()
            best_tree_per_user[user_id] = best_tree
            print(f"Finished training tree: {num}/358")
    fill_csv(best_tree_per_user, "submission_tree.csv")

    # Forest experiments
    best_forrest_per_user: {int, Model} = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_OF_WORKERS) as executor:
        futures = [
            executor.submit(parallel_forrest, user_id, dataframe)
            for user_id, dataframe in user_dataframe_map.items()
        ]

        for num, future in enumerate(concurrent.futures.as_completed(futures), 1):
            user_id, best_forrest = future.result()
            best_forrest_per_user[user_id] = best_forrest
            print(f"Finished training forest: {num}/358")
    fill_csv(best_forrest_per_user, "submission_forest.csv")

    validate.drop(columns=['id', 'movie_id'], inplace=True)
    full_train_data.drop(columns=['id', 'movie_id'], inplace=True)

    print(f"+++++++++++++++++++VALIDATION RESULTS TREE VARIANT+++++++++++++++++++")
    predicted = []
    expected = []
    for z in range(len(validate)):
        row = validate.iloc[z, 2:]
        user_id = validate.iloc[z]['user_id']
        exp = validate.iloc[z]['grade']
        pred = best_tree_per_user[user_id].predict(pd.DataFrame([row]))
        expected.append(exp)
        predicted.append(pred)
    print(evaluate_predictions(expected, predicted))

    print(f"+++++++++++++++++++TESTWISE RESULTS TREE VARIANT+++++++++++++++++++")
    predicted = []
    expected = []
    _, test_portion = train_test_split(full_train_data, test_size=0.2, random_state=42)
    for z in range(len(test_portion)):
        row = test_portion.iloc[z, 2:]
        user_id = test_portion.iloc[z]['user_id']
        exp = test_portion.iloc[z]['grade']
        pred = best_tree_per_user[user_id].predict(pd.DataFrame([row]))
        expected.append(exp)
        predicted.append(pred)
    print(evaluate_predictions(expected, predicted))

    print(f"+++++++++++++++++++VALIDATION RESULTS FORREST VARIANT+++++++++++++++++++")
    predicted = []
    expected = []
    for z in range(len(validate)):
        row = validate.iloc[z, 2:]
        user_id = validate.iloc[z]['user_id']
        exp = validate.iloc[z]['grade']
        pred = best_forrest_per_user[user_id].predict(pd.DataFrame([row]))
        expected.append(exp)
        predicted.append(pred)
    print(evaluate_predictions(expected, predicted))

    print(f"+++++++++++++++++++TESTWISE RESULTS FORREST VARIANT+++++++++++++++++++")
    predicted = []
    expected = []
    _, test_portion = train_test_split(full_train_data, test_size=0.2, random_state=42)
    for z in range(len(test_portion)):
        row = test_portion.iloc[z, 2:]
        user_id = test_portion.iloc[z]['user_id']
        exp = test_portion.iloc[z]['grade']
        pred = best_forrest_per_user[user_id].predict(pd.DataFrame([row]))
        expected.append(exp)
        predicted.append(pred)
    print(evaluate_predictions(expected, predicted))


    # generate PNGs
    for i in range(50):
        user_id = list(best_tree_per_user.keys())[i]
        print_tree(best_tree_per_user[user_id], f"tree{i}")
