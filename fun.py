import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from graphviz import Digraph
from pandas import Series

from Model import Model
from Tree import Tree
from main import API_KEY


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


def print_tree(root: Tree, filename):
    graph = Digraph(format='png')

    def add_nodes_edges(node: Tree, parent_caption=None):
        node_caption = str(node.caption)
        if node.is_leaf:
            node_caption = f"Class: {node.predicted_class}"
            graph.node(node_caption, shape='ellipse', color='lightblue', style='filled')

        if parent_caption:
            graph.edge(parent_caption, node_caption)

        if not node.is_leaf:
            if node.left:
                add_nodes_edges(node.left, node_caption)
            if node.right:
                add_nodes_edges(node.right, node_caption)

    graph.node(str(root.caption), shape='box', color='lightgray')
    add_nodes_edges(root, None)
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
