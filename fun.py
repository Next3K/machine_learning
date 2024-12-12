import json
import time
from datetime import datetime

import numpy as np
import requests
import pandas as pd
import os


API_KEY = "api-key"


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

def confusion_matrix(expected, predicted, labels):
    expected = np.array(expected)
    predicted = np.array(predicted)
    num_labels = len(labels)
    matrix = np.zeros((num_labels, num_labels), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for e, p in zip(expected, predicted):
        if e in label_to_index and p in label_to_index:
            matrix[label_to_index[e], label_to_index[p]] += 1
    return matrix

def evaluate_predictions(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    arr1 = np.array(list1)
    arr2 = np.array(list2)

    accuracy = np.mean(arr1 == arr2) * 100
    accuracy_plus_minus = np.mean(np.abs(arr1 - arr2) <= 1) * 100
    average_abs_error = np.mean(np.abs(arr1 - arr2))
    matrix_error = confusion_matrix(arr1, arr2, labels=[0, 1, 2, 3, 4, 5])

    return {
        "accuracy": accuracy,
        "accuracy_plus_minus": accuracy_plus_minus,
        "average_abs_error": average_abs_error,
        "matrix_error": matrix_error
    }