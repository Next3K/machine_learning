import requests
import random
from datetime import datetime
import json
import time
import csv
import os
import json
import pandas as pd

API_KEY = "api-key"


class Knn:
    def __init__(self, k, mask):
        self.mask = mask
        self.k = k
        self.train_data = None

    @staticmethod
    def metric(a, b, mask: [int]):
        # TODO impl
        return 1.0

    def predict(self, x) -> int:
        # TODO impl
        return 3

    def train(self, dataframe):
        # TODO impl
        self.train_data = dataframe


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
    # TODO implement
    task = pd.read_csv("task.csv", sep=';', header=None)
    task.columns = ['id', 'user_id', 'movie_id', "grade"]


if __name__ == '__main__':
    # scrap_to_json()
    # extract_from_json("movies")

    train = pd.read_csv("train.csv", sep=';', header=None)
    train.columns = ['id', 'user_id', 'movie_id', "grade"]
    extracted_movies = pd.read_csv("extracted_movies.csv")
    dataframe_per_user = {user_id: group_df for user_id, group_df in train.groupby('user_id')}

    # find the best possible knn for every user
    KNNs: {int, Knn} = {}
    for user_id, dataframe in dataframe_per_user.items():
        best_knn: Knn = None
        best_score: float = .0
        for k in [1, 3, 5, 7, 11]:
            for i in range(20):
                mask = [1 if i in
                             random.sample(range(13), random.randint(3, 8))
                        else 0 for i in range(13)]

                knn = Knn(k=k, mask=mask)
                knn.train(train)
                # TODO evaluate knn with cross validation etc
                current_score = 0
                if current_score > best_score:
                    best_score = current_score
                    best_knn = knn
        KNNs[user_id] = best_knn

    # fill task.csv
    fill_task_csv(KNNs)

    print("Done!")
