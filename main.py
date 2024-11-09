import requests
import json
import time
import csv
import os
import json
import pandas as pd

API_KEY = "none"


def scrap_to_json():
    data = pd.read_csv("movie.csv", sep=';', header=None)
    data.columns = ['1', '2', '3']
    ids = data['2'].tolist()

    url_template = "https://api.themoviedb.org/3/movie/{}?api_key=" + API_KEY
    headers = {
        "accept": "application/json"
    }

    for movie_id in ids:
        url = url_template.format(movie_id)
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            movie_data = response.json()
            file_path = os.path.join("movies", f"movie_{movie_id}.json")
            with open(file_path, "w") as json_file:
                json.dump(movie_data, json_file, indent=4)
            print(f"Saved data for movie ID {movie_id}")
            time.sleep(0.5)
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error for movie ID {movie_id}: {http_err}")
        except Exception as err:
            print(f"Error for movie ID {movie_id}: {err}")


def extract_from_json():
    directory = "movies"
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as file:
                movie_data = json.load(file)
                genres = [genre['name'] for genre in movie_data.get("genres", [])]
                budget = movie_data.get("budget")
                popularity = movie_data.get("popularity")
                production_companies = [company['name'] for company in movie_data.get("production_companies", [])]
                production_countries = [country['name'] for country in movie_data.get("production_countries", [])]
                release_date = movie_data.get("release_date")
                revenue = movie_data.get("revenue")
                runtime = movie_data.get("runtime")
                title = movie_data.get("title")
                vote_average = movie_data.get("vote_average")
                vote_count = movie_data.get("vote_count")
                data.append({
                    "genres": ", ".join(genres),
                    "budget": budget,
                    "popularity": popularity,
                    "production_companies": ", ".join(production_companies),
                    "production_countries": ", ".join(production_countries),
                    "release_date": release_date,
                    "revenue": revenue,
                    "runtime": runtime,
                    "title": title,
                    "vote_average": vote_average,
                    "vote_count": vote_count
                })

    df = pd.DataFrame(data)
    df.to_csv("extracted_movies.csv", index=False)


if __name__ == '__main__':
    # scrap_to_json()
    # extract_from_json()

    train = pd.read_csv("train.csv", sep=';', header=None)
    train.columns = ['id', 'user_id', 'movie_id', "grade"]

    task = pd.read_csv("task.csv", sep=';', header=None)
    task.columns = ['id', 'user_id', 'movie_id', "grade"]

    extracted_movies = pd.read_csv("extracted_movies.csv")

    print("Done!")

