import pandas as pd
from tfidf import get_highest_value_from_tfidf, calculate_tfidf


def split_by_id(vectors, id):
    selected = vectors[vectors["movie_id"] == id]
    vectors = vectors[vectors["movie_id"] != id]
    ids = vectors[["movie_id"]]
    selected = selected.drop(columns="movie_id")
    vectors = vectors.drop(columns="movie_id")
    return selected, vectors, ids


def calculate_total_distance(selected, vectors, tfidf_values, selected_row):
    distances = []
    columns = selected.columns
    for i in range(0, len(vectors)):
        total = 0
        for j in range(0, len(selected.columns)):
            if columns[j] == "overview":
                if selected_row <= i:
                    total += tfidf_distance(tfidf_values[selected_row], tfidf_values[i + 1])
                else:
                    total += tfidf_distance(tfidf_values[selected_row], tfidf_values[i])
            else:
                value_a = selected.iloc[0, j]
                value_b = vectors.iloc[i, j]
                if columns[j] == "production_companies" or columns[j] == "production_countries" or columns[j] == "genres":
                    total += array_text_feature(value_a, value_b)
                else:
                    total += euclidean_distance(value_a, value_b)
        distances.append(total)
    print(distances)
    return distances


def euclidean_distance(a, b):
    return abs(a - b)


def tfidf_distance(string_a, string_b):
    return 1 - get_highest_value_from_tfidf(string_a, string_b)


def array_text_feature(string_a, string_b):
    array_a = string_a.split(",")
    array_b = string_b.split(",")
    length = max(len(array_a), len(array_b))
    value = set(array_a).intersection(array_b)
    similarity = len(value) / length
    return 1 - similarity


if __name__ == '__main__':
    features = pd.read_csv("vector.csv")
    tfidf_values = calculate_tfidf(features["overview"])
    selected_row = 0
    selected, vectors, movie_ids = split_by_id(features, features.get("movie_id")[selected_row])
    calculate_total_distance(selected, vectors, tfidf_values, selected_row)

