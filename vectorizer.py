import pandas as pd


def create_vector(movies):
    budget = normalize(movies.get("budget"))
    popularity = normalize(movies.get("popularity"))
    revenue = normalize(movies.get("revenue"))
    runtime = normalize(movies.get("runtime"))
    vote_average = normalize(movies.get("vote_average"))
    vote_count = normalize(movies.get("vote_count"))
    votes = movies.get("vote_count") * movies.get("vote_average")
    votes.name = "votes"
    normalized_votes = normalize(votes)
    date = normalize(movies.get("release_date"))
    result = pd.concat([movies.get("movie_id"),
                        budget, popularity, date, revenue,
                        runtime, vote_average, vote_count, normalized_votes,
                        movies.get("genres"),
                        movies.get("production_companies"),
                        movies.get("production_countries"),
                        movies.get("overview"),
                        ], axis=1)
    return result


def normalize(column):
    max_value = max(column)
    min_value = min(column)
    column = (column - min_value)/(max_value - min_value)
    return column



def save_vector(vector):
    vector.to_csv("vector.csv", index=False)


if __name__ == '__main__':
    extracted_movies = pd.read_csv("extracted_movies.csv")
    vector = create_vector(extracted_movies)
    print(vector)
    save_vector(vector)


