import statistics
from typing import List

from Forrest import Forrest
from Model import Model
from Tree import Tree
import pandas as pd

from fun import print_tree, fill_csv, get_score, scrap_to_json, extract_from_json

API_KEY = "api-key"
NUM_OF_PARTITIONS: int = 10

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
                    for criterion in ["gini", "entropy"]:
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
                            best_score, best_tree = current_score, Tree(max_tree_depth, min_samples_split,
                                                                        min_samples_leaf, criterion, dataframe)

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
