import pandas as pd
import numpy as np


def get_dataframe(name):
    df = pd.read_csv(name, sep=';', header=None)
    df.columns = ['id', 'user_id', 'movie_id', "grade"]
    df = df.drop('id', axis=1)
    if name == "train.csv":
        df = df.pivot(index="user_id", columns="movie_id", values="grade")
    else:
        df = df.sort_values(by="user_id", ignore_index=True)
    return df


def calculate_distance(use_binary, a, b):
    if use_binary:
        if a == b:
            return 0
        else:
            return 1
    else:
        return abs(a - b)


def calculate_difference(extracted_grades, i, use_binary):
    distances = []
    for j in range(0, len(extracted_grades)):
        amount_of_comparison = 0
        distance = 0
        if (j == i):
            distances.append(float('nan'))
            continue
        for l in range(0, len(extracted_grades.columns)):
            if pd.isna(extracted_grades.iloc[i, l]):
                continue
            if pd.isna(extracted_grades.iloc[j, l]):
                continue
            distance += calculate_distance(use_binary, extracted_grades.iloc[i, l], extracted_grades.iloc[j, l])
            amount_of_comparison += 1
        if amount_of_comparison == 0:
            distances.append(float('nan'))
        else:
            distances.append(distance / amount_of_comparison)
    distances = np.argsort(np.isnan(distances).astype(int) + np.nan_to_num(distances, nan=float('inf')))
    return distances


def get_grade(extracted_grades, movie_id, distances, k):
    grade = 0
    j = 0
    for i in range(0, len(distances)):
        if pd.isna(extracted_grades.iloc[distances[i], movie_id - 1]):
            continue
        grade += extracted_grades.iloc[distances[i], movie_id - 1]
        j += 1
        if (j == k):
            break
    return int(round(grade / k, 0))


def resolve_task(extracted_grades, task, use_binary, k):
    l = 0
    solution = task
    for i in range(0, len(extracted_grades)):
        distances = calculate_difference(extracted_grades, i, use_binary)
        for j in range(l, len(task)):
            print(f"Calculating Grade: {j}/{len(task)}")
            if extracted_grades.index[i] == task.loc[j, 'user_id']:
                solution.loc[j, "grade"] = get_grade(extracted_grades, task.loc[j, 'movie_id'], distances, k)
            if extracted_grades.index[i] != task.loc[j, 'user_id']:
                l = j
                break
    return solution


def fill_task_csv(solution, file_name):
    df = pd.read_csv("task.csv", sep=';', header=None)
    for i in range(0, len(df)):
        print(f"Setting Grade: {i}/{len(solution)}")
        for j in range(0, len(solution)):
            if df.iloc[i, 1] == solution.iloc[j, 0] and df.iloc[i, 2] == solution.iloc[j, 1]:
                df.iloc[i, 3] = solution.iloc[j, 2]
                break
    df = df.astype(int)
    df.to_csv(file_name, index=False, header=False, sep=";")


if __name__ == '__main__':
    extracted_grades = get_dataframe("train.csv")
    task = get_dataframe("task.csv")
    solution = resolve_task(extracted_grades, task, True, 1)
    fill_task_csv(solution, "submission_binary.csv")