import random
import numpy as np
import itertools
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import time
import itertools
from sklearn.model_selection import KFold

DEVICE = torch.device('cpu')
CRITERION = torch.nn.MSELoss()
MAX_EPOCHS = 80
BATCH_SIZE = 512


class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, num_features, lambda_reg):
        super(MatrixFactorization, self).__init__()
        self.user_features = torch.nn.Embedding(num_users, num_features)
        self.item_features = torch.nn.Embedding(num_items, num_features)
        torch.nn.init.uniform_(self.user_features.weight, 0, 1)
        torch.nn.init.uniform_(self.item_features.weight, 0, 1)
        self.lambda_reg = lambda_reg

    def forward(self, user_indices, item_indices):
        item_vecs = self.item_features(item_indices)
        user_vecs = self.user_features(user_indices)
        interaction = user_vecs * item_vecs
        predicted_ratings = interaction.sum(dim=(0 if interaction.dim() == 1 else 1))
        return predicted_ratings

    def regularization_loss(self):
        return self.lambda_reg * (
                self.user_features.weight.norm(p=2) ** 2 +
                self.item_features.weight.norm(p=2) ** 2
        )


def cross_validate_model(matrix, num_features, lambda_reg, learning_rate, max_epochs, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    items, users = torch.nonzero(matrix > -1, as_tuple=True)
    ratings = matrix[items, users]
    data = list(zip(users, items, ratings))

    validation_losses = []
    for train_index, val_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        val_data = [data[i] for i in val_index]

        criterion = torch.nn.MSELoss()
        model = train_model(train_data, criterion, lambda_reg, learning_rate, max_epochs, num_features)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for user, item, rating in val_data:
                user = torch.tensor(user, device=DEVICE).long()
                item = torch.tensor(item, device=DEVICE).long()
                rating = torch.tensor(rating, device=DEVICE).float()

                pred = model(user, item)
                loss = criterion(pred, rating)
                val_loss += loss.item()
        validation_losses.append(val_loss / len(val_data))

    return sum(validation_losses) / len(validation_losses)


def train_model(train_data, criterion, lambda_reg, learning_rate, max_epochs, num_features):
    num_items = matrix.shape[0]
    num_users = matrix.shape[1]
    model = MatrixFactorization(num_users, num_items, num_features, lambda_reg).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    users = torch.tensor([data[0] for data in train_data], device=DEVICE).long()
    items = torch.tensor([data[1] for data in train_data], device=DEVICE).long()
    ratings = torch.tensor([data[2] for data in train_data], device=DEVICE).float()

    for epoch in range(max_epochs):
        model.train()

        batch_size = BATCH_SIZE
        num_batches = len(users) // batch_size

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = (batch + 1) * batch_size
            user_batch = users[start_idx:end_idx]
            item_batch = items[start_idx:end_idx]
            rating_batch = ratings[start_idx:end_idx]
            pred = model(user_batch, item_batch)
            loss = criterion(pred, rating_batch) + lambda_reg * model.regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def grid_search_with_cv(matrix, hyperparameter_grid, max_epochs, n_splits=5):
    best_params, best_loss = None, float('inf')
    i = 0
    I = len(hyperparameter_grid)
    for params in hyperparameter_grid:
        num_features, learning_rate, lambda_reg = params
        avg_val_loss = cross_validate_model(matrix, num_features, lambda_reg, learning_rate, max_epochs, n_splits)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_params = params
        i += 1
        print(f"Done testing {i}/{I} -- num_features={num_features}, learning_rate={learning_rate}, reg={lambda_reg}")
    return best_params


def fill_task_csv(model: MatrixFactorization, ids: [str]):
    assert len(ids) == 358
    task = pd.read_csv("task.csv", sep=';', header=None)
    task.columns = ['id', 'user_id', 'movie_id', "grade"]
    model.eval()
    for i in range(len(task)):
        user_id = str(int(task.iloc[i]['user_id']))
        movie_id = int(task.iloc[i]['movie_id'])
        shifted_user_index: int = ids.index(user_id)
        shifted_movie_index: int = movie_id - 1
        user_tensor = torch.tensor([shifted_user_index], device=DEVICE).long()
        movie_tensor = torch.tensor([shifted_movie_index], device=DEVICE).long()
        tmp = model(user_tensor, movie_tensor)
        predicted = (tmp.item())
        pred = int(round(predicted))
        task.loc[i, "grade"] =  pred if pred in [0, 1, 2, 3, 4, 5] else random.randint(0, 5)
    task = task.astype(int)
    task.to_csv("submission.csv", index=False, header=False, sep=";")


if __name__ == '__main__':
    start_time = time.time()

    train = pd.read_csv("train.csv", sep=';', header=None)
    train.columns = ['id', 'user_id', 'movie_id', "grade"]
    train = train.drop('id', axis=1)

    train_pivot = train.pivot(index='movie_id', columns='user_id', values='grade')
    train_pivot.index = train_pivot.index.astype(str)
    train_pivot.columns = train_pivot.columns.astype(str)

    original_user_ids = train_pivot.columns.tolist()

    # replacing NaNs with placeholder -1
    train_matrix = train_pivot.fillna(-1).values
    matrix = torch.tensor(train_matrix, dtype=torch.float32).to(DEVICE)

    # Define hyperparameters
    num_features = [8, 11, 19, 21]
    learning_rate = [0.01, 0.02, 0.03, 0.05]
    lambda_reg = [0.01, 0.02, 0.05, 0.1]
    hyperparameter_grid = list(itertools.product(num_features, learning_rate, lambda_reg))

    # Create and train model with best hyperparameters
    num_features, learning_rate, lambda_reg = grid_search_with_cv(matrix, hyperparameter_grid, MAX_EPOCHS, n_splits=5)
    print(f"Finally: num_features={num_features}, learning_rate={learning_rate}, reg={lambda_reg}")
    model = train_model(matrix, CRITERION, lambda_reg, learning_rate, MAX_EPOCHS, num_features)

    # Use the model to fill missing grades in task CSV
    fill_task_csv(model, original_user_ids)

    end_time = time.time()

    print(f"Task finished in: {end_time - start_time:.2f} seconds")
