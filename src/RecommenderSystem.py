import time

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.Metrics import rmse


def split_train_val(data):
    return train_test_split(data, test_size=0.3, stratify=data.user_id_int)


class MatrixFactorizationBasic:
    def __init__(self, data, n_unique_users, n_unique_businesses, latent_features, learning_rate,
                 regularization_factor, max_epochs, log_session=None):
        self.latent_features = latent_features
        self.train_data, self.val_data = split_train_val(data)

        self.w_user_latent_matrix = np.random.rand(n_unique_users, latent_features)
        self.w_business_latent_matrix = np.random.rand(latent_features, n_unique_businesses)

        self.w_business_bias = np.random.rand(n_unique_businesses)
        self.w_user_bias = np.random.rand(n_unique_users)

        self.learning_rate = learning_rate
        self.regularization_factor = regularization_factor
        self.log_session = log_session
        self.max_epochs = max_epochs

        if self.log_session is not None:
            self.inner_epoch_err = [self.log_session.graph(f'train_epoch_{idx}') for idx in range(self.max_epochs)]
            self.epoch_graph = self.log_session.graph('epoch_err', display_interval=1)

    def predict_user_business(self, user_index, business_index):
        if user_index.size > 1:
            return np.sum(self.w_user_latent_matrix[user_index] * self.w_business_latent_matrix[:, business_index].T,
                          axis=1) + \
                   self.w_user_bias[user_index] + self.w_business_bias[business_index]
        else:
            return np.dot(self.w_user_latent_matrix[user_index], self.w_business_latent_matrix[:, business_index]) + \
                   self.w_user_bias[user_index] + self.w_business_bias[business_index]

    def calculate_error(self, user_index, business_index, true_val):
        predicted_val = self.predict_user_business(user_index, business_index)
        return true_val - predicted_val

    def update_rule(self, err, main_values, secondary_values):
        return main_values + self.learning_rate * (err * secondary_values - self.regularization_factor * main_values)

    def step(self, idx):
        curr_row = self.train_data.iloc[idx]
        curr_user_idx = curr_row.user_id_int
        curr_business_idx = curr_row.business_id_int
        real_rank = curr_row.stars

        err = self.calculate_error(curr_user_idx, curr_business_idx, real_rank)
        curr_user_latent = self.w_user_latent_matrix[curr_user_idx]
        curr_business_latent = self.w_business_latent_matrix[:, curr_business_idx]
        self.w_user_latent_matrix[curr_user_idx] = self.update_rule(err, curr_user_latent, curr_business_latent)
        self.w_business_latent_matrix[:, curr_business_idx] = self.update_rule(err, curr_business_latent,
                                                                               curr_user_latent)
        curr_user_bias = self.w_user_bias[curr_user_idx]
        curr_business_bias = self.w_business_bias[curr_business_idx]
        self.w_user_bias[curr_user_idx] = self.update_rule(err, curr_user_bias, 1)
        self.w_business_bias[curr_business_idx] = self.update_rule(err, curr_business_bias, 1)

        return err

    def train(self):
        last_val_error = np.inf
        patience_counter = 0

        for epoch_idx in range(self.max_epochs):
            row_order = np.arange(len(self.train_data))
            np.random.shuffle(row_order)
            batch_train_errs = []
            all_train_errs = []

            for log_idx, curr_idx in enumerate(tqdm(row_order)):
                log_idx += 1
                err = self.step(curr_idx)
                all_train_errs.append(err)
                batch_train_errs.append(err)

                if (log_idx + 1) % 1000 == 0:
                    self.inner_epoch_err[epoch_idx].append(log_idx, {'train_err': rmse(batch_train_errs)})
                    batch_train_errs.clear()

            val_err = self.calculate_validation_err()
            train_err = rmse(all_train_errs)
            self.epoch_graph.append(epoch_idx + 1, {'train_err': train_err, 'validation_err': val_err})

    def calculate_validation_err(self):
        all_errs = self.calculate_error(self.val_data.user_id_int, self.val_data.business_id_int, self.val_data.stars)
        return rmse(all_errs)


class MatrixFactorizationImproved(MatrixFactorizationBasic):
    def __init__(self, data, n_unique_users, n_unique_businesses, latent_features, learning_rate,
                 regularization_factor, max_epochs, log_session=None):
        super().__init__(data, n_unique_users, n_unique_businesses, latent_features, learning_rate,
                         regularization_factor, max_epochs, log_session)

        self.R = dict(data.groupby('user_id_int').apply(lambda x: x.business_id_int.unique()))
        self.w_y = np.random.rand(n_unique_businesses, latent_features)
        self.train_data_average_stars = self.train_data.stars.mean()

    def predict_user_business(self, user_index, business_index):
        b_ui = self.w_user_bias[user_index] + self.w_business_bias[business_index] + self.train_data_average_stars
        curr_R = self.R[user_index]

        if user_index.size > 1:
            # TODO implement this part
            return np.sum(self.w_user_latent_matrix[user_index] * self.w_business_latent_matrix[:, business_index].T,
                          axis=1) + b_ui
        else:
            user_representation = self.w_user_latent_matrix[user_index] + (len(curr_R) ** -0.5) * self.w_y[curr_R].sum(
                axis=0)
            return np.dot(user_representation, self.w_business_latent_matrix[:, business_index]) + b_ui

    def step(self, idx):
        curr_row = self.train_data.iloc[idx]
        curr_user_idx = curr_row.user_id_int
        curr_business_idx = curr_row.business_id_int
        real_rank = curr_row.stars

        err = self.calculate_error(curr_user_idx, curr_business_idx, real_rank)

        curr_R = self.R[curr_user_idx]
        curr_user_latent = self.w_user_latent_matrix[curr_user_idx]
        curr_business_latent = self.w_business_latent_matrix[:, curr_business_idx]

        user_representation = curr_user_latent + (len(curr_R) ** -0.5) * self.w_y[curr_R].sum(axis=0)


        self.w_user_latent_matrix[curr_user_idx] = self.update_rule(err, user_representation, curr_business_latent)
        self.w_business_latent_matrix[:, curr_business_idx] = self.update_rule(err, curr_business_latent,
                                                                               curr_user_latent)
        curr_user_bias = self.w_user_bias[curr_user_idx]
        curr_business_bias = self.w_business_bias[curr_business_idx]
        self.w_user_bias[curr_user_idx] = self.update_rule(err, curr_user_bias, 1)
        self.w_business_bias[curr_business_idx] = self.update_rule(err, curr_business_bias, 1)

        self.w_y[curr_R] = self.update_rule(err, self.w_y[curr_R], (len(curr_R) ** -0.5) * curr_business_latent)

        return err
