import time

import numpy as np
from tqdm import tqdm

from src.Metrics import rmse


class TrainBaseModel():
    def __init__(self, train_data, n_unique_users, n_unique_businesses, latent_features, learning_rate,
                 regularization_factor, log_session):
        self.latent_features = latent_features
        self.train_data = train_data
        self.w_user_latent_matrix = np.random.rand(n_unique_users, latent_features)
        self.w_business_latent_matrix = np.random.rand(latent_features, n_unique_businesses)

        self.w_business_bias = np.random.rand(n_unique_businesses)
        self.w_user_bias = np.random.rand(n_unique_users)

        self.learning_rate = learning_rate
        self.regularization_factor = regularization_factor
        self.log_session = log_session
        self.graph = self.log_session.graph('err', kind='min')

    def predict_user_business(self, user_index, business_index):
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
        row_order = np.arange(len(self.train_data))
        np.random.shuffle(row_order)
        all_errs = []

        for idx, curr_idx in tqdm(enumerate(row_order)):
            err = self.step(curr_idx)

            all_errs.append(err)

            if (idx + 1) % 1000 == 0:
                self.graph.append(idx // 1000, {'err': np.mean(np.abs(all_errs))})
                all_errs = []

        self.log_session.done()

    def split_to_train_validation(self):
        pass
