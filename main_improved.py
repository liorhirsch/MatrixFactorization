import argparse
import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import losswise

from main import convert_string_ids_to_int, load_data
from src.RecommenderSystem import MatrixFactorizationBasic, MatrixFactorizationImproved


def main(learning_rate, regularization_factor, latent_features, max_epochs, metric):
    np.random.seed(1)

    print("Loading Data")
    train_df, test_df = load_data()

    n_unique_users = len(train_df.user_id.unique())
    n_unique_businesses = len(train_df.business_id.unique())

    print("Starting Training")

    losswise.set_api_key('W20EQ09CW')
    session = losswise.Session(tag='matrix_factorization_improved',
                               max_iter=10000,
                               params={'learning rate': learning_rate,
                                       'regularization': regularization_factor,
                                       'latent_features': latent_features,
                                       'max_epochs': max_epochs,
                                       'metric': metric})

    mf_model = MatrixFactorizationImproved(data=train_df,test_data=test_df, n_unique_users=n_unique_users, n_unique_businesses=n_unique_businesses, latent_features=latent_features,
                                        learning_rate=learning_rate, regularization_factor=regularization_factor,
                                        max_epochs=max_epochs, log_session= session, metric=metric)
    mf_model.train()
    session.done()


def extract_args_from_cmd():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', type=float, nargs='?')
    parser.add_argument('--regularization', type=float, nargs='?')
    parser.add_argument('--latent_features', type=int, nargs='?')
    parser.add_argument('--max_epochs', type=int, nargs='?')
    parser.add_argument('--metric', type=str, nargs='?')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = extract_args_from_cmd()
    main(args.lr, args.regularization, args.latent_features, args.max_epochs, args.metric)
