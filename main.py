import argparse
import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import losswise

from src.RecommenderSystem import MatrixFactorizationBasic


def load_data():
    data_path = 'data'
    train_path = join(data_path, 'userTrainData.csv')
    test_path = join(data_path, 'userTestData.csv')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def convert_string_ids_to_int(column_values):
    unique_values = column_values.unique()
    str_id_to_int = dict(zip(unique_values, range(len(unique_values))))
    return column_values.apply(str_id_to_int.get)


def main(learning_rate, regularization_factor, latent_features, max_epochs):
    np.random.seed(1)

    print("Loading Data")
    train_df, test_df = load_data()

    print("Preprocess Data")
    train_df['user_id_int'] = convert_string_ids_to_int(train_df.user_id)
    train_df['business_id_int'] = convert_string_ids_to_int(train_df.business_id)
    n_unique_users = len(train_df.user_id.unique())
    n_unique_businesses = len(train_df.business_id.unique())

    print("Starting Training")

    losswise.set_api_key('W20EQ09CW')
    session = losswise.Session(tag='matrix_factorization',
                               max_iter=10000,
                               params={'learning rate': learning_rate,
                                       'regularization': regularization_factor,
                                       'latent_features': latent_features,
                                       'max_epochs': max_epochs})

    mf_model = MatrixFactorizationBasic(train_df, n_unique_users, n_unique_businesses, latent_features=latent_features,
                                        learning_rate=learning_rate, regularization_factor=regularization_factor,
                                        log_session=session, max_epochs=max_epochs)
    mf_model.train()
    session.done()


def extract_args_from_cmd():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', type=float, nargs='?')
    parser.add_argument('--regularization', type=float, nargs='?')
    parser.add_argument('--latent_features', type=int, nargs='?')
    parser.add_argument('--max_epochs', type=int, nargs='?')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = extract_args_from_cmd()
    main(args.lr, args.regularization, args.latent_features, args.max_epochs)
