# @Author: nilanjan
# @Date:   2018-10-19T21:40:19+05:30
# @Email:  nilanjandaw@gmail.com
# @Filename: nn.py
# @Last modified by:   nilanjan
# @Last modified time: 2018-10-20T02:56:10+05:30
# @Copyright: Nilanjan Daw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

iterations = 500000
learning_rate = 0.0001
lambda_regularizer = 4500

error_rate = 0.00000001


def init(input, num_hidden_layer, num_hidden_nodes, output):
    np.random.seed(99)
    hidden_layers = np.random.random_sample((num_hidden_layer - 1, num_hidden_nodes, num_hidden_nodes))
    input_layer = np.random.random_sample((input, num_hidden_nodes))
    output_layer = np.random.random_sample((num_hidden_nodes, output))
    bias_hidden = np.random.random_sample((num_hidden_layer, num_hidden_nodes))
    bias_output = np.random.random_sample((output,))
    return input_layer, hidden_layers, output_layer, bias_hidden, bias_output


def normalisation(data):
    return (data - data.mean()) / data.std()


def read_data_train(file_path):
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path, skiprows=[0],
                           names=["page_likes", "page_checkin", "daily_crowd",
                                  "page_category", "F1", "F2", "F3", "F4", "F5", "F6",
                                  "F7", "F8", "c1", "c2", "c3", "c4", "c5", "base_time",
                                  "post_length", "share_count", "promotion",
                                  "h_target", "post_day", "basetime_day", "target"])

        # plot(data)

        y_train = data.loc[:, "target"]
        basetime_day = pd.get_dummies(
            data["basetime_day"], prefix='basetime_day')
        post_day = pd.get_dummies(data["post_day"], prefix="post_day")

        data = pd.concat([data, basetime_day], axis=1)
        data = pd.concat([data, post_day], axis=1)

        data.drop(['basetime_day'], axis=1, inplace=True)
        data.drop(["post_day"], axis=1, inplace=True)

        data = normalisation(data)
        data.drop(["target"], axis=1, inplace=True)
        y_train = y_train.values
        x_train = data.values
        number_of_samples, _ = np.shape(x_train)

        return x_train, y_train
    else:
        print("file not found")


def read_data_test(file_path):
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path, skiprows=[0],
                           names=["page_likes", "page_checkin", "daily_crowd",
                                  "page_category", "F1", "F2", "F3", "F4", "F5", "F6",
                                  "F7", "F8", "c1", "c2", "c3", "c4", "c5", "base_time",
                                  "post_length", "share_count", "promotion",
                                  "h_target", "post_day", "basetime_day"])

        basetime_day = pd.get_dummies(
            data["basetime_day"], prefix='basetime_day')
        post_day = pd.get_dummies(data["post_day"], prefix="post_day")

        data = pd.concat([data, basetime_day], axis=1)
        data = pd.concat([data, post_day], axis=1)

        data.drop(['basetime_day'], axis=1, inplace=True)
        data.drop(["post_day"], axis=1, inplace=True)
        x_test = normalisation(data)

        x_test[np.isnan(x_test)] = 0.0
        x_test = x_test.values
        number_of_tests, _ = np.shape(x_test)

        return x_test
    else:
        print("file not found")


def sigmoid(data):
    return 1 / (1 + np.exp(-data))


def sigmoid_dx(data):
    return sigmoid(data) * (1 - sigmoid(data))


def train_network(x_train, y_train, x_validate, y_validate):

    num_hidden_layer = 1
    num_hidden_nodes = 100
    num_classes = 3

    input_layer, hidden_layers, output_layer, bias_hidden, bias_output = \
                                    init(np.shape(x_train[0])[0],
                                    num_hidden_layer, num_hidden_nodes, num_classes)
    x = np.add(np.dot(x_train[0], input_layer), bias_hidden[0])
    x = sigmoid(x)
    print(x)
    for index in range(num_hidden_layer - 1):
        print(np.shape(x))
        x = np.add(np.dot(x, hidden_layers[index]), bias_hidden[index + 1])
        x = sigmoid(x)
        print(x)
    print(np.shape(x))
    x = np.add(np.dot(x, output_layer), bias_output)
    x = sigmoid(x)
    print(x)
    return 0


def main():
    global learning_rate, lambda_regularizer
    parser = argparse.ArgumentParser()

    parser.add_argument('--lambda', dest='lambda_rate', type=float)
    parser.add_argument('--alpha', dest='learning_rate', type=float)
    parser.add_argument('--train', dest='train', type=str)
    parser.add_argument('--test', dest='test', type=str)
    args = parser.parse_args()

    learning_rate = args.learning_rate
    lambda_regularizer = args.lambda_rate

    x_train, y_train = read_data_train(args.train)
    x_validate = x_train[25600:, :]
    x_train = x_train[:25600, :]

    y_validate = y_train[25600:]
    y_train = y_train[:25600]

    number_of_samples, number_of_features = np.shape(x_train)
    print("train_data", number_of_samples, number_of_features)
    x_test = read_data_test(args.test)
    print("test_data", np.shape(x_test))
    train_network(x_train, y_train, x_validate, y_validate)
    # test_data = test(x_test, weight)
    #
    # i = 0
    # with open('submission.csv', 'w') as file:
    #     fieldNames = ["Id", "target"]
    #     writer = csv.DictWriter(file, fieldNames)
    #     writer.writeheader()
    #     for row in test_data:
    #         writer.writerow({"Id": str(i), "target": str(row)})
    #         i = i + 1

    print("done")


if __name__ == '__main__':
    main()
