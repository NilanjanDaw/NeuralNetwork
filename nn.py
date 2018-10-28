# @Author: nilanjan
# @Date:   2018-10-19T21:40:19+05:30
# @Email:  nilanjandaw@gmail.com
# @Filename: nn.py
# @Last modified by:   nilanjan
# @Last modified time: 2018-10-29T01:43:37+05:30
# @Copyright: Nilanjan Daw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import csv

iterations = 1000
learning_rate = 0.0001
lambda_regularizer = 4500

num_hidden_layers = 1
num_hidden_nodes = 100
num_classes = 3

error_rate = 0.00000001
batch_size = 100
epsilon = 1e-12


def init(input, num_hidden_layer, num_hidden_nodes, output):
    np.random.seed(99)
    hidden_layers = np.random.random_sample(
        (num_hidden_layers - 1, num_hidden_nodes, num_hidden_nodes)) - 0.5
    input_layer = np.random.random_sample((input, num_hidden_nodes)) - 0.5
    output_layer = np.random.random_sample((num_hidden_nodes, output)) - 0.5
    bias_hidden = np.random.random_sample(
        (num_hidden_layer, num_hidden_nodes)) - 0.5
    bias_output = np.random.random_sample((output,)) - 0.5
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
        data = data.sample(frac=1).reset_index(drop=True)
        y_train = pd.get_dummies(data["target"], prefix="target")
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
    data = np.asarray(data)
    return data * (1 - data)


def cross_entropy(predicted, true_label):
    loss = np.nan_to_num(true_label * np.log(predicted)
                         + (1 - true_label) * np.log(1 - predicted))
    return np.mean(loss)


def del_output(predicted, true_label):
    return predicted - true_label


def update_weight(layers, delta_loss, activations):
    global learning_rate
    for index in reversed(range(len(layers))):
        del_error = np.dot(delta_loss[index], np.asarray(activations[index]))
        del_error = learning_rate * np.array(del_error)
        layers[index] -= del_error.T
    return layers


def ps(name, data):
    print(name, np.shape(data))


def forward_pass(activations, layers):

    for index in range(len(layers)):
        x = np.dot(activations[index], layers[index])
        x = sigmoid(x)
        activations.append(x)

    return activations


def accuracy_calculate(predicted, true_label):
    predicted = np.argmax(predicted, axis=1) + 1
    true_label = np.argmax(true_label, axis=1) + 1
    acc = np.count_nonzero(predicted == true_label) / len(true_label)
    return acc


def evaluate(layers, x, y=None):

    for index in range(len(layers)):
        x = np.dot(x, layers[index])
        x = sigmoid(x)

    if y is not None:
        cross_entropy_loss = cross_entropy(x, y)
        accuracy = accuracy_calculate(x, y)
        return x, -cross_entropy_loss, accuracy
    return x


def back_propagation(x, true_label, layers, bias):

    activations = []
    delta_loss = []
    activations = [x]

    activations = forward_pass(activations, layers)

    for index in range(len(activations)):
        activations[index] = np.squeeze(activations[index])

    cross_entropy_loss = cross_entropy(activations[-1], true_label)

    delta_loss.insert(0, del_output(activations[-1], true_label).T)
    for index in reversed(range(len(layers) - 1)):
        del_sigmoid = sigmoid_dx(activations[index + 1])
        hidden_layer_loss = np.dot(np.asmatrix(
            layers[index + 1]), delta_loss[0])
        hidden_layer_loss = hidden_layer_loss * del_sigmoid
        delta_loss.insert(0, hidden_layer_loss)

    layers = update_weight(layers, delta_loss, activations)
    return -cross_entropy_loss, layers


def unison_shuffled(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


def plot_fig(data, label):
    x = np.arange(len(data))
    plt.plot(x, data)
    plt.title(label=label)
    plt.show()


def adagrad(loss_validate, validation_loss):
    global learning_rate
    decay_rate = 0.01
    loss_validate = np.asarray(loss_validate)
    loss_validate = decay_rate * np.sum(validation_loss ** 2)
    learning_rate += - learning_rate * validation_loss / (np.sqrt(loss_validate) + 0.000001)


def train_network(x_train, y_train, x_validate, y_validate):

    print(num_hidden_layers)
    input_layer, hidden_layers, output_layer, bias_hidden, bias_output = \
        init(np.shape(x_train[0])[0],
             num_hidden_layers, num_hidden_nodes, num_classes)

    print(np.shape(input_layer), np.shape(
        hidden_layers), np.shape(output_layer))
    layers = []
    bias = []
    layers.append(input_layer)
    if (num_hidden_layers > 1):
        for hidden_layer in hidden_layers:
            layers.append(hidden_layer)
    layers.append(output_layer)
    bias.append(bias_hidden)
    bias.append(bias_output)
    loss_train = []
    loss_validate = []
    acc_validate = []
    print(np.shape(layers))
    for i in range(iterations):
        x_train, y_train = unison_shuffled(x_train, y_train)
        loss_train_i, layers = back_propagation(
            x_train[0:batch_size, :], y_train[0:batch_size, :], layers, bias)
        print("Iteration", i, "loss", loss_train_i)

        loss_train.append(loss_train_i)
        eval, validation_loss, validation_acc = \
            evaluate(layers, x_validate, y_validate)
        loss_validate.append(validation_loss)
        acc_validate.append(validation_acc)
        # adagrad(loss_validate, validation_loss)
    plot_fig(loss_train, label="train_loss")
    plot_fig(acc_validate, label="validation_acc")
    plot_fig(loss_validate, label="validation_loss")
    return loss_train, layers


def test(x_test, layers):
    x = x_test
    for index in range(len(layers)):
        x = np.dot(x, layers[index])
        x = sigmoid(x)
    x = np.argmax(x, axis=1) + 1
    print(x)
    return x


def main():
    global learning_rate, lambda_regularizer, num_hidden_layers, num_hidden_nodes
    parser = argparse.ArgumentParser()

    parser.add_argument('--lambda', dest='lambda_rate', type=float)
    parser.add_argument('--alpha', dest='learning_rate', type=float)
    parser.add_argument('--num_hidden_layers',
                        dest='num_hidden_layers', type=float)
    parser.add_argument('--num_hidden_nodes',
                        dest='num_hidden_nodes', type=float)
    parser.add_argument('--train', dest='train', type=str)
    parser.add_argument('--test', dest='test', type=str)
    args = parser.parse_args()

    learning_rate = args.learning_rate if args.learning_rate is not None \
        else learning_rate
    lambda_regularizer = args.lambda_rate
    num_hidden_layers = int(args.num_hidden_layers)
    num_hidden_nodes = int(args.num_hidden_nodes)

    x_train, y_train = read_data_train(args.train)
    x_validate = x_train[25600:, :]
    x_train = x_train[:25600, :]

    y_validate = y_train[25600:]
    y_train = y_train[:25600]

    number_of_samples, number_of_features = np.shape(x_train)
    print("train_data", number_of_samples, number_of_features)
    x_test = read_data_test(args.test)
    print("test_data", np.shape(x_test))
    loss, layers = train_network(x_train, y_train, x_validate, y_validate)
    test_data = test(x_test, layers)

    i = 1
    with open('submission.csv', 'w') as file:
        fieldNames = ["Id", "predicted_class"]
        writer = csv.DictWriter(file, fieldNames)
        writer.writeheader()
        for row in test_data:
            writer.writerow({"Id": str(i), "predicted_class": str(row)})
            i = i + 1
    print("done")


if __name__ == '__main__':
    main()
