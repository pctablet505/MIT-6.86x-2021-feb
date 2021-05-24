import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.cuda.empty_cache()
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 1024
nb_classes = 10
nb_epoch = 100
num_classes = 10
img_rows, img_cols = 42, 28  # input image dimensions


class CNN(nn.Module):

    def __init__(self, input_dimension):

        super(CNN, self).__init__()
        # TODO initialize model layers here
        self.l1 = nn.Conv2d(1, 32, 3)
        self.l2 = nn.Dropout(0.3)
        self.l3 = nn.ReLU()
        self.l4 = nn.MaxPool2d(2)
        self.l5 = nn.Conv2d(32, 64, 3)
        self.l6 = nn.Dropout(0.3)
        self.l7 = nn.ReLU()
        self.l8 = nn.MaxPool2d(2)
        self.l9 = nn.Conv2d(64, 256, 3)
        self.l10 = nn.Flatten()
        self.l11 = nn.Linear(256*7*3, 128)
        self.l12 = nn.Dropout(0.3)
        self.l13 = nn.Linear(128, 256)
        self.l14 = nn.Dropout(0.3)
        self.l15 = nn.Linear(256, 20)

    def forward(self, x):

        # TODO use model layers to predict the two digits
        z = self.l1(x)
        z = self.l2(z)
        z = self.l3(z)
        z = self.l4(z)
        z = self.l5(z)
        z = self.l6(z)
        z = self.l7(z)
        z = self.l8(z)
        z = self.l9(z)
        z = self.l10(z)
        z = F.relu(self.l11(z))
        z = self.l12(z)
        z = F.relu(self.l13(z))
        z = self.l14(z)
        z = self.l15(z)

        out_first_digit = z[:, :10]
        out_second_digit = z[:, 10:]

        return out_first_digit, out_second_digit


def main():
    X_train, y_train, X_test, y_test = U.get_data(
        path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation],
               [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension)  # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model,
                lr=0.02, n_epochs=nb_epoch, momentum=0.8)

    # Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(
        loss[0], acc[0], loss[1], acc[1]))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
