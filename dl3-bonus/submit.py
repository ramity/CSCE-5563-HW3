import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import time
import os
import sys

from decode import decode
from model import *

# init parameters
batch_size = 200
classes = 11
cuda = True if torch.cuda.is_available() else False
epochs = 200
kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}
start_epoch = 1
log_interval = 1
lr = 0.005
seed = 1
validate_batch_size = 200

# set seed
torch.manual_seed = seed
if cuda:
    torch.cuda.manual_seed = seed
np.random.seed(seed)

# load the dataset
test_data = np.load("mnist_test.npy")
test_data = torch.Tensor(test_data)
# Don't worry about me loading in the mnist_dev_labels file. I'm only doing this
# because pytorch's TensorDataset doesn't support not having labels, sadly.
test_labels = np.load("Mnist_dev_labels.npy").astype(int)
test_labels = torch.IntTensor(test_labels)
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    **kwargs
)

# load and initialize the model
model = Net(cuda)

model.load_state_dict(torch.load("./compiled/train-0.06-test-0.11.pt"))

if cuda:
    model.cuda()

def submit():
    model.eval()

    with torch.no_grad():

        for input, target in test_loader:
            # reset states
            batch_size = input.shape[0]
            target = target + 1

            model.reset_hidden(batch_size)
            model.reset_cell(batch_size)
            model.zero_grad()

            if cuda:
                input, target = input.cuda(), target.cuda()

            input = input.view(input.shape[0], 1, input.shape[1], input.shape[2])
            input, target = Variable(input), Variable(target)
            softmax_output, log_softmax_output = model(input)

            softmax_output = softmax_output.view(batch_size, -1, classes)
            predicted_digits = decode(softmax_output)

            with open("LewisBrown_010744629.csv", 'a') as fd:
                for id in range(0, batch_size):
                    row_string = str(id) + ","
                    for digit_id in range(0, 10):
                        row_string = row_string + str(predicted_digits[id][digit_id])

                    fd.write(row_string + "\n")

submit()
