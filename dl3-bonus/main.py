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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from model import *

# init parameters
batch_size = 500
classes = 11
cuda = True if torch.cuda.is_available() else False
epochs = 200
kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}
start_epoch = 1
log_interval = 1
lr = 0.001
seed = 1
validate_batch_size = 200

# set seed
torch.manual_seed = seed
if cuda:
    torch.cuda.manual_seed = seed
np.random.seed(seed)

# load the dataset
train_data = np.load("mnist_train.npy")
train_data = torch.Tensor(train_data)
train_labels = np.load("mnist_train_labels.npy").astype(int)
train_labels = torch.IntTensor(train_labels)
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    **kwargs
)

validate_data = np.load("mnist_dev.npy")
validate_data = torch.Tensor(validate_data)
validate_labels = np.load("mnist_dev_labels.npy").astype(int)
validate_labels = torch.IntTensor(validate_labels)
validate_dataset = torch.utils.data.TensorDataset(validate_data, validate_labels)
validate_loader = torch.utils.data.DataLoader(
    validate_dataset,
    batch_size=validate_batch_size,
    shuffle=False,
    **kwargs
)

# load and initialize the model
model = Net(cuda)

model.load_state_dict(torch.load("./compiled/1575605491.081466-epoch-25-train-0.055310836993157864-test-0.10805314779281616.pt"))

if cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CTCLoss(
    blank=0,
    reduction='mean',
    zero_infinity=False
)

# define train method
def train(epoch):

    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_id, (input, target) in enumerate(train_loader):

        batch_size = input.shape[0]
        target = target + 1

        model.reset_hidden(batch_size)
        model.reset_cell(batch_size)
        model.zero_grad()

        if cuda:
            input, target = input.cuda(), target.cuda()

        input = input.view(batch_size, 1, input.shape[1], input.shape[2])
        input, target = Variable(input), Variable(target)
        softmax_output, log_softmax_output = model(input)

        softmax_output = softmax_output.view(batch_size, -1, classes)
        log_softmax_output = log_softmax_output.view(batch_size, -1, classes)

        # nn.CTCLoss expects a LogSoftmaxed output
        log_softmax_output = log_softmax_output.permute(1, 0, 2)
        input_lengths = torch.full(size=(batch_size,), fill_value=71, dtype=torch.int32)
        target_lengths = torch.full(size=(batch_size,), fill_value=10, dtype=torch.int32)
        loss = criterion(log_softmax_output, target, input_lengths, target_lengths)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_id % log_interval == 0 and batch_id != 0:
            print('Train Epoch: {:03d} [{:04d}/{:04d} ({:03.0f}%)]\t'
                  'Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'
                  'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}, sum: {batch_time.sum:.3f})\t'.format(
                epoch, batch_id * len(input), len(train_loader.dataset),
                100. * batch_id / len(train_loader), loss = losses, batch_time = batch_time))

    return losses.avg

# define validate method
def validate():

    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    with torch.no_grad():

        for input, target in validate_loader:
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
            log_softmax_output = log_softmax_output.view(batch_size, -1, classes)

            # nn.CTCLoss expects a LogSoftmaxed output
            log_softmax_output = log_softmax_output.permute(1, 0, 2)
            input_lengths = torch.full(size=(batch_size,), fill_value=71, dtype=torch.int32)
            target_lengths = torch.full(size=(batch_size,), fill_value=10, dtype=torch.int32)
            loss = criterion(log_softmax_output, target, input_lengths, target_lengths)
            losses.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

    print('-------------------------------------------------------------------')
    print('Validation pass:\t\t\t'
          'Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'
          'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}, sum: {batch_time.sum:.3f})\t'.format(
          loss = losses, batch_time = batch_time))
    print('-------------------------------------------------------------------')

    return losses.avg

train_loss_average = None
validate_loss_average = None

# train
for epoch in range(start_epoch, epochs + 1):
    train_loss_average = train(epoch)
    validate_loss_average = validate()

    if epoch % 25 == 0:
        filename = str(time.time())
        filename = filename + "-epoch-" + str(epoch)
        filename = filename + "-train-" + str(train_loss_average)
        filename = filename + "-test-" + str(validate_loss_average)
        filepath = "./compiled/" + filename + ".pt"
        print("Saving ", filename)
        torch.save(model.state_dict(), filepath)
        print("Saved ", filename)
