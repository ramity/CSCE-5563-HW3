import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self, cuda_enabled = True):
        super(Net, self).__init__()

        self.classes = 10 + 1
        self.cuda_enabled = cuda_enabled

        # conv1
        self.conv1_input_channel = 1
        self.conv1_output_channel = 10
        self.conv1_kernel_size = (3, 3)
        self.conv1_stride = (1, 1)
        self.conv1 = nn.Conv2d(
            self.conv1_input_channel,
            self.conv1_output_channel,
            self.conv1_kernel_size,
            self.conv1_stride
        )

        # conv2
        self.conv2_input_channel = 10
        self.conv2_output_channel = 20
        self.conv2_kernel_size = (3, 3)
        self.conv2_stride = (1, 1)
        self.conv2 = nn.Conv2d(
            self.conv2_input_channel,
            self.conv2_output_channel,
            self.conv2_kernel_size,
            self.conv2_stride
        )

        # maxpool
        self.max_pooling_kernel_size = (3, 3)
        self.max_pooling_stride = (3, 3)
        self.max_pooling = nn.MaxPool2d(
            self.max_pooling_kernel_size,
            self.max_pooling_stride
        )

        # conv3
        self.conv3_input_channel = 20
        self.conv3_output_channel = 25
        self.conv3_kernel_size = (2, 2)
        self.conv3_stride = (2, 2)
        self.conv3 = nn.Conv2d(
            self.conv3_input_channel,
            self.conv3_output_channel,
            self.conv3_kernel_size,
            self.conv3_stride
        )

        # batch normalize
        self.batch_normalize = nn.BatchNorm2d(self.conv3_output_channel)

        # drop out
        self.dropout = nn.Dropout2d()

        # lstm
        self.lstm_input_size = 5 * self.conv3_output_channel
        self.lstm_hidden_size = 32
        self.lstm_num_layers = 3
        self.lstm_hidden = None
        self.lstm_cell = None
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            self.lstm_num_layers,
            batch_first = True,
            bidirectional = True
        )

        # linear
        # 2 = number of directions the lstm layer outputs
        self.linear_input_size = 2 * self.lstm_hidden_size
        self.linear_output_size = self.classes
        self.linear = nn.Linear(self.linear_input_size, self.linear_output_size)

        # softmax
        self.softmax = nn.Softmax(dim=1)

        # log_softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

        # initialize
        nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv2.bias, 0.1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv3.bias, 0.1)
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l1, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l2, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l1, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l2, gain=np.sqrt(2))
        # nn.init.constant_(self.lstm.bias_ih_l0, 0.1)
        # nn.init.constant_(self.lstm.bias_ih_l1, 0.1)
        # nn.init.constant_(self.lstm.bias_ih_l2, 0.1)
        # nn.init.constant_(self.lstm.bias_hh_l0, 0.1)
        # nn.init.constant_(self.lstm.bias_hh_l1, 0.1)
        # nn.init.constant_(self.lstm.bias_hh_l2, 0.1)
        nn.init.xavier_uniform_(self.linear.weight, gain=np.sqrt(2))
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, input):
        # currently: [batch_size, H, W]
        batch_size = int(input.size()[0])

        output = self.conv1(input)
        output = F.relu(output)

        output = self.conv2(output)
        output = F.relu(output)
        output = self.max_pooling(output)
        output = self.dropout(output)

        output = self.conv3(output)
        output = self.batch_normalize(output)
        output = F.relu(output)
        output = self.dropout(output)

        # 1: update to: [batch_size, W, H, conv3_output_channel]
        # 2: ensure the output is contiguous
        # 3: condense H and conv3_output_chanel into self.lstm_input_size
        output = output.permute(0, 3, 2, 1)
        output.contiguous()
        output = output.reshape(batch_size, -1, self.lstm_input_size)

        output, self.lstm_hidden = self.lstm(output, (self.lstm_hidden, self.lstm_cell))
        output.contiguous()
        output = output.reshape(-1, self.linear_input_size)

        output = self.linear(output)
        softmax_output = self.softmax(output)
        log_softmax_output = self.log_softmax(output)

        return softmax_output, log_softmax_output

    def reset_hidden(self, batch_size):
        zeros = torch.zeros(self.lstm_num_layers * 2, batch_size, self.lstm_hidden_size)
        zeros = zeros.cuda() if self.cuda_enabled else zeros
        self.lstm_hidden = Variable(zeros)

    def reset_cell(self, batch_size):
        zeros = torch.zeros(self.lstm_num_layers * 2, batch_size, self.lstm_hidden_size)
        zeros = zeros.cuda() if self.cuda_enabled else zeros
        self.lstm_cell = Variable(zeros)
