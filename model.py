from __future__ import print_function
import numpy as np
from sklearn.preprocessing import normalize

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from progressbar import ProgressBar
import time

try:
    import tensorflow as tf
except ImportError:
    print('! TensorFlow not installed. No tensorflow logging.')
    tf = None


def tf_log(writer, key, value, epoch):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, epoch)


def make_train_valid_dataset(name, length=24, train_percentage=0.7):

    print('> reading %s ...' % name)
    aq_matrix = np.load(open(name, 'rb'))
    total_count, dim = aq_matrix.shape
    print('> get matrix of shape', total_count, dim)

    max_values = np.max(aq_matrix, axis=0)
    aq_matrix = aq_matrix / max_values

    train_matrix = aq_matrix[:int(train_percentage * total_count), :]
    valid_matrix = aq_matrix[int(train_percentage * total_count):, :]

    train_dataset = AQDataset(train_matrix, length)
    valid_dataset = AQDataset(valid_matrix, length)
    return train_dataset, valid_dataset, max_values


class AQDataset(Dataset):
    def __init__(self, aq_matrix, length):
        super(AQDataset, self).__init__()
        # (total_count, dim)

        self.total_count, self.dim = aq_matrix.shape
        self.length = length
        self.matrix = aq_matrix

    def __getitem__(self, index):
        input_matrix = self.matrix[index: index + self.length, :]
        target = self.matrix[index + 1: index + 1 + self.length, :]

        input_matrix = torch.from_numpy(input_matrix).float()
        target = torch.from_numpy(target).float()

        return input_matrix, target
        # [length, dim] [length, dim]

    def __len__(self):
        return self.total_count - self.length


class BaseModel(nn.Module):
    def __init__(self, dim, length, hid_dim):
        super(BaseModel, self).__init__()

        self.rnn = nn.LSTM(hid_dim, hid_dim, 1, batch_first=True)

        self.emb = nn.Linear(dim, hid_dim)
        self.output = nn.Linear(hid_dim, dim)
        self.dim = dim
        self.length = length
        self.hid_dim = hid_dim

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (1, batch, self.hid_dim)
        return (Variable(weight.new(*hid_shape).zero_()),
                Variable(weight.new(*hid_shape).zero_()))

    def forward(self, x):
        # x [batch, length/2, dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        x = self.emb(x)
        output, hidden = self.rnn(x, hidden)
        output = self.output(output)

        return output


if __name__ == '__main__':
    feature_dim = 6
    use_length = 24
    rnn_hid_dim = 64
    lr = 0.001
    batch_size = 256
    num_epochs = 40
    grad_clip_rate = 0.5
    epoch_per_display = 4
    output = './output/tf_log'

    train_dataset, valid_dataset, max_values = make_train_valid_dataset('bj_aq.npy', use_length)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=1)
    train_writer = tf and tf.summary.FileWriter(os.path.join(output, 'train/'))
    valid_writer = tf and tf.summary.FileWriter(os.path.join(output, 'valid/'))

    model = BaseModel(feature_dim, use_length, rnn_hid_dim)
    model = model.cuda()
    model = nn.DataParallel(model).cuda()
    crit = nn.MSELoss(reduce=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in xrange(num_epochs):

        total_loss = 0.

        for i, (x, t) in enumerate(train_loader):
            x = Variable(x, requires_grad=True).cuda()
            t = Variable(t, requires_grad=False).cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = crit(output, t)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), grad_clip_rate)
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        total_loss /= len(train_loader)

        if epoch % epoch_per_display == 0:

            # evaluate
            valid_loss = 0.
            for i, (x, t) in enumerate(valid_loader):
                x = Variable(x, volatile=True).cuda()
                t = Variable(t, volatile=True).cuda()
                output = model(x)
                loss = crit(output, t)
                valid_loss += loss.item() * x.size(0)
            valid_loss /= len(valid_loader)

            print('> epoch {}, train loss {}, valid loss {}'.format(epoch, total_loss, valid_loss))

            # write to tf log
            tf_log(train_writer, 'loss', total_loss, epoch)
            tf_log(valid_writer, 'loss', valid_loss, epoch)
            train_writer.flush()
            valid_writer.flush()

    # sample
    x, t = valid_dataset[0]
    x = Variable(x, volatile=True).cuda()
    t = Variable(t, volatile=True).cuda()
    x, t = x.unsqueeze(0), t.unsqueeze(0)
    output = model(x)
    output = output.squeeze(0)
    output = output.data
    x, t = x.squeeze(0), t.squeeze(0)
    current = np.multiply(x, max_values)
    output = np.multiply(output, max_values)
    target = np.multiply(t, max_values)

    print(current[2])
    print(current[3])
    print(target[2])
    print(output[2])
