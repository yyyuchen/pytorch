import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader = Data.DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=2,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor(),
)
test_x = test_data.data.type(torch.float32)[:2000] / 255.
test_y = test_data.targets[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

import numpy as np
if __name__ == '__main__':
    rnn = RNN()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x.view(-1, 28, 28))
            b_y = Variable(y)
            output = rnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_out = rnn(test_x)
                pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
                accuracy = (sum(pred_y == np.array(test_y.data)).item()) / test_y.size(0)
                print('Epoch:', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)

    test_output = rnn(test_x[:10].view(-1, 28, 28))
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10], 'real number')
