import numpy as np
import torch
from torch import nn
import torch.utils.data as data

"""
A simple Bayes classifier consists of two linear layers with Relu activation
"""


# Load data
class LoadData(data.Dataset):
    def __init__(self):
        # Assign numerical values to categorical variables
        dict = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4,
                '2': 1, '3': 2, '4': 3, '5more': 4,
                'more': 5,
                'small': 1, 'big': 3,
                'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
        txt_data = np.loadtxt('car.data', dtype=str, delimiter=',')
        num_data = torch.tensor([[dict[i] for i in row] for row in txt_data])
        # Explainable variable (input)
        self.x = num_data[:, [0, 1, 2, 4, 5]].to(torch.float32)
        # Predictable variable (output)
        self.y = num_data[:, 6]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


# A simple classification model
class MLPNet(nn.Module):
    def __init__(self, dim_in=5, dim_out=4, dim_latent=5):
        super(MLPNet, self).__init__()
        seq = []
        seq += [nn.Linear(dim_in, dim_latent),
                nn.LeakyReLU(0.1),
                nn.Linear(dim_latent, dim_out)]
        self.net = nn.Sequential(*seq)

    def forward(self, data_in):
        out = self.net(data_in)
        return out


# Train and test the classification model
class CarPricePrediction:
    def __init__(self):
        data_set = LoadData()
        # split data into train data nad test data
        train_data, test_data = data.random_split(data_set, [len(data_set)-100, 100])
        self.train_data = data.DataLoader(train_data, batch_size=20, shuffle=True)
        self.test_data = test_data

        self.mlp_net = MLPNet()
        self.optimizer = torch.optim.SGD(self.mlp_net.parameters(), lr=1e-4, weight_decay=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_model(self):
        # Train for 50 epochs
        for epoch in range(50):
            for i, data_batch in enumerate(self.train_data):
                x, y = data_batch
                mlp_out = self.mlp_net(x)
                mlp_loss = self.loss_fn(mlp_out, y)
                self.optimizer.zero_grad()
                mlp_loss.backward()
                self.optimizer.step()

                acc = (torch.argmax(mlp_out, dim=1) == y).sum() / len(y)
                print(f"Training epoch:{epoch:>d}), loss:{mlp_loss:>.3f}, acc:{acc:>.1%}", end='\r')

    # Test the trained model
    def test_model(self):
        self.mlp_net.eval()
        x, y = self.test_data[:]
        with torch.no_grad():
            mlp_out = self.mlp_net(x)
            mlp_loss = self.loss_fn(mlp_out, y)
        acc = (torch.argmax(mlp_out, dim=1) == y).sum() / len(y)
        print(f"Test results: loss:{mlp_loss:>.3f}, acc:{acc:>.1%}")


if __name__ == '__main__':
    m = CarPricePrediction()
    m.train_model()
    m.test_model()



