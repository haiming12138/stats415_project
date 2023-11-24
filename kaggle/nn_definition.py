import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, nonlin=nn.ELU(0.6)):
        super().__init__()

        self.dense0 = nn.Linear(106, 256)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.3)
        self.dense1 = nn.Linear(256, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)
        self.init_weights()
    
    def init_weights(self):
        torch.manual_seed(415)
        for dense in [self.dense0, self.dense1, self.dense2, self.dense3, self.output]:
            nn.init.xavier_normal_(dense.weight)
            nn.init.constant_(dense.bias, 0.0)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense2(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense3(X))
        X = self.output(X)
        return X