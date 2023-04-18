from torch.nn import Module, Linear, ReLU, init


class TextEncoder(Module):
    def __init__(self, embedding_size=1000):
        super(TextEncoder, self).__init__()
        self.linear1 = Linear(300, embedding_size)
        self.activation = ReLU()

        self.init_weights()

    def init_weights(self):
        # Linear
        init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.activation(x)
        x = self.linear1(x)
        x = x / x.pow(2).sum(1, keepdim=True).sqrt()
        return x
