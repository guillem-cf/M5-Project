from week1.models.base_model import BaseModel
from week1.models.components.simple_dense_net import SimpleDenseNet


class Model(BaseModel):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.dense = SimpleDenseNet(in_features, out_features)

    def forward(self, x):
        x = self.dense(x)
        return x


if __name__ == '__main__':
    model = Model(784, 10)
    print(model)
