from torch import nn, optim
from torch.nn import functional as F
import secretflow as sf
from secretflow.data.ndarray import load
from secretflow.utils.simulation.datasets import load_mnist
from secretflow.security.aggregation import SecureAggregator
from secretflow.ml.nn import FLModel
from secretflow.ml.nn.core.torch import (
    metric_wrapper,
    optim_wrapper,
    BaseModule,
    TorchModel,
)
from torchmetrics import Accuracy, Precision
from matplotlib import pyplot as plt


class ConvNet(BaseModule):
    """Small ConvNet for MNIST."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        self.fc = nn.Linear(self.fc_in_dim, 10)

    def forward(self, x):
        x = x.to(self.fc.weight.device)
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.fc_in_dim)
        x = self.fc(x)
        return F.softmax(x, dim=1)


if __name__ == "__main__":
    sf.shutdown()

    sf.init(["alice", "bob", "charlie"], address="local", debug_mode=False, num_gpus=1)
    alice, bob, charlie = sf.PYU("alice"), sf.PYU("bob"), sf.PYU("charlie")
    device_list = [alice, bob]

    (train_data, train_label), (test_data, test_label) = load_mnist(
        parts={alice: 0.4, bob: 0.6},
        normalized_x=True,
        categorical_y=True,
        is_torch=True,
    )
    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
    model = None  # 请补全
    
    secure_aggregator = None  # 请补全

    fed_model = None  # 请补全

    history = None  # 请补全
    print(history)
    global_metric = fed_model.evaluate(test_data, test_label, batch_size=128)
    print(global_metric)
    
