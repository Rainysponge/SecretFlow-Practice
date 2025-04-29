import torch
import numpy as np
import torch.nn as nn
from torchmetrics import AUROC, Accuracy
import secretflow as sf
import pandas as pd

from secretflow.utils.simulation.datasets import dataset
from secretflow.utils.simulation.datasets import load_bank_marketing
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel

from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.preprocessing.encoder import LabelEncoder
from secretflow.data.split import train_test_split

from secretflow.ml.nn.core.torch import (
    metric_wrapper,
    optim_wrapper,
    TorchModel,
)

from models import BaseModel, FuseModel
from dp_pic import pic_SLModel_result


if __name__ == "__main__":
    sf.shutdown()
    sf.init(["alice", "bob"], address="local")
    alice, bob = sf.PYU("alice"), sf.PYU("bob")

    df = pd.read_csv(dataset("bank_marketing"), sep=";")

    spu = sf.SPU(sf.utils.testing.cluster_def(["alice", "bob"]))
    data = load_bank_marketing(parts={alice: (0, 4), bob: (4, 16)}, axis=1)
    label = load_bank_marketing(parts={alice: (16, 17)}, axis=1)

    encoder = LabelEncoder()
    data["job"] = encoder.fit_transform(data["job"])
    data["marital"] = encoder.fit_transform(data["marital"])
    data["education"] = encoder.fit_transform(data["education"])
    data["default"] = encoder.fit_transform(data["default"])
    data["housing"] = encoder.fit_transform(data["housing"])
    data["loan"] = encoder.fit_transform(data["loan"])
    data["contact"] = encoder.fit_transform(data["contact"])
    data["poutcome"] = encoder.fit_transform(data["poutcome"])
    data["month"] = encoder.fit_transform(data["month"])
    label = encoder.fit_transform(label)
    label = label.astype(np.float32)
    print(f"label= {type(label)},\ndata = {type(data)}")
    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)
    random_state = 1234
    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state
    )
    train_label, test_label = train_test_split(
        label, train_size=0.8, random_state=random_state
    )

    hidden_size = 64

    metrics = [
        metric_wrapper(Accuracy, task="binary"),
        metric_wrapper(AUROC, task="binary"),
    ]

    learning_rate = 1e-3

    base_model_alice = TorchModel(
        model_fn=BaseModel,
        optim_fn=optim_wrapper(torch.optim.Adam, lr=learning_rate),
        input_dim=4,
        output_dim=hidden_size,
    )

    base_model_bob = TorchModel(
        model_fn=BaseModel,
        optim_fn=optim_wrapper(torch.optim.Adam, lr=learning_rate),
        # 请补充
    )

    fuse_model = TorchModel(
        model_fn=FuseModel,
        loss_fn=nn.BCELoss,
        metrics=metrics,
        optim_fn=optim_wrapper(torch.optim.Adam, lr=learning_rate),
        # 请补充
    )

    base_model_dict = {alice: base_model_alice, bob: base_model_bob}

    train_batch_size = 128

    sl_model = SLModel(
        # 请补充
        backend="torch",
    )

    history = sl_model.fit(
        # 请补充
    )

    pic_SLModel_result(history, "SLModel")
