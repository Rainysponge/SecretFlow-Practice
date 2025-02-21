import secretflow as sf
import numpy as np
import jax.numpy as jnp
from jax import grad
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score


def breast_cancer(party_id=None, train: bool = True):
    pass


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)


# Training loss is the negative log-likelihood of the training examples.
def loss(W, b, inputs, targets):
    pass


def train_step(W, b, x1, x2, y, learning_rate):
    pass


def fit(W, b, x1, x2, y, epochs=1, learning_rate=1e-2):
    pass


def validate_model(W, b, X_test, y_test):
    pass


if __name__ == "__main__":

    # Check the version of your SecretFlow
    print("The version of SecretFlow: {}".format(sf.__version__))

    # In case you have a running secretflow runtime already.
    sf.shutdown()

    sf.init(["alice", "bob"], address="local")

    alice, bob = sf.PYU("alice"), sf.PYU("bob")
    spu = sf.SPU(sf.utils.testing.cluster_def(["alice", "bob"]))
    pass
    # auc = validate_model(sf.reveal(W_), sf.reveal(b_), X_test, y_test)
    # print(f"auc={auc}")
