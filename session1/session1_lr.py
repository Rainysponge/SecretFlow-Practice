import numpy as np
from jax import grad
import jax.numpy as jnp
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
    x = jnp.concatenate([x1, x2], axis=1)
    pass


def fit(W, b, x1, x2, y, epochs=1, learning_rate=1e-2):
    pass


def validate_model(W, b, X_test, y_test):
    pass


if __name__ == "__main__":
    pass
