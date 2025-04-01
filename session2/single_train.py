import numpy as np
from tensorflow import keras
from tensorflow.python.keras import layers
from secretflow.utils.simulation.datasets import load_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from secretflow.utils.simulation.datasets import dataset


def create_model(input_shape, num_classes):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":

    num_classes = 10
    input_shape = (28, 28, 1)
    single_model = create_model(input_shape, num_classes)

    mnist = np.load(dataset("mnist"), allow_pickle=True)
    image = mnist["x_train"]
    label = mnist["y_train"]

    alice_x = image[:10000]
    alice_y = label[:10000]
    alice_y = OneHotEncoder(sparse_output=False).fit_transform(alice_y.reshape(-1, 1))

    random_seed = 1234
    alice_X_train, alice_X_test, alice_y_train, alice_y_test = train_test_split(
        alice_x, alice_y, test_size=0.1, random_state=random_seed
    )

    history = single_model.fit(
        alice_X_train,
        alice_y_train,
        validation_data=(alice_X_test, alice_y_test),
        batch_size=128,
        epochs=5,
    )

    print(history)
