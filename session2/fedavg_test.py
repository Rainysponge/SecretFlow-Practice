from matplotlib import pyplot as plt
import secretflow as sf
from secretflow.security.aggregation import SPUAggregator, SecureAggregator
from secretflow.ml.nn import FLModel
from secretflow.utils.simulation.datasets import load_mnist


def create_conv_model(input_shape, num_classes, name="model"):
    def create_model():
        from tensorflow import keras
        from tensorflow.python.keras import layers

        # Create model
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
        # Compile model
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    return create_model


if __name__ == "__main__":

    sf.shutdown()

    sf.init(["alice", "bob", "charlie"], address="local")
    alice, bob, charlie = sf.PYU("alice"), sf.PYU("bob"), sf.PYU("charlie")
    spu = sf.SPU(sf.utils.testing.cluster_def(["alice", "bob"]))

    (x_train, y_train), (x_test, y_test) = load_mnist(
        parts={alice: 0.5, bob: 0.5},
        normalized_x=True,
        categorical_y=True,
        is_torch=False,
    )
    num_classes = 10
    input_shape = (28, 28, 1)
    model = create_conv_model(input_shape, num_classes)

    device_list = [alice, bob]

    secure_aggregator = SecureAggregator(charlie, [alice, bob])
    spu_aggregator = SPUAggregator(spu)

    fed_model = None  # 请补全

    history = None  # 请补全

    global_metric = fed_model.evaluate(x_test, y_test, batch_size=128)
    print("global_metric", global_metric)
    plt.plot(history["global_history"]["accuracy"])
    plt.plot(history["global_history"]["val_accuracy"])
    plt.title("FLModel accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Valid"], loc="upper left")
    plt.savefig("./figs/acc.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    plt.plot(history["global_history"]["loss"])
    plt.plot(history["global_history"]["val_loss"])
    plt.title("FLModel loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Valid"], loc="upper left")
    plt.show()
    plt.tight_layout()
    plt.savefig("./figs/loss.pdf", format="pdf", bbox_inches="tight")
