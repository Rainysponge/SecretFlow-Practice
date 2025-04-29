import pickle
import matplotlib.pyplot as plt


def pic_SLModel_result(history: dict, title: str):
    train_acc_color = "#A19AD3"
    val_acc_color = "#A1D6CB"

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    print(history.keys())
    epoch_nums = len(history["train_BinaryAccuracy"])
    x = [i for i in range(epoch_nums)]

    # AUC
    axes[0].plot(
        x,
        history["train_BinaryAUROC"],
        label="Train",
        color=train_acc_color,
        marker="o",
        markevery=5,
    )
    axes[0].plot(
        x,
        history["val_BinaryAUROC"],
        label="Valid",
        color=val_acc_color,
        marker="s",
        markevery=5,
    )
    axes[0].set_title(f"{title} AUROC", fontweight="bold")
    axes[0].set_ylabel("AUROC", fontweight="bold")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].legend()

    # Accuracy
    axes[2].plot(
        x,
        history["train_BinaryAccuracy"],
        label="Train",
        color=train_acc_color,
        marker="o",
        markevery=5,
    )
    axes[2].plot(
        x,
        history["val_BinaryAccuracy"],
        label="Valid",
        color=val_acc_color,
        marker="s",
        markevery=5,
    )
    axes[2].set_title(f"{title} Accuracy", fontweight="bold")
    axes[2].set_ylabel("Accuracy", fontweight="bold")
    axes[2].set_xlabel("Epoch", fontweight="bold")
    axes[2].legend()

    # Loss
    ax12 = axes[1].twinx()
    axes[1].plot(
        x,
        history["train_loss"],
        label="Train Loss",
        color=train_acc_color,
        marker="o",
        markevery=5,
    )
    axes[1].plot(
        x,
        history["val_loss"],
        label="Valid Loss",
        color=val_acc_color,
        marker="s",
        markevery=5,
    )

    axes[1].set_title(f"{title} Loss", fontweight="bold")
    axes[1].set_ylabel("Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(f"./figs/{title}.pdf", format="pdf", bbox_inches="tight")
    print(f"Figure saved to \033[92m./figs/{title}.pdf\033[0m")


if __name__ == "__main__":
    pass
