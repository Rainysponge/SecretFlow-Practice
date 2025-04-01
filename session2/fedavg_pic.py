from matplotlib import pyplot as plt


def pic_fedavg_pic_torch(history: dict):
    train_acc_color = "#A19AD3"
    val_acc_color = "#A1D6CB"
    train_loss_color = "#FF8989"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    epoch_nums = len(history["global_history"]["multiclassaccuracy"])
    x = [i + 1 for i in range(epoch_nums)]
    # global average
    axes[0].plot(
        x,
        history["global_history"]["multiclassaccuracy"],
        label="Train",
        color=train_acc_color,
        marker="o",
        markevery=5,
    )
    axes[0].plot(
        x,
        history["global_history"]["val_multiclassaccuracy"],
        label="Valid",
        color=val_acc_color,
        marker="s",
        markevery=5,
    )
    axes[0].set_title("FLModel Global Accuracy", fontweight="bold")
    axes[0].set_ylabel("Accuracy", fontweight="bold")
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].legend()

    # Alice
    ax12 = axes[1].twinx()
    axes[1].plot(
        x,
        history["local_history"]["alice_train_multiclassaccuracy"],
        label="Train Acc",
        color=train_acc_color,
        marker="o",
        markevery=5,
    )
    axes[1].plot(
        x,
        history["local_history"]["alice_val_eval_multiclassaccuracy"],
        label="Valid Acc",
        color=val_acc_color,
        marker="s",
        markevery=5,
    )
    ax12.plot(
        x,
        history["local_history"]["alice_train-loss"],
        label="Train Loss",
        linestyle="dashed",
        color=train_loss_color,
        marker="*",
        markevery=5,
    )
    axes[1].set_title("Alice Local Accuracy & Loss", fontweight="bold")
    axes[1].set_ylabel("Accuracy", fontweight="bold")
    axes[1].set_ylim(0.7, 1.0)
    ax12.set_ylim(1.45, 1.7)
    axes[1].set_xlabel("Epoch", fontweight="bold")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax12.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # Bob
    ax22 = axes[2].twinx()
    axes[2].plot(
        x,
        history["local_history"]["bob_train_multiclassaccuracy"],
        label="Train",
        color=train_acc_color,
        marker="o",
        markevery=5,
    )
    axes[2].plot(
        x,
        history["local_history"]["bob_val_eval_multiclassaccuracy"],
        label="Valid",
        color=val_acc_color,
        marker="s",
        markevery=5,
    )
    ax22.plot(
        x,
        history["local_history"]["bob_train-loss"],
        label="Train Loss",
        linestyle="dashed",
        color=train_loss_color,
        marker="*",
        markevery=5,
    )
    axes[2].set_title("Bob Local Accuracy & Loss", fontweight="bold")
    ax22.set_ylabel("Loss", fontweight="bold")
    ax22.set_ylim(1.45, 1.7)
    axes[2].set_ylim(0.7, 1.0)
    axes[2].set_xlabel("Epoch", fontweight="bold")
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax22.get_legend_handles_labels()

    axes[2].legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    fig.savefig("./figs/output.pdf", format="pdf", bbox_inches="tight")
