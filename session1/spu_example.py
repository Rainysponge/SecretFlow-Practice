import secretflow as sf
import numpy as np
import time
import matplotlib.pyplot as plt


def plus(a, b):
    return a + b


def sub(a, b):
    return a - b


def multi(a, b):
    return a * b


def divide(a, b):
    return a / b


if __name__ == "__main__":

    # Check the version of your SecretFlow
    print("The version of SecretFlow: {}".format(sf.__version__))

    # In case you have a running secretflow runtime already.
    sf.shutdown()

    sf.init(["alice", "bob"], address="local")
    alice, bob = sf.PYU("alice"), sf.PYU("bob")
    spu = sf.SPU(sf.utils.testing.cluster_def(["alice", "bob"]))
    device = spu
    x1 = 20
    x2 = 30
    x1_ = sf.to(alice, x1).to(device)
    x2_ = sf.to(bob, x2).to(device)

    spu_time_list = []
    plain_time_list = []
    # 加法运算
    start_time = time.time()
    res = device(plus)(x1_, x2_)
    res = sf.reveal(res)
    end_time = time.time()
    print(f"加法运算的时间是 {end_time - start_time}s, 结果是{res}")
    spu_time_list.append(end_time - start_time)
    start_time = time.time()
    res = x1 + x2
    end_time = time.time()
    print(f"明文除法运算的时间是 {end_time - start_time}s, 结果是{res}")
    plain_time_list.append(end_time - start_time)
    print("-" * 30)
    # 减法运算
    start_time = time.time()
    res = device(sub)(x1_, x2_)
    res = sf.reveal(res)
    end_time = time.time()
    print(f"减法运算的时间是 {end_time - start_time}s, 结果是{res}")
    spu_time_list.append(end_time - start_time)
    start_time = time.time()
    res = x1 - x2
    end_time = time.time()
    print(f"明文减法运算的时间是 {time.time() - start_time}s, 结果是{res}")
    plain_time_list.append(end_time - start_time)
    print("-" * 30)
    # 乘法运算
    start_time = time.time()
    res = device(multi)(x1_, x2_)
    res = sf.reveal(res)
    end_time = time.time()
    spu_time_list.append(end_time - start_time)
    print(f"乘法运算的时间是 {end_time - start_time}s, 结果是{res}")
    start_time = time.time()
    res = x1 * x2
    end_time = time.time()
    print(f"明文乘法运算的时间是 {end_time - start_time}s, 结果是{res}")
    plain_time_list.append(end_time - start_time)
    print("-" * 30)
    # 除法运算
    start_time = time.time()
    res = device(divide)(x1_, x2_)
    res = sf.reveal(res)
    end_time = time.time()
    spu_time_list.append(end_time - start_time)
    print(f"除法运算的时间是 {end_time - start_time}s, 结果是{res}")

    start_time = time.time()
    res = x1 / x2
    end_time = time.time()
    plain_time_list.append(end_time - start_time)
    print(f"明文除法运算的时间是 {time.time() - start_time}s, 结果是{res}")

    plain_time_list = np.array(plain_time_list)
    spu_time_list = np.array(spu_time_list)

    x = ["Plus", "Sub", "Multi", "Div"]
    fig, ax = plt.subplots()
    bar_width = 0.22
    index = np.arange(len(x))
    ax.bar(
        index,
        spu_time_list,
        bar_width,
        label="SPU",
        color="#B1C29E",
        edgecolor="white",
        hatch="*",
    )
    ax.bar(
        index + bar_width,
        plain_time_list,
        bar_width,
        label="Plain",
        color="#F0A04B",
        edgecolor="white",
        hatch="x",
    )
    ax.set_ylabel("Time (s)", fontweight="bold")
    ax.set_title("SPU vs. Plain: Time Cost Comparison", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(x)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figs/time.pdf", format="pdf", bbox_inches="tight")
