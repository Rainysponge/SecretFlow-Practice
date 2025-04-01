# 基于SecrectFlow进行联邦学习

在本实验中,同学们需要使用隐语的ml.nn.fl模块（新版本为secretflow_fl）完成一个分类问题.同学们可以按照下述流程完成实验:
- 熟悉联邦学习的基础过程和相关变体范式;
- 安装运行隐语，学习隐语中关于联邦学习的相关代码;
- 使用隐语中fl模块撰写FedAvg训练代码并和单方模型进行对比;
- 尝试基于隐语实现一个联邦学习范式（选做）;
- 撰写实验报告.

## 安装隐语

具体可以参考session1中的安装步骤，安装完成后可以通过下面的代码测试是否安装正确

```cmd
[root@localhost SecretFlow-Practice]$ python
Python 3.10.14 (main, May  6 2024, 19:52:50) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
>>> 
```

## 使用隐语进行联邦学习

这里请各位同学根据**正确版本号**对应的文档可以补全文件 fedavg_test.py，并尝试运行。如果代码无误的话，可以得到如下结果：

![](https://raw.githubusercontent.com/Rainysponge/Figurebed/main/img/20250401183759.png)

同学们也可以通过修改 single_train.py 中的代码来测试当仅持有单个客户端时的训练结果,并和联邦学习的全局模型性能进行比较。

此时我们使用的是tf模型，为了验证隐语对于torch的支持，请同学们根据文档和 fedavg_test.py 文件补全fedavg_mnist.py中的代码，并将数据集划分有四六分改为**三七分**，进行实验，并通过脚本 fedavg_pic.py 中给出的绘图函数进行绘图，结果大致如下（此处展示的是数据集四六开的结果）：

![](https://raw.githubusercontent.com/Rainysponge/Figurebed/main/img/20250401190512.png)

## 尝试其他联邦学习范式


接下来我们尝试一些其他联邦学习的训练范式，细心的同学想必已经发现了类FLModel在初始化时有一个 strategy变量，隐语默认支持 FedProx 等范式，这里以FedProx为例，我们只需要修改FLModel初始化的超参输出即可使用其他范式：

```python
fed_model = FLModel(
    ...
    strategy="fed_prox",
    ...
)
```
具体可以查看 secretflow/ml/nn/fl/backend/torch/strategy/ 中相关范式的实现 ，希望同学们选取2-3个范式进行实验，并记录结果。如果想尝试更多的范式，建议更新隐语版本至最新，并修改import即可，先前的fl相关代码全部被转移至secretflow_fl中了。

## 尝试实现联邦学习范式（选做）

这一部分建议同学们可以参考隐语库中 [PR#1587](https://github.com/secretflow/secretflow/pull/1587)进行尝试.

