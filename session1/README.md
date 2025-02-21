# 基于SecrectFlow通过安全多方计算进行逻辑回归分类

在本实验中,同学们需要使用隐语的SPU模块完成一个简单的二元分类问题.同学们可以按照下述流程完成实验:
- 熟悉多方安全计算的目的和基本步骤;
- 安装运行隐语;
- 撰写逻辑回归代码,运行得到基准结果;
- 基于隐语实现逻辑回归,运行得到结果;
- 撰写实验报告.

## 安装隐语
相信在先前的课程和过往的学习中,同学们已经了解了什么是多方安全计算和逻辑回归.如果对这方面的知识仍有欠缺,可以搜索隐语的官方课程.在有了相应的基础之后,现在我们可以开始试着安装运行隐语框架.
使用conda新建并激活环境:
```cmd
conda create -n sf_prac python=3.10
conda activate sf_prac
```
安装方法可以参考官方文档,如下:
```cmd
pip install -U secretflow  // 完整版本
pip install -U secretflow-lite  // 轻量版本
```
两个版本都可以,如果后续不在更换虚拟机可以选择完整版本,一步到位以免出现其他问题.如果是租用的服务器,请检查conda的源是否有被更换过,如果安装失败请检查网络情况,并使用ali源进行安装.

当然也可以直接通过 requirements.txt 进行安装:
```cmd
pip -r requirements.txt
```

在完成安装后,检查安装是否有效.

```cmd
cd session1
python sf_test.py 
```
正常情况下会得到下面的结果

![](https://raw.githubusercontent.com/Rainysponge/Figurebed/main/img/20250217155339.png)

当然这一步可能就有同学失败了,不要慌这是正常的.这里列举几个可能出现的问题.

第一是secretflow模块没有找到.这个情况可能是因为安装中断,或者部分包的版本不对,运行如下指令即可:
```cmd
pip install secretflow==1.6.1b0  // 建议指定版本,隐语的文档目前还没有和最新的版本对齐
```
第二是无法建立node,解决方法和上面一致,还是因为有一些模块的目录位置变动导致现在有些代码无法运行.

## 逻辑回归

按照教程,同学们请补全 session1_lr.py 中的代码，接着来运行一下:
```cmd
python session1_lr.py  // 这里教程的代码存在bug,请自行修复
```
结果如下:

![](https://raw.githubusercontent.com/Rainysponge/Figurebed/main/img/20250217160442.png)

## 使用SPU进行逻辑回归

接着，我们需要使用SPU进行逻辑回归。请同学们按照教程补全session1_lr_spu.py,并试着运行:
```cmd
python session1_lr_spu.py  // 教程中仍存在一些小问题，请自行修正
```
如果代码正确的话输入结果如下:

![](https://raw.githubusercontent.com/Rainysponge/Figurebed/main/img/20250217163337.png)
