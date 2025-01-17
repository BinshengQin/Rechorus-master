
# 机器学习大作业--在[ReChorus](https://github.com/THUwangcy/ReChorus)框架下复现论文：[Cross Pairwise Ranking for Unbiased Item Recommendation](https://arxiv.org/abs/2204.12176)

## 具体代码：

搭载CPR损失的LightGCN模型CPRLightGCN，适配CPR损失的模块CPRReader, CPRRunner

### 具体CPR模型：

*  [./src/models/general/CPRLigthGCN.py--CPRLightGCNCPR](https://github.com/BinshengQin/Rechorus-master/src/models/general/CPRLightGCN.py): 继承自(LightGCNBase, GeneralModel), 定义实现了CPR损失，并在嵌套类Dataset中实现了CPR损失相关动态采样算法

### 具体Reader、Runner模块：

* [./src/helpers/CPRReader.py](https://github.com/BinshengQin/Rechorus-master/src/helpers/CPRReader.py)：实现了对数据集读取和处理，为训练做准备

* [./src/helpers/CPRRunner.py](https://github.com/BinshengQin/Rechorus-master/src/helpers/CPRRunner.py)：继承自BaseRunner, 实现了训练过程的控制
