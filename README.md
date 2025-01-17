
# 机器学习大作业--在[ReChorus](https://github.com/THUwangcy/ReChorus)框架下复现论文：[Cross Pairwise Ranking for Unbiased Item Recommendation](https://arxiv.org/abs/2204.12176)

## 具体代码：

搭载CPR损失的LightGCN模型CPRLightGCN，适配CPR损失的模块CPRReader, CPRRunner

### 具体CPR模型：

*  [./src/models/general/CPRLigthGCN.py--CPRLightGCN](https://github.com/BinshengQin/Rechorus-master/blob/main/src/models/general/CPRLightGCN.py): 继承自(LightGCNBase, GeneralModel), 定义实现了CPR损失，并在嵌套类Dataset中实现了CPR损失相关动态采样算法

### 具体Reader、Runner模块：

* [./src/helpers/CPRReader.py](https://github.com/BinshengQin/Rechorus-master/blob/main/src/helpers/CPRReader.py)：实现了对数据集读取和处理，为训练做准备

* [./src/helpers/CPRRunner.py](https://github.com/BinshengQin/Rechorus-master/blob/main/src/helpers/CPRRunner.py)：继承自BaseRunner, 实现了训练过程的控制

## 实验结果复现

若要复现我们报告中的实验结果，请运行以下命令

### Grocery_and_Gourmet_Food数据集下复现

#### CPRLightGCN

```shell
python src\main.py --model_name CPRLightGCN  --num_workers 0 --batch_size 128 
```
#### LightGCN

```shell
python src\main.py --model_name LightGCN  --num_workers 0 --batch_size 128 
```
#### NeuMF

```shell
python src\main.py --model_name NeuMF  --num_workers 0 --batch_size 128 
```

### ML_1MTOPK数据集下复现

#### CPRLightGCN

```shell
python src\main.py --model_name CPRLightGCN  --num_workers 0 --batch_size 512 --dataset MovieLens_1M/ML_1MTOPK 
```
#### LightGCN

```shell
python src\main.py --model_name LightGCN  --num_workers 0 --batch_size 512 --dataset MovieLens_1M/ML_1MTOPK 
```
#### NeuMF

```shell
python src\main.py --model_name NeuMF  --num_workers 0 --batch_size 512 --dataset MovieLens_1M/ML_1MTOPK 
```
