# MLAction
机器学习常用方法代码及注释。

## KNN(K近邻)：
邻近算法，或者说K最近邻(kNN，k-NearestNeighbor)分类算法是数据挖掘分类技术中最简单的方法之一。
所谓K最近邻，就是k个最近的邻居的意思，说的是每个样本都可以用它最接近的k个邻居来代表。

kNN算法的核心思想是如果一个样本在特征空间中的k个最相邻的样本中的大多数属于某一个类别，
则该样本也属于这个类别，并具有这个类别上样本的特性。
该方法在确定分类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。 
kNN方法在类别决策时，只与极少量的相邻样本有关。由于kNN方法主要靠周围有限的邻近的样本，
而不是靠判别类域的方法来确定所属类别的，因此对于类域的交叉或重叠较多的待分样本集来说，kNN方法较其他方法更为适合。


### KNN算法使用步骤
对未知属性类别的数据，依次执行以下操作：
* 计算已知类别属性的数据集中的每个点与当前点的举例
* 按照举例递增次序排序
* 选取与当前点最近的k个点
* 确定钱k个点所在类别的出现频率
* 选择出现频率最高的点的类别，最为该点的预测类别

### 使用matplotlib绘图示例

![](https://gitee.com/tantexian/mlaction/raw/master/docs/static/knn_1.png)

* 纵轴表示“每周所消费的冰淇淋公升数”
* 横轴表示“玩视频游戏所耗时间百分比”
* PS：从上图中得出似乎消费冰淇淋的人数，和玩视频游戏百分百没有直接关联关系。

### 使用knn算法python识别手写体测试结果：

手写体图片8，经过滤波，灰度化，二值化等一系列图像处理算法之后得到下图矩阵向量：

![](https://gitee.com/tantexian/mlaction/raw/master/docs/static/knn_2.png)


```
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9

the classifier came back with: 9, the real answer is: 9

Ran 1 test in 28.333s

OK

the total number of errors is: 11

the total error rate is: 0.011628
```

错误率为1.1628%。

### KNN算法注意要点
* 对于多维特征，如果特征的度量不一致，需要做归一化，ex：将所有特征值归一到0~1之间。
* 两个样本之间的距离，可是使用欧式距离计算。
* k近邻必须保存全部数据集，如果训练集很大，需要占用很大的存储空间。
* 由于必须对数据集中的每个数据计算距离值，实际使用时，可能非常费时。




## 决策树

决策树(Decision Tree）是在已知各种情况发生概率的基础上，通过构成决策树来求取净现值的期望值大于等于零的概率，
评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。由于这种决策分支画成图形很像一棵树的枝干，故称决策树。
在机器学习中，决策树是一个预测模型，他代表的是对象属性与对象值之间的一种映射关系。
Entropy = 系统的凌乱程度，使用算法ID3, C4.5和C5.0生成树算法使用熵。这一度量是基于信息学理论中熵的概念。