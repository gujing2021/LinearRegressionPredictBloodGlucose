# 无创血糖预测模型测试

导师让接手上一届师姐的遗产，继续研究血糖预测模型。目前刚看了下点机器学习的内容，尝试用机器学习的方法对无创血糖的数据进行简单的预测，主要是利用sklear的线性模型预测无创血糖模型。

## 读取数据



读取的数据是师姐处理过的mat文件，这里只用了`PPP`标签的数据做输入，`TTT`标签的数据做标签，目前数据只有205个，这么小的数据量我觉得没必要上神经网络，简单的使用线性回归尝试一下。

利用scipy.io读取mat文件，根据`PPP`和`TTT`标签提取数据

```python
file_name = "data.mat"
data = scio.loadmat(file_name)
X = data.get("PPP")
X = X.T
y = data.get("TTT")
y = y[0]
print("X.shape :", X.shape, " y.shape: ", y.shape)
```

```shell
X.shape : (205, 4)  y.shape:  (205,)
```

查看数据：

```python
X[1:10,:]
```

```shell
array([[ 1.04333333e+02,  5.80000000e+01,  3.32000000e+01,
        -6.43337135e-01],
       [ 9.50000000e+01,  7.30000000e+01,  3.55000000e+01,
         9.59860109e-02],
       [ 9.40000000e+01,  7.10000000e+01,  3.51000000e+01,
         3.01645487e-01],
       [ 1.03000000e+02,  6.73333333e+01,  3.64666667e+01,
        -4.38108606e-02],
       [ 9.30000000e+01,  6.90000000e+01,  3.64333333e+01,
        -6.66146153e-01],
       [ 1.00000000e+02,  8.10000000e+01,  3.50000000e+01,
         1.00000000e+00],
       [ 8.70000000e+01,  5.60000000e+01,  3.50000000e+01,
         3.18300837e-01],
       [ 8.76666667e+01,  7.53333333e+01,  3.65000000e+01,
         2.24809194e-01],
       [ 9.73333333e+01,  6.96666667e+01,  3.38333333e+01,
        -4.51109746e-01]])
```

```python
y[1:10]
```

```shell
array([6.6, 9.1, 5.1, 5.3, 4.6, 6.3, 8.1, 4.7, 5.2])
```

正常人的血糖浓度值一般在空腹状态下为3.9~6.1 mmol/L，非空腹状态下为3.9~8.0 mmol/L，以上数据可以看出血糖值基本在正常范围。

## 创建测试数据集

将原始数据集分为训练数据集和测试数据集，分割比例为0.8和0.2，即训练数据集为0.8，测试数据集为0.2。

```python
def split_train_test(X, y, test_ratio):
    shuffled_indices = np.random.permutation(len(y))
    test_set_size = int(len(y) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices,:], y[train_indices], X[test_indices,:], y[test_indices]
```

```python
# 分割得到训练集和测试集
train_X, train_y, test_X, test_y = split_train_test(X, y, 0.4)
```

```python
train_X.shape, train_y.shape,test_X.shape, test_y.shape
```

```powershell
((123, 4), (123,), (82, 4), (82,))
```

## 训练数据集

这里选择简单的线性回归模型，采用梯度下降的方法训练数据，利用`sklearn.linear_model`的`LinearRegression`完成训练

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_X, train_y)
```

简单的测试模型的效果

这里选择简单的线性回归模型，采用梯度下降的方法训练数据


```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_X, train_y)
```




    LinearRegression()



以上就是简单的线性回归训练，从原始数据中，随机挑选几个数据检验模型的预测效果


```python
some_data = X[120:125]
some_labels = y[120:125]
print('Predictions:', lin_reg.predict(some_data))
```

    Predictions: [6.16550115 6.25891966 5.47276013 7.38280052 5.2230065 ]



```python
print("Labels:",list(some_labels))
```

    Labels: [4.5, 4.2, 4.8, 8.1, 4.3]



```python
plt.plot(range(5), some_labels, c='r')
plt.plot(range(5), lin_reg.predict(some_data), c='g')
plt.show()
```


​    
![svg](LinearRegressionPredict_files/LinearRegressionPredict_19_0.svg)
​    



```python
predict_y = lin_reg.predict(test_X)
plt.scatter(test_y,predict_y)
plt.show()
```


​    
![svg](LinearRegressionPredict_files/LinearRegressionPredict_20_0.svg)
​    



```python
# clarke网格误差分析
def clarke(y,y_predicted):
    """
    clarke网格分析

    total, percentage = clarke(y, yp)
    
    INPUTS:
    y       Reference values(血糖浓度参考值)
    yp      Predicted/estimated values(血糖预测值)
    
    OUTPUTS:

    """
    # 设置坐标轴范围和坐标轴名称
    plt.xlim(0, 400*0.0556)
    plt.ylim(0, 400*0.0556)
    plt.xlabel('Reference values of Blood Glucose(mmol/L)')
    plt.ylabel('Predicted values of Blood Glucose(mmol/L)')
    # 获取数据长度
    n = len(y)
    # 散点图绘制数据
    plt.scatter(y, y_predicted, s=15)
    # 绘制网线
    # 上半区网线
    plt.plot([0,400*0.0556],[0,400*0.0556],linestyle=':',c='k') 
    plt.plot([0,175*0.0556/3],[70*0.0556,70*0.0556],c='k',linewidth=1)
    plt.plot([175*0.0556/3,400*0.0556/1.2],[70*0.0556,400*0.0556],c='k',linewidth=1)
    plt.plot([70*0.0556,70*0.0556],[84*0.0556,400*0.0556],c='k', linewidth=1)
    plt.plot([0,70*0.0556],[180*0.0556,180*0.0556],c='k',linewidth=1)
    plt.plot([70*0.0556,290*0.0556],[180*0.0556,400*0.0556],c='k', linewidth=1)
    # 下半区网线绘制
    plt.plot([70*0.0556,70*0.0556],[0,56*0.0556],c='k',linewidth=1)
    plt.plot([70*0.0556,400*0.0556],[56*0.0556,320*0.0556],c='k',linewidth=1)
    plt.plot([180*0.0556,180*0.0556],[0,70*0.0556],c='k',linewidth=1)
    plt.plot([180*0.0556,400*0.0556],[70*0.0556,70*0.0556],c='k',linewidth=1)
    plt.plot([240*0.0556,240*0.0556],[70*0.0556,180*0.0556],c='k',linewidth=1)
    plt.plot([240*0.0556,400*0.0556],[180*0.0556,180*0.0556],c='k',linewidth=1)
    plt.plot([130*0.0556,180*0.0556],[0,70*0.0556],c='k',linewidth=1)
    
    # 绘制区域标签A,B,C,D,E
    plt.text(30*0.0556,20*0.0556,'A',fontsize=9);
    plt.text(30*0.0556,150*0.0556,'D',fontsize=9);
    plt.text(30*0.0556,380*0.0556,'E',fontsize=9);
    plt.text(150*0.0556,380*0.0556,'C',fontsize=9);
    plt.text(160*0.0556,20*0.0556,'C',fontsize=9);
    plt.text(380*0.0556,20*0.0556,'E',fontsize=9);
    plt.text(380*0.0556,120*0.0556,'D',fontsize=9);
    plt.text(380*0.0556,260*0.0556,'B',fontsize=9);
    plt.text(280*0.0556,380*0.0556,'B',fontsize=9);

    # 计算数据
    total = np.zeros(5)
    # A区域
    for i in range(0,n):
        if (y[i] <= 70*0.0556 and y_predicted[i] <= 70*0.0556) or (y_predicted[i] <= 1.2*y[i] and y_predicted[i] >= 0.8*y[i]): 
            total[0] = total[0] + 1
        else:
            # E区域
            if(y[i] >= 180*0.0556 and y_predicted[i] <= 70*0.0556) or (y[i] <= 70*0.0556 and y_predicted[i] >= 180*0.0556):
                total[4] = total[4] + 1
            else:
                # 区域C
                if (y[i] >= 70*0.0556 and y[i] <= 290*0.0556) and (y_predicted[i] >= y[i] + 110*0.0556) or (y[i] >= 130*0.0556 and y[i] <= 180*0.0556) and (y_predicted[i] <= (7/5)*y[i] - 182*0.0556):
                    total[2] = total[2] + 1
                else:
                    # 区域D
                    if (y[i] >= 240*0.0556) and (y_predicted[i] >= 70*0.0556) and (y_predicted[i] <= 180*0.0556) or (y[i] <= 175*0.0556/3 and y_predicted[i] <= 180*0.0556) and (y_predicted[i] >= 70*0.0556) or (y[i] >= 175*0.0556/3 and y[i] <= 70*0.0556) and (y_predicted[i] >= (6/5)*y[i]):
                        total[3] = total[3] + 1
                    else:
                        # 区域B
                        total[1] = total[1] + 1

    percentage = (total/n)*100
    plt.show()
    return total, percentage
```


```python
total, percentage = clarke(test_y, predict_y)
```


​    
![svg](LinearRegressionPredict_files/LinearRegressionPredict_22_0.svg)
​    



```python
total
```




    array([56., 26.,  0.,  0.,  0.])



数据点基本落入克拉克网络A,B区域，但是在B区域的点有点过多，准确率不是很高


```python
percentage
```




    array([68.29268293, 31.70731707,  0.        ,  0.        ,  0.        ])





