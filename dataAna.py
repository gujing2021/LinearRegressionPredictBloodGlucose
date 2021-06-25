"""
辅助的一些相关函数
@author: madao33
@date:2021-04-06 19:21:31
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def clarke(y, y_predicted, s_size=20):
    """
    clarke网格分析

    total, percentage = clarke(y, yp)
    
    INPUTS:
    y       Reference values(血糖浓度参考值)
    yp      Predicted/estimated values(血糖预测值)
    
    OUTPUTS:
    total   A，B，C，D，E五个区域落入点的总数，为一个(5, 1)的np.array
    percentage  分别是五个区域落入点数占总数的百分比
    """
    # 设置坐标轴范围和坐标轴名称
    plt.figure(figsize=(10, 10))
    plt.xlim(0, 400*0.0556)
    plt.ylim(0, 400*0.0556)
    plt.xlabel('Reference values of Blood Glucose(mmol/L)')
    plt.ylabel('Predicted values of Blood Glucose(mmol/L)')
    # 获取数据长度
    n = len(y)
    # 散点图绘制数据
    plt.scatter(y, y_predicted, s=s_size)
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
    plt.text(30*0.0556,20*0.0556,'A',fontsize=9)
    plt.text(30*0.0556,150*0.0556,'D',fontsize=9)
    plt.text(30*0.0556,380*0.0556,'E',fontsize=9)
    plt.text(150*0.0556,380*0.0556,'C',fontsize=9)
    plt.text(160*0.0556,20*0.0556,'C',fontsize=9)
    plt.text(380*0.0556,20*0.0556,'E',fontsize=9)
    plt.text(380*0.0556,120*0.0556,'D',fontsize=9)
    plt.text(380*0.0556,260*0.0556,'B',fontsize=9)
    plt.text(280*0.0556,380*0.0556,'B',fontsize=9)

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


def showPredict(true_y, predict_y):
    """
    绘制真实值和预测值的差别
    @params true_y: 真实的血糖预测值，predict_y-预测的血糖值
    @return None
    """
    plt.figure(figsize=(20, 10))
    plt.plot(true_y, color='g', linestyle='-', marker='o', markersize=10,label='true value')
    plt.plot(true_y*0.8, color='g', linestyle='-.', label='-20% line')
    plt.plot(true_y*1.2, color='g', linestyle='-.', label='+20% line')
    plt.plot(predict_y, color='r', linestyle='-.', marker='^', markersize=10, label='predicted value')
    plt.title("NIR predicted value and True value")
    plt.legend()
    plt.show()


def ecg_fft_ana(signal, sampling_rate):
    """
    查看信号的频谱
    @params signal: 时域信号 sampling_rate: 采样率
    @return frq: 频域范围，fft_signal: 频域频谱
    """
    fs = sampling_rate
    ts = 1.0/fs
    t = np.arange(0, 1, ts)
    n = len(signal)
    k = np.arange(n)
    t = n/fs,
    frq = k/t
    frq = frq[range(int(n/2))]
    fft_signal = np.abs(fft(signal))[range(int(n/2))]
    return frq, fft_signal
