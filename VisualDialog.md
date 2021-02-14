首先关于Visual Dialogue而言，这个任务就是关于一个图像由多个questions，而且这个questions会问的东西考虑非常全面，比如any objects，relationships，semantics。

## 关键
就是学习一个非常全面而且富有语义的图像表达并且能够针对不同的问题进行很好的自适应。

# DualVD: An Adaptive Dual Encoding Model for Deep Visual Understanding in Visual Dialogue

## 1.Motivation
1.就是针对事实上对于Visual Dialog而言，我们简单地图像表达@monolithic feature是不够的，因为我们想要的visual content可能会有很大的改变针对不同的问题。

2.我们就是思考人类大脑处理图像的过程，一般按照两个分支，1.会想到image的object和relationship的关系@visual 2.想到对应的higher-level abstraction@semantic view。

## 2.Contribution
1.我们就是
