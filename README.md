# awesome-MovieQA
关于近几年顶会顶刊Movie有关的工作

我个人觉得本质上来讲很多都是interaction，而不一定是relationship。

## 1.CVPR2020@Learning Interactions and Relationships between Movie Characters
这篇文章就是指出关系有的从视觉从可以推断，有的从dialog中推断，有的需要两者的fusion才可以得到。

很多关系都是bottom-up的，而不是parent那种top-down的。
其中interaction以及relationship都是分开讨论的，interaction表示action，然后relationship就是表示关系～我们必须同时建模才可以理解。（因为关系也改变的）

![](MovieInteraction.png)

### Motivation:这里就是探索两个问题
1.can learning to jointly predict relationships and interactions help improve the performance of both?
2.can we use interaction and relationship labels at the clip or movie level and learn to identify the pair of characters involved?

### 关于问题2的核心就是将其当作是一个weak track prediction.这里就是使用label，我们能否找到pair of characters呢？


## Dual Hierarchical Temporal Convolutional Network with QA-Aware Dynamic Normalization for Video Story Question Answering@MM20
### Motviation:
1.先前的注意力网络很热火，但是他们的工作忽视了一些重点:只是考虑single temporal scale。这个就是会忽视可能一些相关的segments of videos/subtitle sequences会表示成不同的temporal scales for different samples。@@temporal granularities。

2.目前的多模态交互都是多个vector直接fuse，而忽视了那种dynamic and finegrained interactions between each word and each video feature unit。

3.先前的方法并没有fully探索QA pairs的信息。shallow exploitation of question and answer choices。


### MovieQA比起VideoQA的挑战
1）视频包含了更多的diverse information，比如背景噪声/flow of actions/所以特征空间更大而且更加复杂@比起text或者image而言。

2）多个异质的模态，比如video and subtitle。

3）video和subtitle sequences往往是非常长的，定位相关的segment是十分困难的事情。

### 1. Input Embedding
video feature就是resnet152extract，然后对于每个video@3 fps的feature进行L2 Norm，然后project into 512D。当然我们也用detected object labels而不是图像特征来作为输入。

对于textual input，比如subtitle和QA pairs，就是BERT来encode。然后Q和A进行concatenate。

### 2.多模态Alignment&Temporal Modeling
就是multihead(normalize)+BILSTM来进行融合啦。～～～套路比较多。x,y,x.y

### 3.QA-Aware Dynamic Normalization








