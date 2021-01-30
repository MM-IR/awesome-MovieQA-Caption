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
