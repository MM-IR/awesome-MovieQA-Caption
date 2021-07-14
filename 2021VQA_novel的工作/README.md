# 1.2021 CVPR: Separating SKills and Concepts for Novel Visual Question Answering@Hengji
<img width="765" alt="image" src="https://user-images.githubusercontent.com/40928887/125571515-b27ba839-7d84-45d2-97d3-567dc2ed82e5.png">

>我们针对的是VQA模型的out-of-distribution的问题。VQA模型需要measure generalization to novel questions.

## idea
>这个表达就是说我们想要说人们要想拥有很强大的回答的问题能力，我们一般是解析这个subquestion，然后能够回答subquestion，那么组合起来的能力就很强大。skills and concepts。


1.我们额想法就是把这个问题进行分组以及分解。比如question的话就是有skills以及concepts，skills呢就是visual tasks，比如counting或者attribute recognition。concepts就是questionb中提到的比如objects以及people。

2.VQA模型呢就是需要compose skills and concepts in novel ways, 而不管这个whether the specific composition has been seen in training.**我们需要很好combine，而且不需要管这个training的风格**

我们这里的方法呢就是选择学习去compose这个skills以及concepts（然后这个默认是separates these two factors implicitly within a model通过学习这个grounded concept表示以及分解这个skills和concepts的encoding）

我们这里就是使用一个对比学习的方法，现有的实验证明我们的effectiveness of our approach。

>我们这里就是使用grounding as a proxy to separate concepts so that the model learns to identify a concept in both the question and image, regardless of the specific context.

>1.我们这里还参照弱监督的grounding, 来训练一个model去recover a concept mentioned in a given image-question pair 通过对比这个masked concept word的表达以及words in other questions的表达。

<img width="436" alt="image" src="https://user-images.githubusercontent.com/40928887/125573980-8debad66-618a-452e-8b9b-3e0c52efd20e.png">
