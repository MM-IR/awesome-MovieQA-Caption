# Video Captioning@2018 CVPR
这个显然是一个multimodal transfer的工作，但是事实上learning an effective mapping from the visual sequence space-》language space是一个挑战性的任务，因为long-term multimodal dependency modalling+semantic misalignment

## 1. M^3: Multimodal Memory Modelling for Video Captioning
### 1.Introduction
1.看到对于long-term 序列信息而言，记忆建模是非常有意义的。并且working memory是visual attention的关键点。

目前现有的方法比如一个single视觉表达对于capture所有的信息over a 长期～是不好的。
### 2.novelty
我们就是提出了一个M^3(多模态记忆建模)去描述视频，build a visual and textual shared memroy去建模长时间的视觉文本依赖以及进一步指导视觉注意力到描述的targets来解决视觉textual alignments。

1.这里就是我们是first model多模态数据通过选择 read/write both 视觉内容以及句子内容with a共享的记忆结构，并且将其在视频caption中进行使用。

### 3.Multimodal Memory的解析
因为我们这里存在bimodal信息，i.e., video and language, 所以我们使用两个独立的read/write操作去指导信息交互。

## 2. Relational Graph Learning for Grounded Video Description Generation@MM 2020
ZJU@Grounded video Description(GVD)这个就是caption model能够动态决定合适的video regions并且产生对应的description。这个就可以帮助解释我们的caption model的决策，而且prevent the model from hallucinating object words(这个就是对于目标和单词的幻听～)

### 1.Video Caption现有的工作总结
1.现在的caption模型的一个主要的缺点就是objects hallucination。caption模型就会产生一些描述objects根本不在视频中出现的，因为相似的语义contexts或者pre-extracted priors during 训练阶段。

2.GVD这个任务就是grounded video Description就是尝试改进grounding能力。这个的想法就是我们学习ground related video regions来预测下一个word。那么这样的设置就可以teach models 仅仅explicitly依赖对应的evidence来产生对应的描述。

3.而目前的GVD的方法有一些limitation:
```
1)我们只是关注related regions of objects，而忽视了对应的fine details，比如related objects或者attributes of grounded object。（这样的话我们很容易产生一个coarse-grained 描述性的句子）

2)尤其有些语句比如climb up or down本身就是推理的结果@序列化的frames，而没有具体的spatial regions可以在words中找到。所以我们仅仅依赖视觉某个region的信息可能会产生不合适的word generation。
```

4.因此一个对于GVD（Grounded Video Description）的一个很重要的优化方向就是尝试产生更佳finegrained的信息@@fine/correct。

## 我们为什么想要使用scene graph呢？
1.SG可以提供一个abstraction of objects and their complex relationships。那么就可以表示非常fine-grained 的信息了。SG在一方面提供了互补的信息@帮助caption model来生成fine-grained phrases。比如man in blue shirt。

2.SG可以帮助ground 正确的relation word

3.




