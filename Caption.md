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
