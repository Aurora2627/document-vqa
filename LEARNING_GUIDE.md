# 完成实验后如何学习这个项目

目标是能够从一张图片、一条问题跟踪到答案、损失、参数更新和评测结论。先阅读本轮报告，再沿下面顺序调试；每次只开一个 MPS 模型任务。全部入口见 [VS Code 说明](VSCODE_START.md)。

## 1. 图片如何进入语言模型

阅读 `src/modeling.py`、`src/batching.py`、`scripts/infer.py`。在 processor 后看 `input_ids`、`pixel_values`、`image_sizes`，在模型前向中观察视觉编码器、projector 和语言层。记录图片占位、视觉特征与序列长度如何对应。先用微型结构检查理解张量，再看预训练模型；随机结构验证不提供任务效果。

能够回答：LLaVA 各模块负责什么？为什么图片尺寸和 processor 路径会影响视觉 token？单图 anyres 和原生多图为何不能只按图片张数比较？

## 2. 训练到底监督什么

阅读 `src/batching.py` 的 `build_example` 和 `answer_loss`。检查图片、用户问题、模板和 padding 的 labels 为 -100，只监督助手答案与 EOS。跟踪 next-token shift，理解为什么答案第一个 token 由它前面的上下文预测。对照完整 causal loss 的自动检查。

能够回答：为什么不能把问题也当训练目标？为什么不能直接截断包含图像占位的序列？只计算答案位置 logits 如何降低内存，又如何验证损失等价？

## 3. LoRA 更新了什么

阅读 `src/lora.py`，推导 `W(x) + alpha/rank * B(A(x))`。观察 B 为零时输出等价，更新后 B 不再为零。根据 q_proj/v_proj 的实际输入输出维度计算 540,672 个训练参数，注意 grouped-query attention 的 q/v 输出维度不同。

能够回答：为什么冻结原权重仍然需要反向传播？rank 和 alpha 如何影响增量？本项目冻结了哪些模块？如何验证 adapter 真更新、能重载，而不是仅保存了一个文件？

## 4. 为什么本机能训练，又为什么较慢

阅读 `scripts/train_lora.py` 的梯度检查点、梯度累积、有限值检查和裁剪。追踪最后不足四条样本的累积窗口。非重入检查点通过重算激活降低内存；训练时关闭 cache。基座半精度、低秩参数单精度，全部仍是 PyTorch。

能够回答：冻结参数为何仍有激活内存？gradient checkpointing 与 gradient accumulation 的作用有什么区别？MPS 与 CUDA 的实际验证边界在哪里？当前结果只支持已实测的 Mac 小模型路径。

## 5. 如何证明效果

阅读 `src/data.py`、`scripts/prepare_expanded_sft.py`、`src/evaluation.py`、`src/paired_statistics.py`。训练、验证和保留评测按源文档隔离；同文档多题不能跨划分。验证 NLL 选模型，保留集生成衡量效果。比较相同图片分辨率下的原模型和 LoRA。

能够回答：为什么训练 loss 下降不等于问答提升？ANLS 风格与 EM 有何差别？为何逐题配对 bootstrap？为什么旧、新 LoRA 的差值不能全部归因于数据量？为什么 20 文档开发 OCR 分数不能写成 64 文档独立评测分数？

## 6. 如何解释区域阅读的成功与失败

阅读 `src/region_reading.py`、`src/ocr_regions.py` 和两个消融入口。查看同页换问题后的候选区域、原图裁剪及 OCR 横带。先与整页基线比，再与相同三图路径的固定/错配区域比；OCR 还要与无图片的文字问答比。

能够回答：受限九数字 softmax 为什么不是证据置信度？BM25 匹配为什么不是学会 grounding？裁剪后小字放大为何仍可能变差？图文融合答对为什么不能证明定位正确？没有真实证据框时能报告什么、不能报告什么？

面试介绍以自己能解释、复现和修改的内容为准。项目中的负面结果也应保留：说明对照设计、观察到的事实、目前无法确定的机制，再提出下一次可检验的实验，不编造收益或原创性。
