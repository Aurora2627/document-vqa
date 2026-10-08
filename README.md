# 基于 LLaVA 的文档视觉问答与证据定位

状态：LLaVA 方案与目录初始化完成，模型实现/正式实验尚未开始。详细设计见 [项目方案](PROJECT_PLAN.md)。

方向：多模态/CV

对应需求：滴滴 Voyager VLM、文档解析、数据与评测 JD

## 任务

输入单页文档或图表与问题，给出短答案和支持它的页码或区域证据。

## 实现阶段

1. 单页数据和统一问答格式
2. OCR+文本模型与 VLM 零样本基线
3. 表格/图表/文字类型的分类别评测
4. 分辨率及裁剪策略对比
5. 外部 GPU LoRA SFT；可选扩展多页检索

## 第一阶段起点

先从 DocVQA 选可用的官方子集。没有原生区域标签时，只在人工标注的小评测集做定位评测，不伪造区域真值。图表数据作为后续扩展。

## 实验与验收

- 答案准确率/ANLS（按数据集官方规范）
- 证据区域 IoU 或人工证据正确性评估
- 阅读/表格/图表分类别结果
- 耗时、峰值内存；分辨率/裁剪/微调消融

## 运行环境

数据准备、OCR、评测在 Mac；模型实现坚持 Python/PyTorch，所有运行从 VS Code 启动。用户可租用云端服务器；7B 推理及 LoRA 默认使用云端 NVIDIA GPU；16GB Mac 的模型推理、量化兼容性和速度尚未验证。当前系统 Python 3.9 不满足最新版 bitsandbytes 的 Python >=3.10 要求，远程环境单独配置。

不预填效果提升数据；只有可复现的真实结果才进入简历。

## 参考

- https://huggingface.co/datasets/lmms-lab-encoder/DocVQA
- https://github.com/LLaVA-VL/LLaVA-NeXT
- https://huggingface.co/docs/transformers/model_doc/llava_next
