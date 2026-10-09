# 基于 LLaVA 的文档视觉问答与证据定位

状态：v0.3.0 已完成模型实现、本机 MPS 兼容性测试及真实 DocVQA 小开发集零样本/分辨率对照；真实数据微调、证据定位和云端 7B 微调尚未完成。此项目作为第二个多模态项目，沿用 document-vqa 仓库；不增加第三个多模态简历项目。

## 问题与输出

输入单页文档图片及问题，例如“这份发票的日期是什么？”；生成简短答案，显示支持答案的文字片段及其页面区域。首先处理英文单页文档；中文、多页、图表作为后续独立扩展，不能在初版混用数据和指标。

回答由视觉语言模型产生。证据候选可以来自 OCR 文本及框坐标，系统需明确区分“模型回答”和“系统定位的候选证据”，不能把关键词匹配当作模型原生 grounding。无法找到可靠证据时，标记证据未确认。

## 模型原理与候选

LLaVA 连接视觉编码器、投影模块和自回归语言模型：图片的视觉 patch 特征映射到语言模型可消费的表示，再和问题一起输入语言模型生成答案。项目使用 Hugging Face Transformers/PyTorch。当前本机使用官方 LLaVA-OneVision-0.5B，固定 revision 74dd0bf867a4cda7950c17663794267c60cf4b40；LLaVA-NeXT 的 llava-hf/llava-v1.6-mistral-7b-hf 作为云端扩展候选。

LLaVA-1.5 可用于原理复现与分辨率对照；LLaVA-NeXT 的高分辨率处理适合研究文档小字。它们是研究基座，选用本身不构成原创方法或最新模型优势。

## 执行阶段

1. 建立统一 JSONL：question_id、document_id、image、question、answers、split；使用官方 DocVQA 数据及划分，先做可复现小子集，记录来源、授权条件和哈希。按文档隔离训练、验证及评测，禁止同页问题跨划分造成泄漏。
2. 跑通 PyTorch 单图推理，固定 processor/chat template，记录预测、时延与内存。先从 20 条开发样例检查格式与明显错误，不能将它作为最终测试集。
3. Baseline：OCR 文本检索/文本问答、LLaVA 零样本直接回答。分别报告 OCR 系统组件版本与模型版本。
4. 方法候选：问题驱动的区域裁剪、整页与局部联合输入、OCR 辅助提示、证据约束输出。控制输入成本并逐项消融，重点研究小字阅读错误及答案缺少依据。
5. 云端 GPU 上开展 LoRA 指令微调（用户已说明可租用服务器，此阶段列为项目主线）；只在官方训练划分构造图片—问题—短答案样本，验证集选择超参数。训练损失遮蔽用户输入、图片占位及 padding，只监督助手答案；检查有效监督 token、梯度与 adapter 保存/重载。QLoRA 是待环境验证的选项。
6. 独立评测、错误分析与演示：锁定方法后评测保留集，保存全量预测和失败案例，不挑选测试样例或伪造性能提升。

## 评测与验收

- 目标是对齐官方 ANLS；当前仅报告明确规范化规则的 ANLS 风格分数，尚未验证官方 evaluator 一致性。另报规范化 Exact Match，保留多答案逻辑。
- 区域定位仅在具有真实区域标注的集合上报告 IoU。缺少区域真值时，建立人工审核小集，标注规范、数量、来源和评审结果单独报告。
- 对比整页、裁剪、OCR 辅助、微调，并报告延迟、峰值内存和输入图片/视觉 token 成本；不能把更多输入或更大计算量的收益全部归因于微调。
- SFT 前后使用相同模型、生成参数和评测集；同时检查领域外能力是否退化。
- 源码、配置、输入哈希、日志、权重和预测按每轮独立保存。模型实现与运行坚持 Python/PyTorch，全部从 VS Code 启动。

## Mac 与外部 GPU

现有 MacBook 16GB 和系统 Python 3.9 可先用于数据整理、评测开发及可用的小规模检查。不能据已有分类头 MPS 成功推断 LLaVA 7B 能运行。

7B 语言模型仅 FP16 权重约 14GB（参数量×2 字节的粗略估算），还需视觉编码器、KV cache、临时张量与系统内存，故不把本机完整半精度推理或训练作为默认方案。用户尚未租用云端服务器。本机已验证 OneVision 小模型的实际推理和 LoRA 流程。正式 7B 推理及 LoRA 默认使用云端 NVIDIA GPU；显存需求由模型、分辨率、序列长度、batch、精度和优化设置实测。

最新 bitsandbytes 文档要求 Python >=3.10，当前本机 Python 3.9 不满足这一要求；量化后端和 Transformers/PEFT 的兼容性也需单独验证。不得静默替换用户的系统 Python，不得把 MLX/llama.cpp 运行报告成 PyTorch。远程训练环境单独记录解释器及依赖，不改变当前本地项目环境。云端可通过 VS Code Remote SSH 调试和启动训练，Python/依赖以选定模型的兼容要求固定。先试跑单批次前向/反向、保存峰值显存，再确定分辨率、batch 和 GPU；服务器供应商、GPU 型号、预算尚未确定，尚未租赁或启动付费资源。

## 可学习与展示的能力

视觉 token 与多模态对齐；自回归生成；chat template/processor；数据构建；SFT、LoRA、标签遮蔽；分辨率与裁剪；OCR 辅助及证据定位；幻觉分析；ANLS 与系统评测；显存与推理成本；可复现实验。

只有实际完成的部分才进入简历。当前已完成本机小模型软件验证及真实文档开发集对照，不能声称已完成真实 DocVQA 微调增益、证据定位或部署。

## 官方资料（2026-10-08 核验）

- 原始 LLaVA 与训练说明：https://github.com/haotian-liu/LLaVA
- LLaVA-NeXT：https://github.com/LLaVA-VL/LLaVA-NeXT
- Transformers 接口：https://huggingface.co/docs/transformers/model_doc/llava_next
- 候选模型：https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
- 官方 DocVQA：https://www.docvqa.org/datasets/docvqa
- bitsandbytes 要求：https://huggingface.co/docs/bitsandbytes/main/en/installation

## 本机验证记录

见 reports/local-validation-v0.2.0.md。单步损失下降与合成样例答对只说明运行流程，不是正式数据的能力提升。所有失败测试及修正后的新目录都保留。

真实文档开发结果见 reports/docvqa-development-v0.3.0.md；512 长边改善小字问答，但 20 题顺序子集不能替代最终评测。
