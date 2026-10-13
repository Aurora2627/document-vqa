# LLaVA 参数高效微调与文档视觉问答

输入单页文档图片和问题，使用 LLaVA-OneVision 生成简短答案。项目研究答案监督 LoRA、输入分辨率、问题驱动区域阅读与 OCR 辅助对问答质量的影响。

## 实现

- 固定官方 OneVision 0.5B checkpoint 与 revision；使用 Transformers/PyTorch。
- 原生 PyTorch LoRA：语言模型 q/v 低秩更新，冻结视觉编码器和基座；答案与 EOS 监督、梯度检查点、梯度累积、验证 NLL 选模和 adapter 重载核验。
- 按源文档隔离数据，比较零样本/LoRA × 输入分辨率；统计规范化 EM、ANLS 风格分数与文档配对 bootstrap。
- 九格区域排序、整页/局部多图融合、OCR 横带检索与图文消融；候选区域未经过真实框标注验证。
- 上传文档问答演示，以及源码快照、输入哈希、配置、日志和逐题结果记录。

## 结果与边界

100 个新源文档的固定权重四组对照：

| 配置 | EM | ANLS 风格 |
|---|---:|---:|
| 原模型 / 512 | 54/100 | 0.7055 |
| 原模型 / 768 | 60/100 | 0.7854 |
| 已有 LoRA / 512 | 62/100 | 0.7442 |
| 已有 LoRA / 768 | 67/100 | 0.8236 |

分辨率主对照的 ANLS 差值为 +0.0799，配对 bootstrap 95% CI [0.0272, 0.1363]；EM 差值区间跨零。详见 [独立评测报告](reports/independent-eval-v0.6.0.md)。

后续扩展至 122 题/64 源文档训练、24 文档验证、64 文档新保留评测。验证 NLL 0.9503→0.8297，但新保留集原模型/旧 LoRA/新 LoRA 为 38/37/37 题；不能宣称扩展训练带来稳定收益。旧 20 文档开发集上，整页/九格融合/OCR 裁剪/图文分别为 12/6/12/8 题。详见 [扩展研究报告](reports/deepening-v0.7.0.md) 与 [失败分析](reports/failure-analysis-v0.7.0.md)。

数据为公开 DocVQA validation 镜像的小规模子集，不是官方盲测或完整榜单；ANLS 风格实现尚未验证与官方 evaluator 完全一致。旧评测结果已被观察，后续调参需要新的最终评测集。硬件与精度按各版实验报告记录，以上历史基准未在当前 CUDA 环境完整复跑。

## 运行

当前环境：Windows、RTX 4060 Laptop 8GB、Python 3.9.18、PyTorch 2.8.0+cu126、Transformers 4.57.6。模型入口默认 CUDA；不可用时明确报错。已验证预训练 LoRA 更新、fixture 训练/推理、新进程重载与 OCR，16 项测试通过；这是流程兼容性验证，不产生新的任务效果指标。

[VS Code 启动](VSCODE_START.md) · [当前验证范围](CURRENT_STATUS.md) · [CUDA 检查记录](reports/v0.7.1/compatibility.json) · [问答演示](DEMO.md)

代码入口：

- src/modeling.py、src/batching.py：官方模型、processor 与答案监督。
- src/lora.py、scripts/train_lora.py：LoRA 与训练。
- scripts/evaluate.py、src/paired_statistics.py：评分与配对统计。
- scripts/run_independent_eval.py、scripts/run_expanded_study.py：固定协议对照。
- src/region_reading.py、src/ocr_regions.py、scripts/run_ocr_study.py：区域/OCR 实验。
- scripts/demo.py：文档问答页面。

原生 adapter.pt 不是 PEFT 格式。图片、逐题标签/预测、模型权重、虚拟环境和本机配置不上传；复现实验需按数据说明准备资源并使用新的输出目录。

[模型导读](MODEL_WALKTHROUGH.md) · [学习路线](LEARNING_GUIDE.md) · [项目方案](PROJECT_PLAN.md) · [版本历史](VERSION_HISTORY.md)
