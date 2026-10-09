# LLaVA 文档视觉问答与轻量微调

状态：v0.3.0 已在 MacBook MPS 上完成真实 DocVQA 文档的零样本基线与分辨率对照。PyTorch LoRA 流程已通过合成样例验证，真实数据微调、定位与 OCR 辅助尚未完成。

## 当前任务

输入单页文档图片和问题，生成简短答案。围绕 LLaVA 的视觉编码器、投影模块、语言模型、图像 token、指令格式与微调设计可复现实验。先做模型结构复现和运行验证，再使用正式数据比较零样本/LoRA、分辨率与裁剪策略。

## 模型与设备

- 本机起点：官方 `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`，revision `74dd0bf867a4cda7950c17663794267c60cf4b40`，权重 1,787,445,680 字节，下载后与官方 LFS SHA256 校验一致。0.5B 是语言基座规模，视觉编码器也占参数和内存。
- 全部运行使用 `/usr/bin/python3`、Python 3.9、PyTorch 2.8、Transformers 4.57.6，从 VS Code 启动。MPS 上基座 Float16、LoRA Float32。
- LLaVA-NeXT 7B 的代码接口已准备，真实推理/云端微调尚未验证；需要时再使用云端 GPU。不是所有 LLaVA 规模都能在 16GB Mac 上运行。

## 已验证的内容

1. 官方 Transformers 微型随机 LLaVA：视觉 token 数匹配、投影和语言模型前向、反向、冻结视觉编码器、LoRA 与 projector 更新、精确重载。仅用于结构验证，不代表预训练性能。
2. 官方预训练 OneVision 小模型：在合成发票软件样例上读取金额，生成耗时约 3.40 秒（不含模型加载与输入预处理），不能当作正式任务性能。
3. 同一预训练模型：本机 MPS LoRA 单步更新，答案损失 0.04254→0.03535；270,336 可训练参数；通过新进程重新加载基座和 adapter 的损失一致性检查。
4. 完整 train_lora.py 入口：在一条合成训练样本和一条独立合成验证样本上运行两轮，保存按验证 NLL 选择的 adapter。这里只证明训练流程可运行。
5. 九项自动检查通过，覆盖答案标签遮蔽、LoRA 初始等价与冻结参数、图像 token 合约、答案后缀损失与完整因果损失一致、文档跨划分泄漏检查、ANLS。

6. 20 份真实 DocVQA 开发文档：输入长边上限 384→512，规范化完全匹配 5/20→10/20，ANLS 风格得分 0.4844→0.7367。只作小样本开发诊断，不代表完整榜单或微调增益。详见 [真实文档开发实验](reports/docvqa-development-v0.3.0.md)。

完整说明：[本机验证报告](reports/local-validation-v0.2.0.md)。源码和每轮快照、配置、输入哈希、环境、日志、权重及预测保留。原始数据、合成 fixture、模型权重和本机配置不上传 GitHub，合成样例从保存的 Python 源码重建。

## 代码入口

- `src/modeling.py`：官方模型加载、微型结构配置。
- `src/lora.py`：直接使用 PyTorch 实现 LoRA；语言模型 q/v 线性层的低秩增量，默认冻结视觉与 projector。
- `src/batching.py`：官方生成前缀＋答案＋EOS，仅监督答案 token，拒绝不匹配或静默截断。只生成答案位置需要的 vocabulary logits，减少无效内存。
- `scripts/infer.py`：零样本/adapter 推理，整页或固定裁剪，记录耗时和 token 数。
- `scripts/train_lora.py`：MPS/CUDA LoRA 训练、梯度累积、验证 NLL 选模型。CUDA 路径尚未实际运行。
- `scripts/prepare_docvqa.py`：真实开发数据下载、源文档去重、版本观察及图片哈希。
- `scripts/evaluate.py` / `src/evaluation.py`：ANLS 风格分数、规范化 Exact Match、题型分析及严格逐题对照；官方 evaluator 一致性尚未验证。

这个 LoRA checkpoint 是本项目的 `.pt` 格式，不是 PEFT 的 adapter 格式。

运行见 [VS Code 启动说明](VSCODE_START.md)，原理见 [模型与代码导读](MODEL_WALKTHROUGH.md)，后续研究设计见 [项目方案](PROJECT_PLAN.md)。
