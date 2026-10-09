# LLaVA 文档视觉问答与轻量微调

状态：v0.6.0 新增 100 份独立文档评测、分辨率 × LoRA 四组对照、配对 bootstrap、失败案例复核和本地问答页面。十四项代码检查通过。此前 20 题保留评测未见 LoRA 收益；本轮新子集观察到正向收益，不能代替完整官方榜单。

## 最新结果

固定模型与已有 adapter 后，从公开 DocVQA validation 镜像选择 100 个新源文档，每文档一道题；排除全部既有真实训练、验证、保留评测及开发文档，并检查图片哈希。候选池有限，不是官方盲测。

| 配置 | 完全匹配 | ANLS 风格 |
|---|---:|---:|
| 原模型，整页 512 | 54/100 | 0.7055 |
| 原模型，整页 768 | 60/100 | 0.7854 |
| 已有 LoRA，整页 512 | 62/100 | 0.7442 |
| 已有 LoRA，整页 768 | 67/100 | 0.8236 |

主对照的分辨率 ANLS 差值为 +0.0799，配对 bootstrap 95% 区间 [0.0272, 0.1363]；EM 增加 6 个百分点，区间 [-1, 13] 个百分点，仍存在不确定性。其他对照是探索性分析，未校正多重比较。本轮没有重新训练或按新评测选择参数。

详见 [独立评测报告](reports/independent-eval-v0.6.0.md) 与 [失败案例复核](reports/failure-analysis-v0.6.0.md)。评测后这些文档也成为已知数据，后续调参需另设新评测集。

本地页面支持上传图片、输入问题和查看真实模型回答，见 [演示说明](DEMO.md)。默认使用原模型，512/768 可选；没有 OCR 或证据定位。

真实 HTTP 问答已在 MPS 核验：页面可访问、无效请求拒绝、真实图片回答与保存的 512 基线一致，见 [演示核验记录](reports/demo-validation-v0.6.0.json)。

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
5. 十二项自动检查通过，覆盖答案标签遮蔽、LoRA 初始等价与冻结参数、图像 token 合约、答案后缀损失与完整因果损失一致、文档跨划分泄漏检查、ANLS。

6. 20 份真实 DocVQA 开发文档：输入长边上限 384→512，规范化完全匹配 5/20→10/20，ANLS 风格得分 0.4844→0.7367。只作小样本开发诊断，不代表完整榜单或微调增益。详见 [真实文档开发实验](reports/docvqa-development-v0.3.0.md)。

7. 真实训练 59 题（32 源文档）、验证 12 题、保留评测 20 题；两轮 LoRA，验证 NLL 0.558→0.498，验证完全匹配 9/12→10/12，保留评测均为 14/20。详见 [真实 LoRA 对照](reports/real-docvqa-lora-v0.4.0.md)。

8. 固定裁剪与重复整页三图使用同一提示和 token 数对照；裁剪 7/20、重复 3/20、整页 512 为 10/20。整页 768 提高到 12/20，仅为已查看开发集诊断。详见 [区域阅读与分辨率对照](reports/document-views-v0.5.0.md)。

9. 100 个新文档四组对照：原模型/已有 LoRA × 512/768；完整预测、逐题配对、题型分数、10,000 次文档 bootstrap 与本地失败复核页均保存。表格/列表 23 题的 ANLS 风格分数在最优整体配置中仍为 0.6008，作为后续开发方向，不能将其说成已解决。

完整说明：[本机验证报告](reports/local-validation-v0.2.0.md)。源码和每轮快照、配置、输入哈希、环境、日志、权重及预测保留。原始数据、合成 fixture、模型权重和本机配置不上传 GitHub，合成样例从保存的 Python 源码重建。

## 代码入口

- `src/modeling.py`：官方模型加载、微型结构配置。
- `src/lora.py`：直接使用 PyTorch 实现 LoRA；语言模型 q/v 线性层的低秩增量，默认冻结视觉与 projector。
- `src/batching.py`：官方生成前缀＋答案＋EOS，仅监督答案 token，拒绝不匹配或静默截断。只生成答案位置需要的 vocabulary logits，减少无效内存。
- `scripts/infer.py`：零样本/adapter 推理，整页或固定裁剪，记录耗时和 token 数。
- `scripts/train_lora.py`：MPS/CUDA LoRA 训练、梯度累积、验证 NLL 选模型。CUDA 路径尚未实际运行。
- `scripts/prepare_docvqa.py`：真实开发数据下载、源文档去重、版本观察及图片哈希。
- `scripts/evaluate.py` / `src/evaluation.py`：ANLS 风格分数、规范化 Exact Match、题型分析及严格逐题对照；官方 evaluator 一致性尚未验证。
- `scripts/prepare_independent_eval.py` / `scripts/run_independent_eval.py`：锁定新文档与四组对照；`src/paired_statistics.py` 使用 PyTorch 做配对文档 bootstrap。
- `scripts/demo.py`：无需新增依赖的本地文档问答页面；每次请求保存源码、图片、配置和结果。`scripts/check_demo.py` 检查真实 HTTP 推理与基线一致性。

这个 LoRA checkpoint 是本项目的 `.pt` 格式，不是 PEFT 的 adapter 格式。

运行见 [VS Code 启动说明](VSCODE_START.md)，原理见 [模型与代码导读](MODEL_WALKTHROUGH.md)，后续研究设计见 [项目方案](PROJECT_PLAN.md)。
