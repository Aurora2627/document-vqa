# MacBook / VS Code / PyTorch 本机验证

2026-10-08，系统 Python 3.9、PyTorch 2.8、Transformers 4.57.6，全部从 VS Code 集成终端运行，使用 MPS。

| 检查 | 结果 | 解释 |
|---|---|---|
| 自动检查 | 7 项通过，无失败/跳过 | 包括真实 HF 微型结构、LoRA、loss 和监督边界 |
| 微型随机 LLaVA | 四次 AdamW 更新、精确重载、冻结视觉参数检查通过 | 不代表预训练问答能力 |
| 官方 OneVision 小模型推理 | 一条合成验证发票读取金额正确，生成约 3.40 秒，输入 1344 token | 不含模型加载/预处理；只是一条软件 fixture |
| 官方小模型 LoRA 单步 | loss 0.04254→0.03535，梯度非零，270336 可训练参数 | 一条合成训练样本，无泛化结论 |
| 新进程重载 | 损失与保存后的模型一致 | 重新加载官方基座和 adapter |
| 完整本机 SFT 入口 | 1 条合成训练＋1 条合成验证，两轮完成 | 验证 NLL 选模型，不能作为正式任务收益 |

单步测试结束时 MPS current allocated 为 1,836,235,008 字节，driver allocated 为 8,656,650,240 字节。它们是当时分配量，不是峰值显存，也不是整机总内存。推理与训练分开执行；不推断大 batch 或高分辨率也能运行。

官方权重文件 1,787,445,680 字节，与官方 LFS SHA256 一致，模型 revision 74dd0bf867a4cda7950c17663794267c60cf4b40。详细模型文件校验见 local-model.json。

第一次预训练模型训练测试因官方 full assistant 模板与生成前缀空格不一致而被严格前缀检查拦截；修正为相同生成前缀＋答案＋EOS 后，保存新的实验目录重新验证。原失败目录没有覆盖。

可复现产物在 runs/llava-architecture-mps-20261008、runs/onevision-local-inference-v020、runs/onevision-local-lora-v020b、runs/onevision-fresh-reload-v020、runs/onevision-local-sft-fixture-v020。每次启动前保存源码快照、配置、输入哈希、环境；模型更新保存训练日志与 .pt 权重。输入图片哈希记录在实际模型实验 config.json。

下一阶段：正式 DocVQA 数据准备、固定开发与保留评测集、零样本/LoRA/分辨率/裁剪对照。证据定位和 OCR 辅助尚未实现；没有企业应用效果或性能提升声明。
