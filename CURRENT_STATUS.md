# 当前项目状态

当前版本 v0.7.3。技术栈为 Python/PyTorch/Transformers，模型默认 NVIDIA CUDA；VS Code 启动步骤见 [VSCODE_START.md](VSCODE_START.md)。

已完成 OneVision 答案监督 LoRA、文档隔离评测、分辨率/区域/OCR 消融及文档问答演示。效果与限制见 [README](README.md)。

Windows RTX 4060 Laptop 8GB 上已验证预训练 LoRA 更新、fixture 训练、adapter 推理、新进程重载、OCR 与 16 项测试。该检查确认流程兼容性，未完整重跑早期真实数据基准，不产生新的算法效果结论。历史实验设备按原报告记录。

[CUDA 检查证据](reports/v0.7.1/compatibility.json) · [扩展研究](reports/deepening-v0.7.0.md)
