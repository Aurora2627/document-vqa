# 当前项目入口

当前维护与求职展示以 main 分支的 Windows / NVIDIA CUDA 版本为准。后续编写、调试、测试与实验均通过 VS Code Remote SSH 在 Windows 完成；Mac 仅作为远程连接终端。旧提交、旧标签和已完成实验保留真实历史，不代表当前默认运行方式。

- 推荐入口：main；当前版本 v0.7.2。
- 技术栈：Python、PyTorch 2.8.0+cu126、Transformers 4.57.6；Windows RTX 4060 Laptop 8GB。
- 启动：按 VSCODE_START.md 配置隔离解释器，使用 configs/vscode-launch.cuda.json 调试模板。
- 历史版本仍可从 Git 查看，用于展示迭代过程；请勿照搬旧 MPS 命令作为当前启动方式。

已在 CUDA 验证预训练 LoRA 更新、正式 fixture 训练入口、adapter 推理、新进程重载、OCR 与 16 项测试。该验证确认流程可运行，未完整重跑早期真实数据基准，也不产生新的算法效果结论。

早期真实数据研究与局限见 README 和 reports；兼容性证据见 [CUDA 检查](reports/v0.7.1/compatibility.json)。
