# VS Code / CUDA 运行说明

在 Windows 上打开本仓库文件夹，使用隔离的 Python 环境；也可通过 VS Code Remote SSH 打开服务器上的仓库。选择 Python 解释器后，复制对应 configs/vscode-launch*.json 模板到本机 .vscode/launch.json。模板使用所选解释器，不要求固定盘符、账户名或 SSH 别名。

已验证环境：Python 3.9.18、torch 2.8.0+cu126、torchvision 0.23.0+cu126、transformers 4.57.6、RTX 4060 Laptop 8GB。重建环境参考 requirements-cuda.txt，使用新的虚拟环境。模型入口默认 CUDA；CUDA 不可用会报错。数据处理、指标和标量校准使用 CPU。

先运行 configs/vscode-launch.cuda.json 中的 CUDA preflight，再运行模型任务。下载的模型、数据和 adapter 必须与实验要求匹配；每次使用新的输出目录，保留源码、配置、输入哈希、日志、预测和权重。不要覆盖已有 runs；同一张 GPU 同时运行一个模型任务。

## 文档问答入口

configs/vscode-launch.cuda.json 提供 CUDA pretrained LoRA compatibility、CUDA pretrained adapter reload、CUDA fixture LoRA training、CUDA fixture inference 和 CUDA document QA demo。fixture 仅验证流程，不能当作真实任务评测。

默认 OneVision 0.5B；训练在支持时采用 BF16，FP16 使用梯度缩放。8GB 显存下从较低分辨率、梯度检查点及梯度累积开始，按实际峰值显存调整。

完整数据协议与历史结果见 README 和 reports；当前 CUDA 验证范围见 [CURRENT_STATUS.md](CURRENT_STATUS.md)。兼容性验证不等于完整真实数据基准复跑。演示启动后仅监听本机回环地址；远程访问使用 VS Code 端口转发。
