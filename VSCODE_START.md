# VS Code 启动与调试

打开 mm02-document-vqa 文件夹，Python 解释器选择 /usr/bin/python3。当前本机依赖仍在共享工作区 .system-python-packages，项目 .vscode 已配置 PYTHONPATH。该本机配置不上传 GitHub；其他设备根据实际依赖目录设置。后续云端使用 VS Code Remote SSH 与独立 Python 3.11 环境。

## 本机已验证流程

所有命令在本项目 VS Code 集成终端运行。每次更换 runs 目录名，不能覆盖。

```sh
/usr/bin/python3 scripts/prepare_sample.py
/usr/bin/python3 scripts/run_checks.py
/usr/bin/python3 scripts/download_local_model.py
/usr/bin/python3 scripts/infer.py --manifest data/sample/fixtures.jsonl --model llava-hf/llava-onevision-qwen2-0.5b-ov-hf --revision 74dd0bf867a4cda7950c17663794267c60cf4b40 --device mps --max-image-edge 384 --output runs/my-zero-shot
/usr/bin/python3 scripts/train_lora.py --device mps --train data/sample/train.jsonl --val data/sample/val.jsonl --model llava-hf/llava-onevision-qwen2-0.5b-ov-hf --revision 74dd0bf867a4cda7950c17663794267c60cf4b40 --max-image-edge 384 --epochs 2 --accumulation 1 --output runs/my-lora
/usr/bin/python3 scripts/infer.py --manifest data/sample/fixtures.jsonl --model llava-hf/llava-onevision-qwen2-0.5b-ov-hf --revision 74dd0bf867a4cda7950c17663794267c60cf4b40 --adapter runs/my-lora/adapter.pt --device mps --max-image-edge 384 --output runs/my-lora-predictions
/usr/bin/python3 scripts/evaluate.py --predictions runs/my-zero-shot/predictions.jsonl --output runs/my-zero-shot/answer-metrics.json
```

下载器获取当时官方 revision 并记录 SHA256；若未来官方 revision 不同，请使用 reports/local-model.json 中的真实 revision。配置网络代理使用自己的终端环境变量，不把本机地址写入公共源码。

上面是合成 fixture 兼容性测试，不能用于简历效果数字。正式数据按 question_id、document_id、image（相对清单目录）、question、answers（多答案列表）、split（train/val/test）组织 JSONL。训练入口需要分开的 train/val 清单，同一文档不能跨划分。

## 调试入口

运行和调试中已有：01 代码与测试、02 微型结构 MPS 验证、03 生成测试文档、04 本机预训练零样本问答、05 本机 LoRA 流程。可在 lora.py 的 forward、batching.py 的 labels 和 train_lora.py 的 loss 下断点。

云端 7B 实验需要固定模型 SHA、安装 requirements-cloud.txt 并使用匹配服务器的 PyTorch CUDA 轮子；当前尚未验证该路径。一次小 batch 测量显存后再定训练规模。

## 真实开发数据实验（v0.3.0）

本地已有 runs/docvqa-dev-v030/dev.jsonl；调试配置 06 可直接重跑 512 基线，每次填写新的实验目录名。建议在 infer.py 的 processor 输入、model.generate 和答案解码处下断点。公开仓库不包含数据，其他设备先下载。下载脚本使用终端中配置的 HTTPS_PROXY，不硬编码代理。

```sh
/usr/bin/python3 scripts/prepare_docvqa.py --count 20 --output runs/my-docvqa-data
/usr/bin/python3 scripts/infer.py --manifest runs/my-docvqa-data/dev.jsonl --model llava-hf/llava-onevision-qwen2-0.5b-ov-hf --revision 74dd0bf867a4cda7950c17663794267c60cf4b40 --device mps --max-image-edge 384 --output runs/my-dev-384
/usr/bin/python3 scripts/infer.py --manifest runs/my-docvqa-data/dev.jsonl --model llava-hf/llava-onevision-qwen2-0.5b-ov-hf --revision 74dd0bf867a4cda7950c17663794267c60cf4b40 --device mps --max-image-edge 512 --output runs/my-dev-512
/usr/bin/python3 scripts/evaluate.py --predictions runs/my-dev-384/predictions.jsonl --output runs/my-dev-384/answer-metrics.json
/usr/bin/python3 scripts/evaluate.py --predictions runs/my-dev-512/predictions.jsonl --compare-to runs/my-dev-384/predictions.jsonl --output runs/my-dev-512/answer-metrics.json
```

开发集来自 viewer 顺序页面，未锁定 viewer revision，保存的图片哈希及原始响应用于本轮审计。不是完整官方基准评测。

## 真实数据 LoRA 对照（v0.4.0）

本机已有 runs/docvqa-sft-data-v040c 中的 train/val/test 清单。这里的 test 是从带答案的官方 validation 镜像划出的本轮保留评测，不能称官方盲测。上轮 20 题开发集被排除。公开仓库不含数据，下载时配置自己的终端代理。

```sh
/usr/bin/python3 scripts/prepare_sft_data.py --exclude-manifest runs/docvqa-dev-v030/dev.jsonl --output runs/my-sft-data
/usr/bin/python3 scripts/run_sft_experiment.py --data runs/my-sft-data --output runs/my-real-lora
```

参数协议保存在 configs/real-sft-v0.4.0.json。总入口依次启动检查、原模型验证/评测预测、LoRA 训练、选中 adapter 的新进程预测、严格配对评测；子进程同样使用当前系统 Python 和 PyTorch，在 VS Code 集成终端内运行。每阶段保存日志，每轮保存源码和输入快照，失败不会覆盖旧目录。

调试入口 07 可单独运行真实 LoRA 训练。在 batching.py 的 labels、answer_loss 和 train_lora.py 的 backward、梯度裁剪及验证损失处下断点。每个 epoch 的 adapter 保存在 checkpoints，adapter.pt 指向验证答案 token NLL 最低的训练轮次；该选择依据不是评测集准确率。
