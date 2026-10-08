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
