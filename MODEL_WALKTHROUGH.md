# LLaVA 模型与代码导读

项目先把模型、实验和结果做完整，后续结合代码学习；不把当前学习进度作为实施前提。

## 结构复现与预训练模型的区别

src/modeling.py 的 tiny_llava 创建真实 Transformers LlavaForConditionalGeneration，但参数随机初始化、维度极小。它用于检查图像 token 数、视觉特征注入、语言损失与梯度，不能代替预训练问答实验。

本机正式加载的 OneVision checkpoint 属于 LLaVA 系列：SigLIP SO400M 视觉编码器、MLP 投影与 Qwen2 语言基座。它与第三个 Qwen3-VL 项目是不同模型。经典 LLaVA-1.5 常用 CLIP 与 Llama/Vicuna；LLaVA-NeXT 的所选 7B checkpoint 使用 CLIP 与 Mistral。学习时需要分清共有结构与各版本的视觉 token/分辨率处理差异。

## 可以在代码里看到的模型机制

1. AutoProcessor 使用官方 chat template，预处理图像，并展开图像占位 token。infer.py 记录 input_tokens；它包含问题、模板与视觉占位。
2. 视觉编码器产生 patch 特征，projector 把视觉维度映射到语言模型隐藏维度。图像 token 和特征数量不一致时模型会拒绝运行，test_real_llava_image_token_contract 验证这一点。
3. 语言模型根据图像表示和问题逐 token 生成答案。当前采用 greedy decoding，固定生成上限；它不能保证答案正确。
4. batching.py 将“官方生成前缀＋答案＋EOS”编码，确认前缀 token 完全一致；问题、图像和 padding 的 labels=-100。OneVision 模板的完整 assistant 消息与 generation prefix 空格不一致，使用同一生成前缀避免监督错位。
5. answer_loss 只计算答案后缀所需的 logits，但使用正确的 next-token shift。自动检查验证它与完整 causal loss 一致。这项内存调整不改变训练目标。
6. LoRALinear 实现 W(x)+alpha/rank·B(A(x))，B 初始化为零，使初始输出等同于冻结基座。只在语言 q_proj/v_proj 注入，a/b 使用 Float32；视觉编码器冻结。projector 联合训练仅作为可选 bf16 CUDA 模式，本机结构测试另以 Float32 验证 projector 梯度。
7. train_lora.py 实现 AdamW、梯度累积、梯度裁剪、验证答案 token NLL 选模型。当前选择标准不是 ANLS；最终回答质量需要独立生成评测。
8. 保存低秩参数而不重复保存预训练基座；推理时核对基座 ID/revision，再重建 adapter 并加载。

## 预定对照

相同正式数据划分和生成协议下比较：零样本与 LoRA；整页与固定裁剪/后续问题驱动裁剪；不同分辨率；是否有 OCR 辅助；可选 projector 适配。裁剪不能使用测试答案或真值区域作为输入，否则是 oracle 实验。

官方资料：https://github.com/haotian-liu/LLaVA 、https://github.com/LLaVA-VL/LLaVA-NeXT 、https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf 。
# 真实数据微调的阅读顺序（v0.4.0）

先读 `configs/real-sft-v0.4.0.json`，明确这轮实验控制了什么：固定模型版本、512 长边上限、greedy 生成，语言模型 q/v LoRA rank=8，视觉编码器及 projector 冻结。再读 `scripts/prepare_sft_data.py` 的源文档分组，确认同一源文档的不同页也不能跨训练、验证和评测。

接着跟踪 `scripts/train_lora.py`：`build_example` 将图片和问题转换成官方格式；labels 只保留答案与 EOS；`answer_loss` 计算答案的下一 token 预测损失；四个样本累积一次优化器更新，最后不足四个样本也按实际数量平均。这里优化的是每个样本答案 token 的平均损失，再对样本平均；验证 NLL 则按答案 token 数加权，二者口径需分清。

LoRA 的基础线性层保持冻结，更新低秩 A/B。B 初始为零，所以插入 adapter 时模型输出与原模型等价。真实模型本轮有 540,672 个可训练参数：语言模型有 24 层、隐藏维度 896、14 个 attention heads、2 个 key/value heads，所以 q_proj 输出 896 维，v_proj 输出 128 维；rank=8 的 q/v LoRA 参数总数是 `24 × 8 × [(896+896)+(896+128)]`。这也体现了 grouped-query attention 的 q 与 v 输出维度不同。每轮 adapter 单独保存；`adapter.pt` 是训练轮次中验证 NLL 最低的版本。即使某轮训练 loss 下降，也不能直接宣称生成准确率提高，要用独立生成和逐题配对检查。

512 输入的首次训练触发 MPS 内存上限。冻结基础权重并不意味着中间计算都无需保存：LoRA 的梯度仍依赖长序列中的激活。`--gradient-checkpointing` 通过重新计算中间激活减少保存量，代价是更多计算。采用 `use_reentrant=False`，让冻结输入嵌入的 LoRA 训练仍能计算梯度；测试会比较开启前后的损失和梯度。没有通过取消 MPS 内存上限来运行。

最后读 `scripts/run_sft_experiment.py`：原模型生成、LoRA 训练、重新加载 adapter 生成、评测在独立子进程中依次执行。验证集负责选训练轮次，保留评测负责测本轮泛化。这里的 test 清单来自带答案的官方 validation 镜像，不能称官方盲测；看过结果后，后续调参应视其为开发数据。
