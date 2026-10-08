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
