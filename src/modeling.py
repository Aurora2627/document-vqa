"""Real checkpoint loader and random tiny LLaVA for architecture-only checks."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText,AutoProcessor,CLIPVisionConfig,LlamaConfig,LlavaConfig,LlavaForConditionalGeneration

def load_pretrained(model_id,revision,device,dtype=torch.bfloat16):
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision.lower()):raise ValueError('Use an immutable 40-character model revision SHA')
    root=Path(__file__).resolve().parents[1];report=root/'reports/local-model.json';source=model_id;kwargs={'revision':revision}
    if report.exists():
        info=json.loads(report.read_text())
        if info['model_id']==model_id and info['revision']==revision:
            source=str(root/info['local_directory']);kwargs={'local_files_only':True}
    processor=AutoProcessor.from_pretrained(source,**kwargs)
    model=AutoModelForImageTextToText.from_pretrained(source,torch_dtype=dtype,attn_implementation='sdpa',**kwargs).to(device)
    processor.tokenizer.padding_side='left'
    if model.config.model_type in ['llava','llava_next']:
        processor.patch_size=model.config.vision_config.patch_size
        processor.vision_feature_select_strategy=model.config.vision_feature_select_strategy
        processor.num_additional_image_tokens=1
    # OneVision processor already defines SigLIP patch counts and its full-token strategy.
    return model,processor

def tiny_llava():
    vision=CLIPVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,image_size=28,patch_size=14,projection_dim=32)
    text=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,pad_token_id=0,bos_token_id=1,eos_token_id=2)
    config=LlavaConfig(vision_config=vision.to_dict(),text_config=text.to_dict(),image_token_index=63,image_seq_length=4,vision_feature_layer=-1)
    return LlavaForConditionalGeneration(config)

def tiny_batch(device='cpu'):
    ids=torch.tensor([[1,63,63,63,63,10,11,12,13,2]],device=device)
    labels=ids.clone();labels[:,:8]=-100
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'pixel_values':torch.randn(1,3,28,28,device=device),'labels':labels}
