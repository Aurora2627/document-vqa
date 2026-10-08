"""Supervise assistant answer only; reject silent prompt/image token truncation."""
import torch
from PIL import Image

def prompt_messages(question):
    return [{'role':'user','content':[{'type':'image'},{'type':'text','text':question+' Answer with a short answer only.'}]}]

def build_example(processor,row,max_length=4096,max_image_edge=0):
    messages=prompt_messages(row['question'])
    prefix=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    # Train against the same official generation prefix used at inference.
    # Some shipped templates render assistant spacing differently in full conversations.
    eos=processor.tokenizer.eos_token
    if not eos:raise ValueError('Tokenizer EOS token required')
    full=prefix+row['answers'][0]+eos
    with Image.open(row['image_path']) as image:
        image=image.convert('RGB')
        if max_image_edge>0:image.thumbnail((max_image_edge,max_image_edge))
        prompt=processor(text=prefix,images=image,return_tensors='pt')
        batch=processor(text=full,images=image,return_tensors='pt')
    prefix_ids=prompt['input_ids'];ids=batch['input_ids'];length=prefix_ids.shape[1]
    if ids.shape[1]>max_length:raise ValueError('Example exceeds max_length; do not silently truncate image/answer tokens')
    if length>=ids.shape[1] or not torch.equal(prefix_ids,ids[:,:length]):raise ValueError('Chat-template prefix mismatch or missing assistant tokens')
    labels=ids.clone();labels[:,:length]=-100;labels[batch['attention_mask']==0]=-100
    labels[ids==processor.tokenizer.convert_tokens_to_ids(processor.image_token)]=-100
    if int((labels[:,1:]!=-100).sum())==0:raise ValueError('No shifted answer supervision tokens')
    batch['labels']=labels
    return batch

def move_batch(batch,device,dtype):
    return {k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in batch.items()}

def answer_loss(model,batch):
    # Avoid allocating vocabulary logits for thousands of masked prompt/image tokens.
    labels=batch['labels'];positions=(labels[0]!=-100).nonzero().flatten()
    if len(labels)!=1 or len(positions)==0 or int(positions[0])<1:raise ValueError('Single example with an assistant suffix required')
    start=int(positions[0]);inputs={k:v for k,v in batch.items() if k!='labels'}
    output=model(**inputs,use_cache=False,logits_to_keep=labels.shape[1]-start+1)
    return torch.nn.functional.cross_entropy(output.logits[:,:-1].float().reshape(-1,output.logits.shape[-1]),labels[:,start:].reshape(-1),ignore_index=-100)
