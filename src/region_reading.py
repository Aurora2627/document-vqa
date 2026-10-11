"""Question-conditioned coarse-to-fine reading; candidate regions are NOT verified evidence."""
import time
import torch
from batching import move_batch,prompt_messages
from views import multiview_messages

def region_boxes(size,fraction=0.6):
    w,h=size
    if w<3 or h<3 or not 0<fraction<=1:raise ValueError('Invalid document/crop size')
    cw,ch=max(1,round(w*fraction)),max(1,round(h*fraction));boxes=[]
    for row in range(3):
        for col in range(3):
            left=max(0,min(w-cw,round(w*(col+.5)/3-cw/2)))
            top=max(0,min(h-ch,round(h*(row+.5)/3-ch/2)))
            boxes.append((left,top,left+cw,top+ch))
    return boxes

def locator_messages(question):
    text=('Find the region most relevant to answering this document question: '+question+
        '\nImagine the page divided into a 3 by 3 grid. Region numbers in reading order are '
        '1 top-left, 2 top-center, 3 top-right, 4 middle-left, 5 center, 6 middle-right, '
        '7 bottom-left, 8 bottom-center, 9 bottom-right. Locate the supporting text or figure. '
        'Reply with only one region number from 1 to 9, not the answer to the question.')
    return [{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]

def rank_regions(logits):
    scores=torch.as_tensor(logits,dtype=torch.float32).detach().cpu()
    if scores.shape!=(9,) or not torch.isfinite(scores).all():raise ValueError('Nine finite region logits required')
    order=torch.argsort(scores,descending=True,stable=True)
    return (order+1).tolist(),torch.softmax(scores,dim=0).tolist()

def synchronize(device):
    if device=='mps':torch.mps.synchronize()
    if device=='cuda':torch.cuda.synchronize()

def locate_regions(model,processor,image,question,device,dtype,max_edge=512):
    # No annotation/answer is accepted by this API. Ranking uses only page + question.
    synchronize(device);start=time.perf_counter();preview=image.copy();preview.thumbnail((max_edge,max_edge))
    prefix=processor.apply_chat_template(locator_messages(question),tokenize=False,add_generation_prompt=True)
    batch=move_batch(processor(text=prefix,images=preview,return_tensors='pt'),device,dtype)
    ids=[processor.tokenizer.encode(str(i),add_special_tokens=False) for i in range(1,10)]
    if any(len(tokens)!=1 for tokens in ids) or len({t[0] for t in ids})!=9:raise ValueError('Locator requires distinct single-token region digits')
    with torch.inference_mode():
        output=model(**batch,use_cache=False,logits_to_keep=1)
        logits=output.logits[0,-1,[t[0] for t in ids]]
    ranking,probabilities=rank_regions(logits);synchronize(device)
    return {'ranking':ranking,'restricted_digit_probabilities':probabilities,'input_tokens':batch['input_ids'].shape[1],
        'seconds':time.perf_counter()-start,'confidence_note':'probabilities normalized over nine forced digits, not calibrated grounding confidence'}

def read_views(model,processor,views,question,device,dtype):
    synchronize(device);total_start=time.perf_counter()
    messages=(multiview_messages(question,len(views)) if len(views)>1 else prompt_messages(question)) if views else [
        {'role':'user','content':[{'type':'text','text':question+' Answer with a short answer only.'}]}]
    prefix=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=prefix,images=[views] if len(views)>1 else views[0],return_tensors='pt') if views else processor(text=prefix,return_tensors='pt')
    batch=move_batch(inputs,device,dtype)
    synchronize(device);start=time.perf_counter()
    with torch.inference_mode():tokens=model.generate(**batch,max_new_tokens=64,do_sample=False)
    synchronize(device);seconds=time.perf_counter()-start
    answer=processor.tokenizer.decode(tokens[0,batch['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    return {'prediction':answer,'seconds':seconds,'reader_wall_seconds':time.perf_counter()-total_start,
        'input_tokens':batch['input_ids'].shape[1],'image_count':len(views),'view_sizes':[list(v.size) for v in views]}

def prepare_region_views(image,ids,max_edge=768,include_whole=True):
    if len(ids)!=len(set(ids)) or any(type(i)is not int or i<1 or i>9 for i in ids):raise ValueError('Unique region IDs 1..9 required')
    boxes=region_boxes(image.size);views=([image.copy()] if include_whole else [])+[image.crop(boxes[i-1]) for i in ids]
    for view in views:view.thumbnail((max_edge,max_edge))
    return views,[list(boxes[i-1]) for i in ids]
