"""Question-independent document views and matched three-image prompts."""
def prepare_views(image,mode,max_edge=0):
    w,h=image.size
    if mode=='whole':views=[image.copy()]
    elif mode=='top':views=[image.crop((0,0,w,h//2))]
    elif mode=='bottom':views=[image.crop((0,h//2,w,h))]
    elif mode=='repeat3':views=[image.copy() for _ in range(3)]
    elif mode=='whole-split':
        if h<2:raise ValueError('Document too short to split')
        views=[image.copy(),image.crop((0,0,w,h//2)),image.crop((0,h//2,w,h))]
    else:raise ValueError('Unknown view mode')
    if max_edge>0:
        for view in views:view.thumbnail((max_edge,max_edge))
    return views

def multiview_messages(question,count):
    if count!=3:raise ValueError('This experiment uses exactly three images')
    return [{'role':'user','content':[{'type':'image'} for _ in range(count)]+[
        {'type':'text','text':'The images are views of the same document. Read the relevant text across them. '+question+' Answer with a short answer only.'}]}]
