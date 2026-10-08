"""DocVQA answer metrics: ANLS threshold 0.5, best over valid reference answers."""
def normalize(text):return ' '.join(str(text).lower().strip().split())
def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return previous[-1]
def score_answer(prediction,answers):
    if not answers:raise ValueError('Reference answers required')
    pred=normalize(prediction);scores=[]
    for answer in answers:
        answer=normalize(answer);distance=edit_distance(pred,answer)/max(len(pred),len(answer),1)
        scores.append(1-distance if distance<.5 else 0.)
    return {'anls':max(scores),'exact_match':float(any(pred==normalize(a) for a in answers))}
