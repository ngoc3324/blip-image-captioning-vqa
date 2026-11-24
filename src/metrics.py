"""Caption and VQA metrics."""
import math, re
from collections import Counter

def normalize_answer(text):
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", text)).strip()
def exact_match(prediction, reference): return float(normalize_answer(prediction) == normalize_answer(reference))
def token_f1(prediction, reference):
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p or not r: return float(p == r)
    overlap = sum((Counter(p) & Counter(r)).values())
    if not overlap: return 0.0
    precision, recall = overlap/len(p), overlap/len(r)
    return 2*precision*recall/(precision+recall)
def rouge_l(prediction, reference):
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    matrix = [[0]*(len(r)+1) for _ in range(len(p)+1)]
    for i, a in enumerate(p, 1):
        for j, b in enumerate(r, 1): matrix[i][j] = matrix[i-1][j-1]+1 if a == b else max(matrix[i-1][j], matrix[i][j-1])
    return matrix[-1][-1]/len(r) if r else float(not p)
def bleu4(prediction, reference):
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p or not r: return float(p == r)
    values=[]
    for n in range(1,5):
        a=Counter(tuple(p[i:i+n]) for i in range(len(p)-n+1)); b=Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1)); total=sum(a.values()); hit=sum(min(v,b[k]) for k,v in a.items()); values.append((hit+1)/(total+1))
    return min(1, math.exp(1-len(r)/len(p)))*math.exp(sum(math.log(v) for v in values)/4)
def caption_metrics(prediction, reference):
    try:
        import sacrebleu
        bleu=sacrebleu.sentence_bleu(prediction,[reference]).score/100
    except Exception: bleu=bleu4(prediction,reference)
    try:
        from rouge_score import rouge_scorer
        rouge=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True).score(reference,prediction)["rougeL"].fmeasure
    except Exception: rouge=rouge_l(prediction,reference)
    return {"bleu4": bleu, "rouge_l": rouge, "bertscore_f1": None}
def vqa_metrics(prediction, reference): return {"exact_match": exact_match(prediction,reference), "token_f1": token_f1(prediction,reference)}
def bert_score(predictions, references):
    try:
        from bert_score import score
        _,_,f1=score(list(predictions),list(references),lang="en",verbose=False); return float(f1.mean().item())
    except Exception: return None
