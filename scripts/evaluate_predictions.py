"""Evaluate benchmark JSONL predictions."""
import argparse,json,statistics,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_annotations
from src.metrics import bert_score,caption_metrics,vqa_metrics
def validate(row,record):
    if not isinstance(row.get("caption"),str) or not row["caption"].strip(): raise ValueError("caption invalid")
    qa=row.get("qa")
    if not isinstance(qa,list) or len(qa)!=len(record.qa_pairs): raise ValueError("QA count mismatch")
    for p,e in zip(qa,record.qa_pairs):
        if p.get("type")!=e.type or p.get("question")!=e.question or not isinstance(p.get("answer"),str): raise ValueError("QA order/type/question mismatch")
        if not isinstance(p.get("latency_ms"),(int,float)): raise ValueError("QA latency invalid")
    if not isinstance(row.get("caption_latency_ms"),(int,float)): raise ValueError("caption latency invalid")
def mean(rows):
    keys=set().union(*(x.keys() for x in rows)) if rows else set(); return {k:statistics.mean([x[k] for x in rows if x.get(k) is not None]) if any(x.get(k) is not None for x in rows) else None for k in keys}
def main():
    p=argparse.ArgumentParser();p.add_argument("predictions");p.add_argument("--manifest",default="dataset/annotations_300.json");p.add_argument("--output",default="artifacts/metrics.json");a=p.parse_args();records={x.image_id:x for x in load_annotations(a.manifest)}; groups=defaultdict(list)
    for n,line in enumerate(Path(a.predictions).open(encoding="utf-8"),1):
        row=json.loads(line); r=records.get(row.get("image_id"));
        if r is None: raise ValueError(f"line {n}: unknown image")
        validate(row,r);groups[row["preset"]].append(row)
    result={"manifest":a.manifest,"presets":{}}
    for preset,rows in groups.items():
        cs=[]; refs=[]; preds=[]; vs=[]; by_type=defaultdict(list); by_label=defaultdict(list); lat=[]
        for row in rows:
            r=records[row["image_id"]];preds.append(row["caption"]);refs.append(r.caption);cs.append(caption_metrics(row["caption"],r.caption));lat.append(row["caption_latency_ms"])
            for got,expected in zip(row["qa"],r.qa_pairs):
                score=vqa_metrics(got["answer"],expected.answer);vs.append(score);by_type[expected.type].append(score);by_label[r.source_label].append(score);lat.append(got["latency_ms"])
        result["presets"][preset]={"caption":{**mean(cs),"bertscore_f1":bert_score(preds,refs)},"vqa":mean(vs),"vqa_by_question_type":{k:mean(v) for k,v in by_type.items()},"vqa_by_source_label":{k:mean(v) for k,v in by_label.items()},"latency_ms":{"mean":statistics.mean(lat),"median":statistics.median(lat),"std":statistics.stdev(lat) if len(lat)>1 else 0},"records":len(rows),"qa_pairs":len(vs)}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding="utf-8");print(f"Wrote {out}")
if __name__ == "__main__": main()
