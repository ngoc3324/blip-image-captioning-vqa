"""Run repeated BLIP decoding comparisons."""
import argparse, hashlib, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.captioner import BlipCaptioner
from src.data_loader import load_annotations
from src.model_runtime import GenerationConfig
from src.vqa import BlipVQA

def presets(): return [("greedy",GenerationConfig()),("beam-3",GenerationConfig(num_beams=3)),("beam-5",GenerationConfig(num_beams=5)),("sample-07",GenerationConfig(do_sample=True,temperature=.7)),("sample-10",GenerationConfig(do_sample=True,temperature=1.0))]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",default="dataset/annotations_300.json"); p.add_argument("--output",default="artifacts/predictions.jsonl"); p.add_argument("--limit",type=int,default=0); p.add_argument("--repeats",type=int,default=3); p.add_argument("--device",default="auto"); p.add_argument("--no-warmup",action="store_true"); a=p.parse_args()
    records=load_annotations(a.manifest); records=records[:a.limit] if a.limit else records; cap=BlipCaptioner(requested_device=a.device); vqa=BlipVQA(requested_device=a.device); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    import torch, transformers
    meta={"caption_model":cap.model_name,"vqa_model":vqa.model_name,"device":cap.device,"seed":42,"manifest_sha256":hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),"python_version":platform.python_version(),"torch_version":torch.__version__,"transformers_version":transformers.__version__}
    with out.open("w",encoding="utf-8") as f:
        for name,cfg in presets():
            if not a.no_warmup: cap.caption(records[0].image_path,cfg); vqa.answer(records[0].image_path,records[0].qa_pairs[0].question,cfg)
            for repeat in range(a.repeats):
                for r in records:
                    t=time.perf_counter(); caption=cap.caption(r.image_path,cfg); c_ms=(time.perf_counter()-t)*1000; qa=[]
                    for q in r.qa_pairs:
                        t=time.perf_counter(); ans=vqa.answer(r.image_path,q.question,cfg); qa.append({"type":q.type,"question":q.question,"answer":ans,"latency_ms":(time.perf_counter()-t)*1000})
                    f.write(json.dumps({**meta,"timestamp":datetime.now(timezone.utc).isoformat(),"image_id":r.image_id,"preset":name,"repeat":repeat,"config":cfg.__dict__,"caption":caption,"caption_latency_ms":c_ms,"qa":qa},ensure_ascii=False)+"\n")
    print(f"Wrote {out}")
if __name__ == "__main__": main()
