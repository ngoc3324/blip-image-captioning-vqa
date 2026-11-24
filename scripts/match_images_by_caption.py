"""Match restored annotations to source images using BLIP caption retrieval."""
from __future__ import annotations
import argparse, json, re, sys
from difflib import SequenceMatcher
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STOP = {"a", "an", "the", "of", "with", "and", "in", "on", "is", "photograph", "image", "shown"}
def words(text): return {x for x in re.findall(r"[a-z]+", text.lower()) if x not in STOP}
def score(reference, generated):
    a, b = words(reference), words(generated)
    overlap = len(a & b) / max(1, len(a | b))
    return overlap * .7 + SequenceMatcher(None, reference.lower(), generated.lower()).ratio() * .3

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",default="dataset/source/extracted/images");p.add_argument("--manifest",default="dataset/annotations_verified.json");p.add_argument("--captions",default="dataset/source_captions.json");p.add_argument("--output",default="dataset/image_match_candidates.json");p.add_argument("--batch-size",type=int,default=8);a=p.parse_args()
    from PIL import Image
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor
    processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu");model.eval()
    paths=sorted([x for x in Path(a.source).rglob("*") if x.suffix.lower() in {".jpg",".jpeg",".png"}])
    generated=[]
    for start in range(0,len(paths),a.batch_size):
        batch_paths=paths[start:start+a.batch_size];images=[Image.open(x).convert("RGB") for x in batch_paths]
        inputs=processor(images=images,return_tensors="pt",padding=True)
        with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=30,num_beams=1)
        for path, tokens in zip(batch_paths,out): generated.append({"path":str(path),"caption":processor.decode(tokens,skip_special_tokens=True).strip()})
        print(f"captioned {min(start+len(batch_paths),len(paths))}/{len(paths)}",flush=True)
    refs=json.loads(Path(a.manifest).read_text(encoding="utf-8-sig"))["items"]
    matches=[]
    for ref in refs:
        ranked=sorted(({**candidate,"score":score(ref["caption"],candidate["caption"])} for candidate in generated),key=lambda x:x["score"],reverse=True)
        matches.append({"image_id":ref["image_id"],"source_index":ref["source_index"],"reference_caption":ref["caption"],"candidates":ranked[:10]})
    Path(a.captions).write_text(json.dumps(generated,indent=2),encoding="utf-8")
    Path(a.output).write_text(json.dumps(matches,indent=2),encoding="utf-8")
    print(f"Wrote {a.output}")
if __name__ == "__main__": main()
