"""Fill the image manifest with provisional labels/captions without the 1.5GB VQA model."""
from __future__ import annotations
import argparse, json, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTIONS = [
    ("object", "What is a prominent object or subject in the image?"),
    ("scene", "What kind of scene is shown?"),
    ("visual_detail", "What other visible object or feature appears in the scene?"),
]
LABELS = ["door", "lamp", "photo frame", "wardrobe", "window"]

def label_for(root, archive, entries, index):
    parts = entries[index].filename.split("/")
    path = root / "dataset" / "source" / "extracted" / "labels" / parts[1] / f"{Path(parts[2]).stem}.txt"
    if not path.exists(): return "unknown"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return LABELS[int(lines[0].split()[0])] if lines else "unknown"

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",default="dataset/annotations_300.json");p.add_argument("--batch-size",type=int,default=4);a=p.parse_args()
    from PIL import Image
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor
    root=Path(__file__).resolve().parents[1];data=json.loads((root/a.manifest).read_text(encoding="utf-8-sig"));items=data["items"]
    processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base");model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
    with zipfile.ZipFile(root/"dataset/source/HomeObjects-3K_5class_143_per_class.zip") as archive:
        entries=[x for x in archive.infolist() if x.filename.startswith("images/") and x.filename.lower().endswith((".jpg",".jpeg",".png"))]
        for start in range(0,len(items),a.batch_size):
            batch=items[start:start+a.batch_size];images=[Image.open(root/"dataset"/x["image_path"]).convert("RGB") for x in batch]
            inputs=processor(images=images,return_tensors="pt",padding=True)
            with torch.inference_mode():outputs=model.generate(**inputs,max_new_tokens=30,num_beams=3)
            captions=[processor.decode(row,skip_special_tokens=True).strip() for row in outputs]
            for item,caption in zip(batch,captions):
                label=label_for(root,archive,entries,int(item["source_index"]));item["source_label"]=label;item["caption"]=caption;item["qa_pairs"]=[]
                answers={"object":label,"scene":caption,"visual_detail":caption}
                for qa_type,question in QUESTIONS:item["qa_pairs"].append({"type":qa_type,"question":question,"answer":answers[qa_type]})
                item["status"]="auto_generated_needs_review";item["annotation_note"]="Caption/QA draft generated without BLIP VQA; review before ground-truth use."
            print(f"annotated {min(start+len(batch),len(items))}/{len(items)}",flush=True)
    data["source"]="restored_image_folder_and_blip_caption";data["annotation_status"]="auto_generated_draft";data["annotation_notes"]="Object labels from source YOLO labels; captions from BLIP; QA answers are provisional label/caption-derived drafts.";out=root/a.manifest;tmp=out.with_suffix(".tmp.json");tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8");tmp.replace(out);print(f"Wrote {out}")
if __name__=="__main__":main()
