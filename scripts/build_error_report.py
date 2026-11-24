import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_annotations
from src.metrics import exact_match,token_f1
from scripts.evaluate_predictions import validate
def main():
    p=argparse.ArgumentParser();p.add_argument("predictions");p.add_argument("--manifest",default="dataset/annotations_verified.json");p.add_argument("--output",default="artifacts/error-review.csv");a=p.parse_args();records={r.image_id:r for r in load_annotations(a.manifest)};out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    fields=["preset","image_id","question_type","question","expected","prediction","exact_match","token_f1","category","severity","notes"]
    with Path(a.predictions).open(encoding="utf-8") as source,outfile.open("w",newline="",encoding="utf-8") as outfile:
        writer=csv.DictWriter(outfile,fieldnames=fields);writer.writeheader()
        for line in source:
            row=json.loads(line);r=records[row["image_id"]];validate(row,r)
            for got,expected in zip(row["qa"],r.qa_pairs):
                score=token_f1(got["answer"],expected.answer)
                if score<1: writer.writerow({"preset":row["preset"],"image_id":row["image_id"],"question_type":expected.type,"question":expected.question,"expected":expected.answer,"prediction":got["answer"],"exact_match":exact_match(got["answer"],expected.answer),"token_f1":score,"category":"unlabeled","severity":"review","notes":"Classify hallucination/count/color/relation/outside-knowledge/other."})
    print(f"Wrote {out}")
if __name__ == "__main__": main()
