import json
import numpy as np
import pandas as pd 
from pathlib import Path
from src.rag import get_embed_model,retrieval
from src.llm import answer
from src.config import TOP_K,TEST_PATH

GROUND_TRUTH_PATH=TEST_PATH/ "Ground_truth.json"
RESULTS_PATH = TEST_PATH / "results.csv"


#### loader func

def test_set_loader(path: Path=GROUND_TRUTH_PATH) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(f"There was not dataset found in {path} ")

    with open(path, "r", encoding="utf-8") as f:
        data=json.load(f)

    if not isinstance(data,list):
        raise ValueError("The ground truth dataset should be a list of dicts.")

    required_keys = {"question", "ground_truth"}
    for i, item in enumerate(data):
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(
                f"The item #{i} (id={item.get('id', '?')}) is missing: {missing}"
            )
    return data

#### cosine similarity func

def similarity_func(llm_answer: str, truth: str) -> float:

    embed_model=get_embed_model()
    embeddings=embed_model.encode([llm_answer, truth], normalize_embeddings=True)
    v1, v2=embeddings[0], embeddings[1]
    score=np.dot(v1,v2)
    return float(score)

def run_eval(output_path=None, verbose=True):
    test_set=test_set_loader()
    if output_path is None:
        output_path=RESULTS_PATH

    results=[]

    for i,item in enumerate(test_set,1):
        question=item["question"]
        truth=item["ground_truth"]
        chunks,_,metas=retrieval(question,TOP_K)
        llm_answer=answer(question,chunks)
        cos_sim=similarity_func(llm_answer,truth)
        entry={
        "id": item["id"],
        "question": question,
        "llm_answer": llm_answer,
        "ground_truth": truth,
        "similarity": cos_sim,
        "difficulty": item.get("difficulty", "unknown"),
        "source_pdf": item.get("source_pdf", "unknown"),
        }
        results.append(entry)
        pd.DataFrame(results).to_csv(output_path, index=False)
        if verbose:
            print(f"[{i}/{len(test_set)}] id={item['id']} | difficulty={item.get('difficulty')}")
            print(f"  Q: {question[:80]}")
            print(f"  A: {llm_answer[:80]}")
            print(f"  GT: {truth[:80]}")
            print(f"  similarity={cos_sim:.4f}")
            print()

    similarities=[r["similarity"] for r in results]
    mean_sim,median_sim=np.mean(similarities),np.median(similarities)
    worst=min(results, key=lambda r: r["similarity"])
    best=max(results, key=lambda r: r["similarity"])
    print(f"Evaluation:\n"
          f"Evaluated {len(results)} questions\n"
          f"Mean similarity: {mean_sim:.4f}\n"
          f"Median similarity: {median_sim:.4f}\n"
          f"Worst: id={worst['id']} sim={worst['similarity']:.4f}\n"
          f"Best:  id={best['id']}  sim={best['similarity']:.4f}\n"
          )






#### Debugging
if __name__ == "__main__":
    run_eval(verbose=True)
