from src.rag import retrieval
from src.llm import answer
from src.config import TOP_K
import sys 

def main():
    print("Welcome to my custom rag system for academic reports analysis.\n" 
          "After setting the docker image, the programm deployments takes place as a chat.\n" 
          "If you want to terminate the programm type 'exit', if you want to terminate the programm\n" 
          "and at the same time evaluate the performance of the programm type 'exit_eval'"
          )
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q=="exit":
            sys.exit(0)
        if q=="exit_eval":
            break
        if not q:
            continue

        docs, dists, metas = retrieval(q, k=TOP_K)
        print(f"[debug] retrieved {len(docs)} chunks")
        response = answer(q, docs)
        print(response)

    print("evaluation placeholder")


if __name__=="__main__":
    main()

