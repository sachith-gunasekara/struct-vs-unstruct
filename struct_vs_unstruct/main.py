from datasets import load_dataset
from pyprojroot import here
from tqdm import tqdm

from struct_vs_unstruct.self_discover import self_discover

def bbh():
    pass

def t4d(instance):
    task_description = f"""Observation:
{instance["story"]}

Question:
{instance["question"]}"""
    
    out = self_discover(task_description, modified=True)

    del out["reasoning_modules"]
    del out["task_description"]

    return out

def math():
    pass

def evaluate(benchmark: str):
    if benchmark == "t4d":
        dataset = load_dataset("sachithgunasekara/t4d")["train"]

        print("Running evaluations on T4D dataset")
        new_ds = dataset.map(t4d, num_proc=5)
        new_ds.save_to_disk(here("struct_vs_instruct/data/t4d_eval"))

        print("Calculating accuracy")
        correct_preds = 0
        for instance in tqdm(new_ds, desc="instance"):
            if (instance["answer"] in instance["answer_pred"]) and instance["answer"] == instance["answer_pred"][3:]:
                correct_preds += 1
            else:
                with open(here("struct_vs_unstruct/logs/evals/t4d_different.txt"), "a") as f:
                    f.write(instance["answer"], instance["answer_pred"])
        
        accuracy = correct_preds / len(new_ds)
        print("Accuracy: ", accuracy)

        with open(here("struct_vs_unstruct/logs/evals/t4d.txt"), "a") as f:
            f.write("Accuracy: ", accuracy)
        


if __name__ == "__main__":
    evaluate("t4d")