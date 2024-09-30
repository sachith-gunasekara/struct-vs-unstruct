import os
import time

from datasets import load_dataset, Dataset, concatenate_datasets
from pyprojroot import here
from tqdm import tqdm

from struct_vs_unstruct.self_discover import self_discover

def bbh():
    pass

def t4d(instance):
    task_description = f"""Observation:
{instance["story"]}
Note that the characters plan to use it seperately, and not together.

Question (Select only one choice):
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
        batch_size = 5
        checkpoint_dir = here("struct_vs_unstruct/data/non_self_synthesis")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path = here("struct_vs_unstruct/logs/non_self_synthesis")
        os.makedirs(log_path, exist_ok=True)

        print("Running evaluations on T4D dataset in bursts of ", batch_size)

        # Iterate over the dataset in bursts of batch_size
        for start_idx in range(0, len(dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(dataset))
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{start_idx}_{end_idx}")

            # Skip the batch if it has already been processed
            if os.path.exists(checkpoint_path):
                print(f"Skipping already processed batch {start_idx}-{end_idx}")
                continue

            # Select the batch and run the evaluation
            batch = dataset.select(range(start_idx, end_idx))
            new_ds = batch.map(t4d, num_proc=10)

            # Save the processed batch to disk as a checkpoint
            new_ds.save_to_disk(checkpoint_path)
            print(f"Saved batch {start_idx}-{end_idx} as checkpoint.")

            # Wait for wait_time minutes before processing the next batch
            wait_time = 0
            print(f"Waiting for {wait_time} minutes to avoid NVIDIA blocking...")
            time.sleep(wait_time * 60)

        print("All batches processed. Loading checkpoints...")

        # Load all checkpoints and concatenate them
        all_datasets = []
        for file in os.listdir(checkpoint_dir):
            if file.startswith("checkpoint"):
                ds = Dataset.load_from_disk(os.path.join(checkpoint_dir, file))
                all_datasets.append(ds)

        full_dataset = concatenate_datasets(all_datasets)
        print(f"Combined dataset contains {len(full_dataset)} instances.")

        # Save the final dataset
        full_dataset.save_to_disk(os.path.join(checkpoint_dir, "t4d_eval"))

        print("Calculating accuracy")
        correct_preds = 0
        for instance in tqdm(full_dataset, desc="instance"):
            if (instance["answer"] in instance["answer_pred"]) and instance["answer"] == instance["answer_pred"][3:]:
                correct_preds += 1
            else:
                with open(os.path.join(log_path, "evals/t4d_different.txt"), "a") as f:
                    f.write(f"{instance['answer']}, {instance['answer_pred']}\n")

        accuracy = correct_preds / len(full_dataset)
        print("Accuracy: ", accuracy)

        # Log accuracy
        with open(os.path.join(log_path, "evals/t4d_different.txt"), "a") as f:
            f.write(f"Accuracy: {accuracy}\n")        


if __name__ == "__main__":
    evaluate("t4d")