import os
import time

from datasets import load_dataset, get_dataset_config_names
from pyprojroot import here
from tqdm import tqdm

from struct_vs_unstruct.self_discover import self_discover
from struct_vs_unstruct.helpers.dataset import load_checkpoints
from struct_vs_unstruct.helpers.evals import calculate_accuracy
from struct_vs_unstruct.helpers.config import config
from struct_vs_unstruct.helpers.logger import logger


def bbh(instance):
    out = self_discover(instance["input"], modified=True)

    del out["reasoning_modules"]
    del out["task_description"]

    return out

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

def evaluate(benchmark: str, dataset_name: str, subset: str, instance_processor):
    
    logger.info("Running evaluations on %s. Loading subset %s from dataset %s", benchmark, subset, dataset_name)
    
    dataset = load_dataset(dataset_name, subset, split="train")

    batch_size = int(config["EVAL"]["BatchSize"])
    checkpoint_dir = here(os.path.join("struct_vs_unstruct", config["PATHS"]["CheckpointDir"], benchmark, f"{benchmark}-{subset}"))
    log_dir = here(os.path.join("struct_vs_unstruct", config["PATHS"]["LogDir"], "evals", benchmark, f"{benchmark}-{subset}"))

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger.info("Running evaluations on %s dataset in bursts of %s", benchmark, batch_size)

    # Iterate over the dataset in bursts of batch_size
    for start_idx in range(0, 5, batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{start_idx}_{end_idx}")

        # Skip the batch if it has already been processed
        if os.path.exists(checkpoint_path):
            logger.debug(f"Skipping already processed batch {start_idx}-{end_idx}")
            continue

        # Select the batch and run the evaluation
        batch = dataset.select(range(start_idx, end_idx))
        new_ds = batch.map(instance_processor, num_proc=batch_size)

        # Save the processed batch to disk as a checkpoint
        new_ds.save_to_disk(checkpoint_path)
        logger.info(f"Saved batch {start_idx}-{end_idx} as checkpoint.")

    logger.info("All batches processed. Loading checkpoints...")

    # Load all checkpoints and concatenate them
    full_dataset = load_checkpoints(checkpoint_dir, benchmark)

    logger.info(f"Combined dataset contains {len(full_dataset)} instances.")

    logger.info("Calculating accuracy")

    accuracy = calculate_accuracy(
        full_dataset, 
        benchmark=benchmark,
        y="answer",
        y_pred="answer_pred",
        log_file_path=os.path.join(log_dir, f"{benchmark}_different.txt")
    )

    logger.info("Accuracy of %s - %s: %f", dataset_name, benchmark, accuracy)

    # Log accuracy
    with open(os.path.join(log_dir, f"{benchmark}.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy}\n") 

    return accuracy


if __name__ == "__main__":
    benchmarks = ["t4d", "bbh"][1:]
    dataset_names = ["sachithgunasekara/t4d", "maveriq/bigbenchhard"][1:]
    subset_list = [[""], get_dataset_config_names(dataset_names[0])][1:]
    instance_processors = [t4d, bbh][1:]

    for benchmark, dataset_name, subsets, instance_processor in zip(benchmarks, dataset_names, subset_list, instance_processors):

        for subset in subsets:

            while True:
                try:
                    acc = evaluate(benchmark, dataset_name, subset, instance_processor)

                    if acc is not None:
                        break
                except Exception as e:
                    # Check for the specific Bearer token error
                    if "Bearer token is malformed" in str(e):
                        wait_time = 30
                        print(f"Bearer token malformed. Waiting for {wait_time} minutes to avoid NVIDIA blocking...")
                        time.sleep(wait_time * 60)
                    else:
                        # Re-raise the exception if it's not related to Bearer token
                        raise e