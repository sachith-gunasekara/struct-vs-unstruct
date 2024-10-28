import os
import time

from datasets import load_dataset, get_dataset_config_names
from pyprojroot import here
from tqdm import tqdm

from struct_vs_unstruct.self_discover import self_discover
from struct_vs_unstruct.helpers.dataset import load_checkpoints
from struct_vs_unstruct.helpers.evals import calculate_accuracy
from struct_vs_unstruct.helpers.config import read_config, save_config
from struct_vs_unstruct.helpers.logger import logger



def call_self_discover(task_description: str, reasoning_formats: str, modified: bool = False, structure_with_llm: bool = False, self_synthesis: bool = False):
    out = self_discover(task_description, reasoning_formats, modified, structure_with_llm, self_synthesis)

    del out["reasoning_modules"]
    del out["task_description"]

    # if isinstance(out["answer_pred"], list):
    #     out["answer_pred"] = ", ".join(out["answer_pred"])

    return out

def bbh(instance):
    reasoning_formats = """
- If the answer is not multiple choice, [answer] should be the decided answer. (For eg: Q: not True or False. A: False)
- If the answer is multiple choice,
    - and the given choices are unlabelled options, [answer] should be the chosen option (For eg: Q: Where does the sun rise from? Options: - East, - West, - North. A: East)
    - and the given choices are labelled options, [answer] should be the letter corresponding to the chosen option (For eg: Q: Where does the sun rise from? Options: - A. West, - B. East, - C. North. A: B)"""
   
    return call_self_discover(instance["input"], reasoning_formats, modified=False, structure_with_llm=False)


def t4d(instance):
    task_description = f"""Observation:
{instance["story"]}

Question:
{instance["question"]}"""
    
    reasoning_formats = """
- should be complete with the letter and correct answer from the list of given choices (Example answer:  K. Ananda))"""
    
    return call_self_discover(task_description, reasoning_formats, True, structure_with_llm=False)

def math():
    pass

def evaluate(benchmark: str, dataset_name: str, subset: str, instance_processor):
    config = read_config()
    
    logger.info("Running evaluations on %s. Loading subset %s from dataset %s", benchmark, subset, dataset_name)
    
    dataset = load_dataset(dataset_name, subset, split="train")

    batch_size = int(config["EVAL"]["BatchSize"])
    checkpoint_dir = here(os.path.join("struct_vs_unstruct", config["PATHS"]["CheckpointDir"], benchmark, f"{benchmark}-{subset}"))
    log_dir = here(os.path.join("struct_vs_unstruct", config["PATHS"]["LogDir"], "evals", benchmark, f"{benchmark}-{subset}"))

    if os.path.exists(os.path.join(checkpoint_dir, f"{benchmark}_eval")):
        logger.debug("The subset %s of the dataset %s has already been processed. Skipping...", subset, dataset_name)

        return "skipped"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config.add_section("CURRENTS")
    config.set("CURRENTS", "log_dir", str(log_dir))
    config = save_config()

    logger.info("Running evaluations on %s dataset in bursts of %s", benchmark, batch_size)

    # Iterate over the dataset in bursts of batch_size
    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{start_idx}_{end_idx}")

        # Skip the batch if it has already been processed
        if os.path.exists(checkpoint_path):
            logger.debug(f"Skipping already processed batch {start_idx}-{end_idx}")
            continue

        # Select the batch and run the evaluation
        batch = dataset.select(range(start_idx, end_idx))
        new_ds = batch.map(instance_processor, load_from_cache_file=False)

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
        y="target",
        y_pred="answer_pred",
        log_file_path=os.path.join(log_dir, f"{benchmark}_different.txt")
    )

    logger.info("Accuracy of %s - %s: %f", dataset_name, subset, accuracy)

    # Log accuracy
    with open(os.path.join(log_dir, f"{benchmark}.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy}\n") 

    return accuracy


if __name__ == "__main__":
    benchmarks = ["t4d", "bbh"]
    dataset_names = ["sachithgunasekara/t4d", "maveriq/bigbenchhard"]
    subset_list = [[""], get_dataset_config_names(dataset_names[1])]
    instance_processors = [t4d, bbh]

    for benchmark, dataset_name, subsets, instance_processor in zip(benchmarks, dataset_names, subset_list, instance_processors):

        for subset in subsets:

            while True:
                try:
                    acc = evaluate(benchmark, dataset_name, subset, instance_processor)

                    if acc is not None:
                        break
                except Exception as e:
                    # Check for the specific Bearer token error
                    if "Rate limit exceeded" in str(e):
                        wait_time = 10
                        print(f"Rate limit exceeded. Waiting for {wait_time} minutes.")
                        time.sleep(wait_time * 60)
                    elif "Invalid invocation request id specified" in str(e):
                        wait_time = 45
                        print(f"Invalid invocation request id specified. Waiting for {wait_time} minutes restart calls...")
                        time.sleep(wait_time * 60)
                    elif "'NoneType' object has no attribute 'group'" in str(e):
                        logger.error("Error extracting answer and trajectory from response. Rerunning...")
                        continue
                    else:
                        # Re-raise the exception if it's not related to Bearer token
                        raise e