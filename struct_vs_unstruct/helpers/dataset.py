import os

from datasets import Dataset, concatenate_datasets
from pyprojroot import here


def load_checkpoints(checkpoint_dir: str, benchmark: str):
    all_datasets = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint"):
            ds = Dataset.load_from_disk(os.path.join(checkpoint_dir, file))
            all_datasets.append(ds)
    full_dataset = concatenate_datasets(all_datasets)
    
    full_dataset.save_to_disk(here(os.path.join(checkpoint_dir, f"{benchmark}_eval")))

    return full_dataset
