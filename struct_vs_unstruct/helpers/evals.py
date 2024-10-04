from tqdm import tqdm


t4d = lambda instance, y, y_pred: instance[y] in instance[y_pred] and instance == instance[y_pred][3:]
bbh = lambda instance, y, y_pred: instance[y] == instance[y_pred] or instance[y][min(len(instance[y]) - 1, 1)] == instance[y_pred]


def calculate_accuracy(full_dataset, benchmark, y: str, y_pred: str, log_file_path: str):
    correct_preds = 0
    for instance in tqdm(full_dataset, desc="Calculating accuracy"):
        if benchmark == "t4d":
            eval_fn = t4d
        elif benchmark == "bbh":
            eval_fn = bbh

        if eval_fn(instance, y, y_pred):
            correct_preds += 1
        else:
            with open(log_file_path, "a") as f:
                f.write(f"{instance[y]}, {instance[y_pred]}\n")
    return correct_preds / len(full_dataset)