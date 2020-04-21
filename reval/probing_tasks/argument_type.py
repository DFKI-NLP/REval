from typing import List, Dict, Any, Optional

import logging
from collections import Counter
from reval.dataset_utils import train_val_split
from reval.probing_task_example import ProbingTaskExample

logger = logging.getLogger(__name__)


def generate_task_examples(
    data: List[Dict[str, Any]],
    argument: str,
    type2idx: [Dict[str, int]],
    keep_types: List[str],
    split: str,
) -> List[ProbingTaskExample]:

    arg_type_field = f"{argument}_type"

    probing_examples = []

    for example in data:
        arg_type = example[arg_type_field]

        if keep_types and arg_type not in keep_types:
            continue

        probing_examples.append(
            ProbingTaskExample(
                tokens=example["tokens"],
                label=type2idx[arg_type],
                split=split,
                head=example["head"],
                tail=example["tail"],
                ner=example["ner"],
                pos=example["pos"],
                dep=example["dep"],
                dep_head=example["dep_head"],
                id=example["id"],
            )
        )
    return probing_examples


def generate(
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    argument: str,
    validation_size: float = 0.1,
    validation_data: Optional[List[Dict[str, Any]]] = None,
    keep_types: Optional[List[str]] = None,
) -> List[ProbingTaskExample]:
    logger.info(
        f"Generating dataset for probing task: ArgumentType{argument.capitalize()}"
    )

    if argument not in ["head", "tail"]:
        raise ValueError(f"'{argument}' is not a valid argument.")

    if validation_data is None:
        train_data, validation_data = train_val_split(train_data, validation_size)

    logger.info(f"Argument: {argument}")
    logger.info(f"Num train examples: {len(train_data)}")
    logger.info(f"Num validation examples: {len(validation_data)}")
    logger.info(f"Num test examples: {len(test_data)}")

    all_arg_types = Counter()
    for data in [train_data, validation_data, test_data]:
        for example in data:
            field = f"{argument}_type"
            all_arg_types.update([example[field]])

    logger.info(f"Label distribution: {all_arg_types.most_common()}")

    type2idx = {arg_type: i for i, arg_type in enumerate(list(all_arg_types))}

    task_examples = []

    train_task_examples = generate_task_examples(
        train_data, argument, type2idx, keep_types, split="tr"
    )
    task_examples.extend(train_task_examples)

    idx2type = {v: k for k, v in type2idx.items()}
    class_distribution = Counter(
        [idx2type[example.label] for example in train_task_examples]
    )
    logger.info(f"CT: {class_distribution}")

    validation_task_examples = generate_task_examples(
        validation_data, argument, type2idx, keep_types, split="va"
    )
    task_examples.extend(validation_task_examples)

    test_task_examples = generate_task_examples(
        test_data, argument, type2idx, keep_types, split="te"
    )
    task_examples.extend(test_task_examples)

    logger.info(f"Num train task examples: {len(train_task_examples)}")
    logger.info(f"Num validation task examples: {len(validation_task_examples)}")
    logger.info(f"Num test task examples: {len(test_task_examples)}")

    return task_examples
