from typing import List, Dict, Any, Optional

import logging
from collections import Counter
from reval.dataset_utils import train_val_split
from reval.probing_task_example import ProbingTaskExample

logger = logging.getLogger(__name__)


def generate_task_examples(
    data: List[Dict[str, Any]],
    argument: str,
    position: str,
    pos2idx: [Dict[str, int]],
    keep_tags: List[str],
    split: str,
) -> List[ProbingTaskExample]:
    probing_examples = []

    for example in data:
        entity_start, entity_end = example[argument]
        pos = example["pos"]

        if position == "left":
            # no POS to the left of the entity
            if entity_start == 0:
                continue
            pos_tag = pos[entity_start - 1]
        else:
            if entity_end == (len(pos) - 1):
                continue
            # TODO: make sure this holds for all datasets (end index inclusive)
            pos_tag = pos[entity_end + 1]

        if keep_tags and pos_tag not in keep_tags:
            continue

        probing_examples.append(
            ProbingTaskExample(
                tokens=example["tokens"],
                label=pos2idx[pos_tag],
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
    position: str,
    validation_size: float = 0.1,
    validation_data: Optional[List[Dict[str, Any]]] = None,
    keep_tags: Optional[List[str]] = None,
) -> List[ProbingTaskExample]:
    logger.info(
        "Generating dataset for probing task: "
        + f"PosTag{argument.capitalize()}{position.capitalize()}"
    )

    if argument not in ["head", "tail"]:
        raise ValueError(f"'{argument}' is not a valid argument.")

    if position not in ["left", "right"]:
        raise ValueError(f"'{position}' is not a valid position.")

    if validation_data is None:
        train_data, validation_data = train_val_split(train_data, validation_size)

    logger.info(f"Argument: {argument}")
    logger.info(f"Position: {position}")
    logger.info(f"Num train examples: {len(train_data)}")
    logger.info(f"Num validation examples: {len(validation_data)}")
    logger.info(f"Num test examples: {len(test_data)}")

    all_pos_tags = Counter()
    for data in [train_data, validation_data, test_data]:
        for example in data:
            all_pos_tags.update(example["pos"])

    logger.info(f"Label distribution: {all_pos_tags}")

    pos2idx = {pos_tag: i for i, pos_tag in enumerate(list(all_pos_tags))}

    task_examples = []

    train_task_examples = generate_task_examples(
        train_data, argument, position, pos2idx, keep_tags, split="tr"
    )
    task_examples.extend(train_task_examples)

    idx2pos = {v: k for k, v in pos2idx.items()}
    class_distribution = Counter(
        [idx2pos[example.label] for example in train_task_examples]
    )
    logger.info(f"CT: {class_distribution}")

    validation_task_examples = generate_task_examples(
        validation_data, argument, position, pos2idx, keep_tags, split="va"
    )
    task_examples.extend(validation_task_examples)

    test_task_examples = generate_task_examples(
        test_data, argument, position, pos2idx, keep_tags, split="te"
    )
    task_examples.extend(test_task_examples)

    logger.info(f"Num train task examples: {len(train_task_examples)}")
    logger.info(f"Num validation task examples: {len(validation_task_examples)}")
    logger.info(f"Num test task examples: {len(test_task_examples)}")

    return task_examples
