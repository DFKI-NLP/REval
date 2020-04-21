from typing import List, Dict, Any, Optional

import logging
from reval.dataset_utils import train_val_split
from reval.probing_task_example import ProbingTaskExample

logger = logging.getLogger(__name__)

MAX_COUNT = 5


def length_in_bucket(tags, bucket):
    bucket_min, bucket_max = bucket
    return bucket_min <= len(tags) <= bucket_max


def generate_task_examples(
    data: List[Dict[str, Any]], tag: str, split: str
) -> List[ProbingTaskExample]:

    probing_examples = []

    for example in data:
        head_start, head_end = example["head"]
        tail_start, tail_end = example["tail"]
        start = min(head_end, tail_end)
        end = max(head_start, tail_start)

        ner_tags = [t for t in example["ner"][start + 1 : end] if t == tag]

        probing_examples.append(
            ProbingTaskExample(
                tokens=example["tokens"],
                label=str(len(ner_tags)) if len(ner_tags) < 5 else MAX_COUNT,
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
    validation_size: float = 0.1,
    validation_data: Optional[List[Dict[str, Any]]] = None,
    ner_count_tag: str = "ORGANIZATION",
) -> List[ProbingTaskExample]:
    logger.info("Generating dataset for probing task: EntCountBetween")

    if validation_data is None:
        train_data, validation_data = train_val_split(train_data, validation_size)

    logger.info(f"Using NER tag: {ner_count_tag}")
    logger.info(f"Num train examples: {len(train_data)}")
    logger.info(f"Num validation examples: {len(validation_data)}")
    logger.info(f"Num test examples: {len(test_data)}")

    task_examples = []

    train_task_examples = generate_task_examples(train_data, ner_count_tag, split="tr")
    task_examples.extend(train_task_examples)

    validation_task_examples = generate_task_examples(
        validation_data, ner_count_tag, split="va"
    )
    task_examples.extend(validation_task_examples)

    test_task_examples = generate_task_examples(test_data, ner_count_tag, split="te")
    task_examples.extend(test_task_examples)

    logger.info(f"Num train task examples: {len(train_task_examples)}")
    logger.info(f"Num validation task examples: {len(validation_task_examples)}")
    logger.info(f"Num test task examples: {len(test_task_examples)}")

    return task_examples
