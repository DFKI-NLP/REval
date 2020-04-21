from typing import List, Dict, Any, Optional, Tuple

import logging
from reval.dataset_utils import train_val_split
from reval.probing_task_example import ProbingTaskExample

logger = logging.getLogger(__name__)


# default buckets as used in Adi et al. 2016
# DEFAULT_BUCKETS = [
#     (5, 8),
#     (9, 12),
#     (13, 16),
#     (17, 20),
#     (21, 25),
#     (26, 29),
#     (30, 33),
#     (34, 70),
# ]

DEFAULT_BUCKETS = [
    (4, 15),
    (16, 23),
    (24, 27),
    (28, 31),
    (32, 35),
    (36, 39),
    (40, 43),
    (44, 47),
    (48, 55),
    (55, 70),
]


def generate_task_examples(
    data: List[Dict[str, Any]], buckets: List[Tuple[int, int]], split: str
) -> List[ProbingTaskExample]:
    def length_in_bucket(tokens, bucket):
        bucket_min, bucket_max = bucket
        return bucket_min <= len(tokens) <= bucket_max

    probing_examples = []

    for example in data:
        tokens = example["tokens"]
        bucket_index = None
        for idx, bucket in enumerate(buckets):
            if length_in_bucket(tokens, bucket):
                bucket_index = idx
                break

        # discard examples that are either to long or to short
        if bucket_index is None:
            continue

        probing_examples.append(
            ProbingTaskExample(
                tokens,
                label=str(bucket_index),
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
    buckets: Optional[List[Tuple[int, int]]] = None,
) -> List[ProbingTaskExample]:
    logger.info("Generating dataset for probing task: SentLength")

    if buckets is None:
        buckets = DEFAULT_BUCKETS

    logger.info(f"Buckets: {buckets}")

    if validation_data is None:
        train_data, validation_data = train_val_split(train_data, validation_size)

    logger.info(f"Num train examples: {len(train_data)}")
    logger.info(f"Num validation examples: {len(validation_data)}")
    logger.info(f"Num test examples: {len(test_data)}")

    task_examples = []

    train_task_examples = generate_task_examples(train_data, buckets, split="tr")
    task_examples.extend(train_task_examples)

    validation_task_examples = generate_task_examples(
        validation_data, buckets, split="va"
    )
    task_examples.extend(validation_task_examples)

    test_task_examples = generate_task_examples(test_data, buckets, split="te")
    task_examples.extend(test_task_examples)

    logger.info(f"Num train task examples: {len(train_task_examples)}")
    logger.info(f"Num validation task examples: {len(validation_task_examples)}")
    logger.info(f"Num test task examples: {len(test_task_examples)}")

    return task_examples
