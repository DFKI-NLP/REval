from typing import List, Dict, Any, Optional

import logging
from reval.dataset_utils import train_val_split
from reval.probing_task_example import ProbingTaskExample

logger = logging.getLogger(__name__)


def generate(
    generate_task_examples,
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    validation_size: float = 0.1,
    validation_data: Optional[List[Dict[str, Any]]] = None,
) -> List[ProbingTaskExample]:

    if validation_data is None:
        train_data, validation_data = train_val_split(train_data, validation_size)

    logger.info(f"Num train examples: {len(train_data)}")
    logger.info(f"Num validation examples: {len(validation_data)}")
    logger.info(f"Num test examples: {len(test_data)}")

    task_examples = []

    train_task_examples = generate_task_examples(train_data, split="tr")
    task_examples.extend(train_task_examples)

    validation_task_examples = generate_task_examples(validation_data, split="va")
    task_examples.extend(validation_task_examples)

    test_task_examples = generate_task_examples(test_data, split="te")
    task_examples.extend(test_task_examples)

    logger.info(f"Num train task examples: {len(train_task_examples)}")
    logger.info(f"Num validation task examples: {len(validation_task_examples)}")
    logger.info(f"Num test task examples: {len(test_task_examples)}")

    return task_examples
