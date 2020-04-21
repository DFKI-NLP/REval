from typing import List, Dict, Any, Optional

import logging
from reval.probing_tasks import probing_task_base
from reval.probing_task_example import ProbingTaskExample

logger = logging.getLogger(__name__)


def generate_task_examples(
    data: List[Dict[str, Any]], split: str
) -> List[ProbingTaskExample]:

    probing_examples = []

    for example in data:
        head_start, head_end = example["head"]
        tail_start, tail_end = example["tail"]
        start = min(head_end, tail_end)
        end = max(head_start, tail_start)

        ner_tags = set(example["ner"][start + 1 : end]) - {"O"}

        probing_examples.append(
            ProbingTaskExample(
                tokens=example["tokens"],
                label="1" if len(ner_tags) > 0 else "0",
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
) -> List[ProbingTaskExample]:
    logger.info("Generating dataset for probing task: EntExistsBetween")

    return probing_task_base.generate(
        generate_task_examples, train_data, test_data, validation_size, validation_data
    )
