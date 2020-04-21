#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 19.02.19
@author: leonhard.hennig@dfki.de
"""

# head/tail
# nsubj, npassivesubj, dobj, indirobj
# nur wenn alle tokens des head/tail args ein konsistenter, geschlossener teilgraph
# nur wenn label direkt über root von head/tail teilgraph
from typing import List, Dict, Any, Tuple, Optional

import logging
from reval.dataset_utils import train_val_split
from reval.probing_task_example import ProbingTaskExample
from reval.dependency_graph_utils import dep_heads_to_tree, Tree

logger = logging.getLogger(__name__)

DEFAULT_ROLES = ["nsubj", "dobj", "iobj", "nsubjpass"]


def find_common_head(
    start: int, end: int, example: Dict[str, Any]
) -> Tuple[int, int, int]:
    """
    Tests if all nodes in [start,end] (inclusive) have a common ancestor that is part of [start,end], and returns
    the index of that node.
    :param start:
    :param end:
    :param tree:
    :return:
    """
    heads = [
        (i, example["dep_head"][i]) for i in range(start, end + 1)
    ]  # heads are 1-based!
    outside_head = set()
    for head in heads:
        if head[1] - 1 not in range(start, end + 1):  # heads are 1-based!
            outside_head.add(head[1])
    if len(outside_head) != 1:
        return (-1, -1, -1)
    last_child = None
    for head in heads:
        if head[1] == list(outside_head)[0]:
            last_child = head
    return last_child[0], last_child[1], example["dep"][last_child[0]]


def generate_task_examples(
    data: List[Dict[str, Any]], argument: str, roles: List[str], split: str
) -> List[ProbingTaskExample]:

    probing_examples = []

    for example in data:
        # dep_heads = example["dep_head"]
        # dep_labels = example["dep"]
        # tree = dep_heads_to_tree(
        #     dep_heads,
        #     len(example["tokens"]),
        #     example["head"],
        #     example["tail"],
        #     prune=0,
        #     dep_labels=dep_labels,
        #     tokens=example["tokens"]
        # )
        arg_start, arg_end = example[argument]
        idx, head, dep_rel = find_common_head(
            arg_start, arg_end, example
        )  # heads are 1-based!
        dep_heads = example["dep_head"]
        dep_labels = example["dep"]
        tree = dep_heads_to_tree(
            dep_heads,
            len(example["tokens"]),
            example["head"],
            example["tail"],
            prune=0,
            dep_labels=dep_labels,
            tokens=example["tokens"],
        )
        if idx >= 0:
            probing_examples.append(
                ProbingTaskExample(
                    tokens=example["tokens"],
                    label=str(roles.index(dep_rel) + 1)
                    if dep_rel in DEFAULT_ROLES
                    else "0",
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
    argument: str = "head",
    roles: Optional[List[str]] = None,
) -> List[ProbingTaskExample]:
    logger.info("Generating dataset for probing task: ArgumentGrammaticalRole")
    if argument not in {"head", "tail"}:
        raise (f"Invalid argument [{argument}]")
    if roles is None:
        roles = DEFAULT_ROLES
    if validation_data is None:
        train_data, validation_data = train_val_split(train_data, validation_size)

    logger.info(f"Using argument: {argument}")
    logger.info(f"Num train examples: {len(train_data)}")
    logger.info(f"Num validation examples: {len(validation_data)}")
    logger.info(f"Num test examples: {len(test_data)}")

    task_examples = []

    train_task_examples = generate_task_examples(
        train_data, argument, roles, split="tr"
    )
    task_examples.extend(train_task_examples)

    validation_task_examples = generate_task_examples(
        validation_data, argument, roles, split="va"
    )
    task_examples.extend(validation_task_examples)

    test_task_examples = generate_task_examples(test_data, argument, roles, split="te")
    task_examples.extend(test_task_examples)

    logger.info(f"Num train task examples: {len(train_task_examples)}")
    logger.info(f"Num validation task examples: {len(validation_task_examples)}")
    logger.info(f"Num test task examples: {len(test_task_examples)}")

    return task_examples
