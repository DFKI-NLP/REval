from typing import Optional, Dict, Any

import fire
import numpy
import logging
from os.path import join
from functools import partial
from collections import Counter
from reval.datasets import (
    load_jsonl_dataset,
    load_tacred_dataset,
    save_probing_task_dataset,
)
from reval.probing_tasks import get_probing_task_generator

logger = logging.getLogger(__name__)

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def generate(
    train_file: str,
    test_file: str,
    output_file: str,
    probing_task: str,
    **kwargs: Dict[str, Any],
) -> None:
    dataset_format = kwargs.pop("dataset_format", "jsonl")
    validation_size = kwargs.pop("validation_size", 0.1)
    seed = kwargs.get("seed", 1111)
    validation_file = kwargs.pop("validation_file", None)

    numpy.random.seed(seed)

    dataset_loader = {"jsonl": load_jsonl_dataset, "tacred": load_tacred_dataset}.get(
        dataset_format
    )

    if dataset_loader is None:
        raise ValueError(f"'{dataset_format}' is not a valid dataset format.")

    train_data = dataset_loader(train_file)
    test_data = dataset_loader(test_file)
    validation_data = dataset_loader(validation_file) if validation_file else None

    kwargs["validation_size"] = validation_size
    kwargs["validation_data"] = validation_data

    probing_task_generator = get_probing_task_generator(probing_task)
    probing_task_examples = probing_task_generator(train_data, test_data, **kwargs)

    class_distribution = Counter([example.label for example in probing_task_examples])
    logger.info(f"Class distribution: {class_distribution.most_common()}")

    save_probing_task_dataset(output_file, probing_task_examples)


def generate_all_from_tacred(
    train_file: str,
    validation_file: str,
    test_file: str,
    output_dir: str,
    **kwargs: Dict[str, Any],
) -> None:
    tacred_generate = partial(
        generate,
        dataset_format="tacred",
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
    )

    tacred_generate(
        probing_task="sentence_length",
        output_file=join(output_dir, "sentence_length.txt"),
        buckets=[
            (5, 15),
            (16, 23),
            (24, 27),
            (28, 31),
            (32, 35),
            (36, 39),
            (40, 43),
            (44, 47),
            (48, 55),
            (55, 70),
        ],
    )

    tacred_generate(
        probing_task="entity_distance",
        output_file=join(output_dir, "entity_distance.txt"),
        buckets=[
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 6),
            (7, 8),
            (9, 11),
            (12, 15),
            (16, 20),
            (21, 25),
        ],
    )

    tacred_generate(
        probing_task="argument_order",
        output_file=join(output_dir, "argument_order.txt"),
    )

    tacred_generate(
        probing_task="entity_exists_between_head_tail",
        output_file=join(output_dir, "entity_exists_between_head_tail.txt"),
    )

    for ner_tag in ["ORGANIZATION", "PERSON", "LOCATION", "MISC", "DATE"]:
        tacred_generate(
            probing_task="entity_type_count_between_head_tail",
            output_file=join(
                output_dir, f"entity_type_count_{ner_tag}_between_head_tail.txt"
            ),
            ner_count_tag=ner_tag,
        )

    for argument, position in [
        ("head", "left"),
        ("head", "right"),
        ("tail", "left"),
        ("tail", "right"),
    ]:
        tacred_generate(
            probing_task="pos_tag_argument_position",
            output_file=join(output_dir, f"pos_tag_{argument}_{position}.txt"),
            argument=argument,
            position=position,
        )

    for argument in ["head", "tail"]:
        tacred_generate(
            probing_task="argument_type",
            output_file=join(output_dir, f"argument_type_{argument}.txt"),
            argument=argument,
        )

    tacred_generate(
        probing_task="tree_depth",
        output_file=join(output_dir, "tree_depth.txt"),
        buckets=[
            (1, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
            (9, 9),
            (10, 10),
            (11, 15),
        ],
    )

    tacred_generate(
        probing_task="sdp_tree_depth",
        output_file=join(output_dir, "sdp_tree_depth.txt"),
        buckets=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10)],
    )

    for argument in ["head", "tail"]:
        tacred_generate(
            probing_task="argument_grammatical_role",
            output_file=join(output_dir, f"argument_{argument}_grammatical_role.txt"),
            argument=argument,
            roles=["nsubj", "dobj", "iobj", "nsubjpass"],
        )


def generate_all_from_semeval(
    train_file: str,
    validation_file: str,
    test_file: str,
    output_dir: str,
    **kwargs: Dict[str, Any],
) -> None:
    semeval_generate = partial(
        generate,
        dataset_format="tacred",
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
    )

    semeval_generate(
        probing_task="sentence_length",
        output_file=join(output_dir, "sentence_length.txt"),
        buckets=[(5, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 35), (36, 70)],
    )

    semeval_generate(
        probing_task="entity_distance",
        output_file=join(output_dir, "entity_distance.txt"),
        buckets=[(1, 2), (3, 4), (5, 6), (7, 8), (9, 25)],
    )

    semeval_generate(
        probing_task="entity_exists_between_head_tail",
        output_file=join(output_dir, "entity_exists_between_head_tail.txt"),
    )

    keep_tags_head_left = ["DT", "JJ", "NN", "IN", "PRP$", "VBN", "NNP"]
    keep_tags_head_right = ["IN", "VBD", "VBZ", "VBP", "VBN", "NN", "WDT"]
    keep_tags_tail_left = ["DT", "JJ", "NN", "IN", "PRP$", "NNP", "POS"]
    keep_tags_tail_right = [".", "IN", ",", "CC", "VBZ", "VBD", "TO"]

    for argument, position, keep_tags in [
        ("head", "left", keep_tags_head_left),
        ("head", "right", keep_tags_head_right),
        ("tail", "left", keep_tags_tail_left),
        ("tail", "right", keep_tags_tail_right),
    ]:
        semeval_generate(
            probing_task="pos_tag_argument_position",
            output_file=join(output_dir, f"pos_tag_{argument}_{position}.txt"),
            argument=argument,
            position=position,
            keep_tags=keep_tags,
        )

    keep_types_head = [
        "Other",
        "Entity",
        "Effect",
        "Collection",
        "Whole",
        "Message",
        "Component",
        "Agency",
        "Producer",
        "Content",
    ]
    keep_types_tail = [
        "Other",
        "Destination",
        "Cause",
        "Member",
        "Origin",
        "Component",
        "Topic",
        "Whole",
        "Instrument",
        "Product",
    ]

    for argument, keep_types in [("head", keep_types_head), ("tail", keep_types_tail)]:
        semeval_generate(
            probing_task="argument_type",
            output_file=join(output_dir, f"argument_type_{argument}.txt"),
            argument=argument,
            keep_types=keep_types,
        )

    semeval_generate(
        probing_task="tree_depth",
        output_file=join(output_dir, "tree_depth.txt"),
        buckets=[(1, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 15)],
    )

    semeval_generate(
        probing_task="sdp_tree_depth",
        output_file=join(output_dir, "sdp_tree_depth.txt"),
        buckets=[(1, 1), (2, 2), (3, 3), (4, 10)],
    )

    for argument in ["head", "tail"]:
        semeval_generate(
            probing_task="argument_grammatical_role",
            output_file=join(output_dir, f"argument_{argument}_grammatical_role.txt"),
            argument=argument,
            roles=["nsubj", "dobj", "iobj", "nsubjpass"],
        )


fire.Fire()
