from typing import List, Tuple, Dict, Any

import os
import json
import csv
from reval.probing_task_example import ProbingTaskExample


def load_jsonl_dataset(file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip("\n")
            example = json.loads(line)

            tokens = example["tokens"]
            head, tail = example["entities"]
            relation = example["label"]

            dataset.append(dict(tokens=tokens, label=relation, head=head, tail=tail))
    return dataset


def load_tacred_dataset(file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset = []

    with open(file_path, "r") as f:
        data = json.load(f)
        for example in data:
            tokens = example["token"]
            head = (example["subj_start"], example["subj_end"])
            tail = (example["obj_start"], example["obj_end"])
            relation = example["relation"]

            ner = example["stanford_ner"]
            pos = example["stanford_pos"]
            dep = example["stanford_deprel"]
            dep_head = example["stanford_head"]
            head_type = example["subj_type"]
            tail_type = example["obj_type"]

            id_ = example["id"]

            dataset.append(
                dict(
                    tokens=tokens,
                    label=relation,
                    head=head,
                    tail=tail,
                    ner=ner,
                    pos=pos,
                    dep=dep,
                    dep_head=dep_head,
                    head_type=head_type,
                    tail_type=tail_type,
                    id=id_,
                )
            )
    return dataset


def save_probing_task_dataset(
    file_path: str, examples: List[ProbingTaskExample]
) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for example in examples:
            writer.writerow(
                [
                    example.split,
                    example.id if example.id else "None",
                    example.label,
                    example.head[0],
                    example.head[1],
                    example.tail[0],
                    example.tail[1],
                    " ".join(example.ner),
                    " ".join(example.pos),
                    " ".join(example.dep),
                    " ".join(map(str, example.dep_head)),
                    " ".join(example.tokens),
                ]
            )
