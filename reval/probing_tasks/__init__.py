import os
import io
import logging
import numpy as np
from senteval.tools.validation import SplitClassifier
from reval.probing_tasks import (
    sent_length,
    entity_distance,
    argument_order,
    entity_exists_between_head_tail,
    entity_type_count_between_head_tail,
    pos_tag_argument_position,
    argument_type,
    tree_depth,
    sdp_tree_depth,
    argument_grammatical_role,
)


def get_probing_task_generator(name: str):
    task_generator = {
        "sentence_length": sent_length.generate,
        "entity_distance": entity_distance.generate,
        "argument_order": argument_order.generate,
        "entity_exists_between_head_tail": entity_exists_between_head_tail.generate,
        "entity_type_count_between_head_tail": entity_type_count_between_head_tail.generate,
        "pos_tag_argument_position": pos_tag_argument_position.generate,
        "argument_type": argument_type.generate,
        "tree_depth": tree_depth.generate,
        "sdp_tree_depth": sdp_tree_depth.generate,
        "argument_grammatical_role": argument_grammatical_role.generate,
    }.get(name)

    if task_generator is None:
        raise ValueError(f"'{name}' is not a valid probing task.")

    return task_generator


class REPROBINGEval(object):
    def __init__(self, task, task_path, seed=1111):
        self.seed = seed
        self.task = task
        logging.debug(
            "***** (Probing) Transfer task : %s classification *****", self.task.upper()
        )
        self.task_data = {
            "train": {
                "X": [],
                "id": [],
                "head": [],
                "tail": [],
                "ner": [],
                "pos": [],
                "dep": [],
                "dep_head": [],
                "y": [],
            },
            "dev": {
                "X": [],
                "id": [],
                "head": [],
                "tail": [],
                "ner": [],
                "pos": [],
                "dep": [],
                "dep_head": [],
                "y": [],
            },
            "test": {
                "X": [],
                "id": [],
                "head": [],
                "tail": [],
                "ner": [],
                "pos": [],
                "dep": [],
                "dep_head": [],
                "y": [],
            },
        }
        self.loadFile(task_path)
        logging.info(
            "Loaded %s train - %s dev - %s test for %s"
            % (
                len(self.task_data["train"]["y"]),
                len(self.task_data["dev"]["y"]),
                len(self.task_data["test"]["y"]),
                self.task,
            )
        )

    def do_prepare(self, params, prepare):
        samples = (
            self.task_data["train"]["X"]
            + self.task_data["dev"]["X"]
            + self.task_data["test"]["X"]
        )
        return prepare(params, samples)

    def loadFile(self, fpath):
        self.tok2split = {"tr": "train", "va": "dev", "te": "test"}
        with io.open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip().split("\t")
                self.task_data[self.tok2split[line[0]]]["X"].append(line[-1].split())
                self.task_data[self.tok2split[line[0]]]["id"].append(line[1])
                self.task_data[self.tok2split[line[0]]]["y"].append(line[2])
                self.task_data[self.tok2split[line[0]]]["head"].append(
                    (int(line[3]), int(line[4]))
                )
                self.task_data[self.tok2split[line[0]]]["tail"].append(
                    (int(line[5]), int(line[6]))
                )
                self.task_data[self.tok2split[line[0]]]["ner"].append(line[7].split())
                self.task_data[self.tok2split[line[0]]]["pos"].append(line[8].split())
                self.task_data[self.tok2split[line[0]]]["dep"].append(line[9].split())
                self.task_data[self.tok2split[line[0]]]["dep_head"].append(
                    list(map(int, line[10].split()))
                )

        labels = sorted(np.unique(self.task_data["train"]["y"]))
        self.tok2label = dict(zip(labels, range(len(labels))))
        self.nclasses = len(self.tok2label)

        for split in self.task_data:
            for i, y in enumerate(self.task_data[split]["y"]):
                self.task_data[split]["y"][i] = self.tok2label[y]

    def run(self, params, batcher):
        task_embed = {"train": {}, "dev": {}, "test": {}}
        bsize = params.batch_size
        logging.info("Computing embeddings for train/dev/test")
        for key in self.task_data:
            # Sort to reduce padding
            sorted_data = sorted(
                zip(
                    self.task_data[key]["X"],
                    self.task_data[key]["id"],
                    self.task_data[key]["y"],
                    self.task_data[key]["head"],
                    self.task_data[key]["tail"],
                    self.task_data[key]["ner"],
                    self.task_data[key]["pos"],
                    self.task_data[key]["dep"],
                    self.task_data[key]["dep_head"],
                ),
                key=lambda z: (len(z[0]), z[1]),
            )
            (
                self.task_data[key]["X"],
                self.task_data[key]["id"],
                self.task_data[key]["y"],
                self.task_data[key]["head"],
                self.task_data[key]["tail"],
                self.task_data[key]["ner"],
                self.task_data[key]["pos"],
                self.task_data[key]["dep"],
                self.task_data[key]["dep_head"],
            ) = map(list, zip(*sorted_data))

            task_embed[key]["X"] = []
            for ii in range(0, len(self.task_data[key]["y"]), bsize):
                batch = self.task_data[key]["X"][ii : ii + bsize]
                id_ = self.task_data[key]["id"][ii : ii + bsize]
                id_ = id_ if id_ != "None" else None
                head = self.task_data[key]["head"][ii : ii + bsize]
                tail = self.task_data[key]["tail"][ii : ii + bsize]
                ner = self.task_data[key]["ner"][ii : ii + bsize]
                pos = self.task_data[key]["pos"][ii : ii + bsize]
                dep = self.task_data[key]["dep"][ii : ii + bsize]
                dep_head = self.task_data[key]["dep_head"][ii : ii + bsize]

                embeddings = batcher(
                    params, batch, head, tail, ner, pos, dep, dep_head, id_
                )
                task_embed[key]["X"].append(embeddings)
            task_embed[key]["X"] = np.vstack(task_embed[key]["X"])
            task_embed[key]["y"] = np.array(self.task_data[key]["y"])
        logging.info("Computed embeddings")

        config_classifier = {
            "nclasses": self.nclasses,
            "seed": self.seed,
            "usepytorch": params.usepytorch,
            "classifier": params.classifier,
        }

        # if self.task == "WordContent" and params.classifier["nhid"] > 0:
        #     config_classifier = copy.deepcopy(config_classifier)
        #     config_classifier["classifier"]["nhid"] = 0
        #     print(params.classifier["nhid"])

        clf = SplitClassifier(
            X={
                "train": task_embed["train"]["X"],
                "valid": task_embed["dev"]["X"],
                "test": task_embed["test"]["X"],
            },
            y={
                "train": task_embed["train"]["y"],
                "valid": task_embed["dev"]["y"],
                "test": task_embed["test"]["y"],
            },
            config=config_classifier,
        )

        devacc, testacc = clf.run()
        logging.debug(
            "\nDev acc : %.1f Test acc : %.1f for %s classification\n"
            % (devacc, testacc, self.task.upper())
        )

        return {
            "devacc": devacc,
            "acc": testacc,
            "ndev": len(task_embed["dev"]["X"]),
            "ntest": len(task_embed["test"]["X"]),
        }


"""
Surface Information
"""


class LengthEval(REPROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, "sentence_length.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "Length", task_path, seed)


class EntityDistanceEval(REPROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, "entity_distance.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "EntityDistance", task_path, seed)


class ArgumentOrderEval(REPROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, "argument_order.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "ArgumentOrder", task_path, seed)


class EntityExistsBetweenHeadTailEval(REPROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, "entity_exists_between_head_tail.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "EntityExistsBetweenHeadTail", task_path, seed)


class EntityTypeCountBetweenHeadTailEval(REPROBINGEval):
    def __init__(self, task_path, ner_tag="ORG", seed=1111):
        task_path = os.path.join(
            task_path, f"entity_type_count_{ner_tag}_between_head_tail.txt"
        )
        # labels: bins
        REPROBINGEval.__init__(
            self, f"EntityTypeCount{ner_tag}BetweenHeadTail", task_path, seed
        )


class PosTagArgPositionEval(REPROBINGEval):
    def __init__(self, task_path, argument, position, seed=1111):
        task_path = os.path.join(task_path, f"pos_tag_{argument}_{position}.txt")
        # labels: bins
        REPROBINGEval.__init__(
            self,
            f"PosTag{argument.capitalize()}{position.capitalize()}",
            task_path,
            seed,
        )


class ArgumentTypeEval(REPROBINGEval):
    def __init__(self, task_path, argument, seed=1111):
        task_path = os.path.join(task_path, f"argument_type_{argument}.txt")
        # labels: bins
        REPROBINGEval.__init__(self, f"ArgType{argument.capitalize()}", task_path, seed)


class TreeDepthEval(REPROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, "tree_depth.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "TreeDepth", task_path, seed)


class SDPTreeDepthEval(REPROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, "sdp_tree_depth.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "SDPTreeDepth", task_path, seed)


class ArgumentGrammaticalRoleEval(REPROBINGEval):
    def __init__(self, task_path, argument, seed=1111):
        task_path = os.path.join(task_path, f"argument_{argument}_grammatical_role.txt")
        REPROBINGEval.__init__(
            self, f"Argument{argument.capitalize()}GrammaticalRole", task_path, seed
        )
