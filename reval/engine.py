from __future__ import absolute_import, division, unicode_literals

from senteval import utils
from reval.probing_tasks import (
    LengthEval,
    EntityDistanceEval,
    ArgumentOrderEval,
    EntityExistsBetweenHeadTailEval,
    EntityTypeCountBetweenHeadTailEval,
    PosTagArgPositionEval,
    ArgumentTypeEval,
    TreeDepthEval,
    SDPTreeDepthEval,
    ArgumentGrammaticalRoleEval,
)


class RE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if "usepytorch" not in params else params.usepytorch
        params.seed = 1111 if "seed" not in params else params.seed

        params.batch_size = 128 if "batch_size" not in params else params.batch_size
        params.nhid = 0 if "nhid" not in params else params.nhid
        params.kfold = 5 if "kfold" not in params else params.kfold

        if "classifier" not in params or not params["classifier"]:
            params.classifier = {"nhid": 0}

        if "nhid" not in params.classifier:
            raise ValueError(
                "Number of hidden units not set. Please set number of hidden units "
                + "in classifier config."
            )

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = [
            "Length",
            "EntityDistance",
            "ArgumentOrder",
            "EntityExistsBetweenHeadTail",
            "EntityCountORGBetweenHeadTail",
            "EntityCountPERBetweenHeadTail",
            "EntityCountDATEBetweenHeadTail",
            "EntityCountMISCBetweenHeadTail",
            "EntityCountLOCBetweenHeadTail",
            "PosTagHeadLeft",
            "PosTagHeadRight",
            "PosTagTailLeft",
            "PosTagTailRight",
            "ArgTypeHead",
            "ArgTypeTail",
            "TreeDepth",
            "SDPTreeDepth",
            "ArgumentHeadGrammaticalRole",
            "ArgumentTailGrammaticalRole",
        ]

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if isinstance(name, list):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + " not in " + str(self.list_tasks)

        # Probing Tasks
        if name == "Length":
            self.evaluation = LengthEval(tpath, seed=self.params.seed)
        elif name == "EntityDistance":
            self.evaluation = EntityDistanceEval(tpath, seed=self.params.seed)
        elif name == "ArgumentOrder":
            self.evaluation = ArgumentOrderEval(tpath, seed=self.params.seed)
        elif name == "EntityExistsBetweenHeadTail":
            self.evaluation = EntityExistsBetweenHeadTailEval(
                tpath, seed=self.params.seed
            )
        elif name == "EntityCountORGBetweenHeadTail":
            self.evaluation = EntityTypeCountBetweenHeadTailEval(
                tpath, "ORGANIZATION", seed=self.params.seed
            )
        elif name == "EntityCountPERBetweenHeadTail":
            self.evaluation = EntityTypeCountBetweenHeadTailEval(
                tpath, "PERSON", seed=self.params.seed
            )
        elif name == "EntityCountLOCBetweenHeadTail":
            self.evaluation = EntityTypeCountBetweenHeadTailEval(
                tpath, "LOCATION", seed=self.params.seed
            )
        elif name == "EntityCountMISCBetweenHeadTail":
            self.evaluation = EntityTypeCountBetweenHeadTailEval(
                tpath, "MISC", seed=self.params.seed
            )
        elif name == "EntityCountDATEBetweenHeadTail":
            self.evaluation = EntityTypeCountBetweenHeadTailEval(
                tpath, "DATE", seed=self.params.seed
            )
        elif name == "PosTagHeadLeft":
            self.evaluation = PosTagArgPositionEval(
                tpath, "head", "left", seed=self.params.seed
            )
        elif name == "PosTagHeadRight":
            self.evaluation = PosTagArgPositionEval(
                tpath, "head", "right", seed=self.params.seed
            )
        elif name == "PosTagTailLeft":
            self.evaluation = PosTagArgPositionEval(
                tpath, "tail", "left", seed=self.params.seed
            )
        elif name == "PosTagTailRight":
            self.evaluation = PosTagArgPositionEval(
                tpath, "tail", "right", seed=self.params.seed
            )
        elif name == "ArgTypeHead":
            self.evaluation = ArgumentTypeEval(tpath, "head", seed=self.params.seed)
        elif name == "ArgTypeTail":
            self.evaluation = ArgumentTypeEval(tpath, "tail", seed=self.params.seed)
        elif name == "TreeDepth":
            self.evaluation = TreeDepthEval(tpath, seed=self.params.seed)
        elif name == "SDPTreeDepth":
            self.evaluation = SDPTreeDepthEval(tpath, seed=self.params.seed)
        elif name == "ArgumentHeadGrammaticalRole":
            self.evaluation = ArgumentGrammaticalRoleEval(
                tpath, "head", seed=self.params.seed
            )
        elif name == "ArgumentTailGrammaticalRole":
            self.evaluation = ArgumentGrammaticalRoleEval(
                tpath, "tail", seed=self.params.seed
            )
        else:
            raise ValueError(f"'{name}' is not a valid task.")

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
