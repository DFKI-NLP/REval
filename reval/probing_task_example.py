from typing import List, Tuple, Optional


class ProbingTaskExample:
    def __init__(
        self,
        tokens: List[str],
        label: str,
        split: str,
        head: Tuple[int, int],
        tail: Tuple[int, int],
        ner: List[str],
        pos: List[str],
        dep: List[str],
        dep_head: List[str],
        id: Optional[str] = None,
    ) -> None:
        self.tokens = tokens
        self.label = label
        self.split = split
        self.head = head
        self.tail = tail
        self.ner = ner
        self.pos = pos
        self.dep = dep
        self.dep_head = dep_head
        self.id = id
