#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 13.02.19
@author: leonhard.hennig@dfki.de
"""

from typing import List, Dict, Any, Tuple

import logging
from collections import Counter
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def train_val_split(train_data: List[Dict[str, Any]],
                    validation_size: float = 0.1) -> Tuple[List[Dict[str, Any]],List[Dict[str, Any]]]:

    logger.info("Splitting training data into train and validation dataset.")

    # stratified splitting requires a class to be present at least twice
    counter = Counter([example["label"] for example in train_data])
    labels_to_filter = [
        label for label, count in counter.most_common() if count < 2
    ]
    logger.info(f"Labels to filter: {labels_to_filter}")

    filtered_train_data = [
        example
        for example in train_data
        if example["label"] not in labels_to_filter
    ]
    labels = [example["label"] for example in filtered_train_data]
    train_data, validation_data = train_test_split(
        filtered_train_data, test_size=validation_size, stratify=labels
    )
    return train_data, validation_data
