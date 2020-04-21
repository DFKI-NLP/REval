# REval

## Table of Contents

* [Introduction](#introduction)
* [Overview](#-overview)
* [Requirements](#-requirements)
* [Installation](#-installation)
* [Probing](#-probing)
* [Usage](#-usage)
* [Citation](#-citation)
* [License](#-license)


## 🎓&nbsp; Introduction

REval is a simple framework for probing sentence-level representations of Relation Extraction models.

## ✅&nbsp; Requirements

REval is tested with:

- Python 3.7


## 🚀&nbsp; Installation

### With pip

```bash
<TBD>
```

### From source
```bash
git clone https://github.com/DFKI-NLP/REval
cd REval
pip install -r requirements.txt
```

## 🔬&nbsp; Probing

### Supported Datasets
- SemEval 2010 Task 8 (CoreNLP annotated version) [[LINK](https://cloud.dfki.de/owncloud/index.php/s/LFDHFmRcrPyf6nL/download)]
- TACRED (obtained via LDC) [[LINK](https://catalog.ldc.upenn.edu/LDC2018T24)]

### Probing Tasks

| Task     	| SemEval 2010        | TACRED             |
|----------	| :-----------------: | :----------------: |
|ArgTypeHead| :heavy_check_mark:  | :heavy_check_mark: |
|ArgTypeTail| :heavy_check_mark:  | :heavy_check_mark: |
|Length| :heavy_check_mark:  | :heavy_check_mark: |
|EntityDistance| :heavy_check_mark: | :heavy_check_mark: |
|ArgumentOrder|  | :heavy_check_mark: |
|EntityExistsBetweenHeadTail| :heavy_check_mark:  | :heavy_check_mark: |
|PosTagHeadLeft| :heavy_check_mark:  | :heavy_check_mark: |
|PosTagHeadRight| :heavy_check_mark:  | :heavy_check_mark: |
|PosTagTailLeft| :heavy_check_mark:  | :heavy_check_mark: |
|PosTagTailRight| :heavy_check_mark:  | :heavy_check_mark: |
|TreeDepth| :heavy_check_mark:  | :heavy_check_mark: |
|SDPTreeDepth| :heavy_check_mark:  | :heavy_check_mark: |
|ArgumentHeadGrammaticalRole| :heavy_check_mark:  | :heavy_check_mark: |
|ArgumentTailGrammaticalRole| :heavy_check_mark:  | :heavy_check_mark: |


## 🔧&nbsp; Usage

### **Step 1**: create the probing task datasets from the original [datasets](#supported-datasets).

#### SemEval 2010 Task 8

```bash
python reval.py generate-all-from-semeval \
    --train-file <SEMEVAL DIR>/train.json \
    --validation-file <SEMEVAL DIR>/dev.json \
    --test-file <SEMEVAL DIR>/test.json \
    --output-dir ./data/semeval/
```

#### TACRED

```bash
python reval.py generate-all-from-tacred \
    --train-file <TACRED DIR>/train.json \
    --validation-file <TACRED DIR>/dev.json \
    --test-file <TACRED DIR>/test.json \
    --output-dir ./data/tacred/
```

### **Step 2**: Run the probing tasks on a model.

For example, download a Relation Extraction model trained with [RelEx](https://github.com/DFKI-NLP/RelEx), e.g., the [CNN](https://cloud.dfki.de/owncloud/index.php/s/F3gf9xkeb2foTFe/download) trained on SemEval.

```bash
mkdir -p models/cnn_semeval
wget --content-disposition https://cloud.dfki.de/owncloud/index.php/s/F3gf9xkeb2foTFe/download -P models/cnn_semeval
```

```bash
python probing_task_evaluation.py \
    --model-dir ./models/cnn_semeval/ \
    --data-dir ./data/semeval/ \
    --dataset semeval2010 \
    --cuda-device 0 \
    --batch-size 64 \
    --cache-representations
```

After the run is completed, the results are stored to `probing_task_results.json` in the `model-dir`.

```json
{
    "ArgTypeHead": {
        "acc": 75.82,
        "devacc": 78.96,
        "ndev": 670,
        "ntest": 2283
    },
    "ArgTypeTail": {
        "acc": 75.4,
        "devacc": 78.79,
        "ndev": 627,
        "ntest": 2130
    },
    [...]
}
```

## 📚&nbsp; Citation

If you use REval, please consider citing the following paper:
```
@inproceedings{alt-etal-2020-probing,
    title={Probing Linguistic Features of Sentence-level Representations in Neural Relation Extraction},
    author={Christoph Alt and Aleksandra Gabryszak and Leonhard Hennig},
    year={2020},
    booktitle={Proceedings of ACL},
    url={https://arxiv.org/abs/2004.08134}
}
```

## 📘&nbsp; License
REval is released under the terms of the [MIT License](LICENSE).
