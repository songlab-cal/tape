# Tasks Assessing Protein Embeddings (TAPE)

![](https://github.com/songlab-cal/tape/workflows/Build/badge.svg)

Data, weights, and code for running the TAPE benchmark on a trained protein embedding. We provide a pretraining corpus, five supervised downstream tasks, pretrained language model weights, and benchmarking code. This code has been updated to use pytorch - as such previous pretrained model weights and code will not work. The previous tensorflow TAPE repository is still available at [https://github.com/songlab-cal/tape-neurips2019](https://github.com/songlab-cal/tape-neurips2019).

This repository is *not* an effort to maintain maximum compatibility and reproducability with the original paper, but is instead meant to facilitate ease of use and future development (both for us, and for the community). Although we provide much of the same functionality, we have not tested every aspect of training on all models/downstream tasks, and we have also made some deliberate changes. Therefore, if your goal is to reproduce the results from our paper, please use the original code.

Our paper is available at [https://arxiv.org/abs/1906.08230](https://arxiv.org/abs/1906.08230).

Some documentation is incomplete. We will try to fill it in over time, but if there is something you would like an explanation for, please open an issue so we know where to focus our effort!

## Contents

* [Installation](#installation)
* [Examples](#examples)
   * [Huggingface API for Loading Pretrained Models](#huggingface-api-for-loading-pretrained-models)
   * [Embedding Proteins with a Pretrained Model](#embedding-proteins-with-a-pretrained-model)
   * [Training a Model](#training-a-model)
   * [List of Models and Tasks](#list-of-models-and-tasks)
   * [Adding New Models and Tasks](#adding-new-models-and-tasks)
* [Data](#data)
   * [LMDB Data](#lmdb-data)
   * [Raw Data](#raw-data)
* [Leaderboard](#leaderboard)
    * [Secondary Structure](#secondary-structure)
    * [Contact Prediction](#contact-prediction)
    * [Remote Homology Detection](#remote-homology-detection)
    * [Fluorescence](#fluorescence)
    * [Stability](#stability)
* [Citation Guidelines](#citation-guidelines)

## Installation

We recommend that you install `tape` into a python [virtual environment](https://virtualenv.pypa.io/en/latest/) using

```bash
$ pip install tape_proteins
```

## Examples

### Huggingface API for Loading Pretrained Models

We build on the excellent [huggingface repository](https://github.com/huggingface/transformers) and use this as an API to define models, as well as to provide pretrained models. By using this API, pretrained models will be automatically downloaded when necessary and cached for future use.

```python
import torch
from tape import ProteinBertModel, TAPETokenizer
model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model

# Pfam Family: Hexapep, Clan: CL0536
sequence = 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ'
token_ids = torch.tensor([tokenizer.encode(sequence)])
output = model(token_ids)
sequence_output = output[0]
pooled_output = output[1]

# NOTE: pooled_output is *not* trained for the transformer, do not use
# w/o fine-tuning. A better option for now is to simply take a mean of
# the sequence output
```

Currently available pretrained models are:

* bert-base (Transformer model)
* babbler-1900 (UniRep model)

If there is a particular pretrained model that you would like to use, please open an issue and we will try to add it!

### Embedding Proteins with a Pretrained Model

Given an input fasta file, you can generate a `.npz` file containing embedding proteins via the `tape-embed` command.

Suppose this is our input fasta file:

```
>seq1
GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ
>seq2
RTIKVRILHAIGFEGGLMLLTIPMVAYAMDMTLFQAILLDLSMTTCILVYTFIFQWCYDILENR
```

Then we could embed it with the UniRep babbler-1900 model like so:

```bash
tape-embed unirep my_input.fasta output_filename.npz babbler-1900 --tokenizer unirep
```

There is no need to download the pretrained model manually - it will be automatically downloaded if needed. In addition, note the change of tokenizer to the `unirep` tokenizer. UniRep uses a different vocabulary, and so requires this tokenzer. If you get a cublas runtime error, please double check that you changed tokenizer correctly.

The embed function is fully batched and will automatically distribute across as many GPUs as the machine has available. On a Titan Xp, it can process around 200 sequences / second.

Once we have the output file, we can load it into numpy like so:

```python
arrays = np.load('output_filename.npz', allow_pickle=True)

list(arrays.keys())  # Will output the name of the keys in your fasta file (or if unnamed then '0', '1', ...)

arrays[<protein_id>]  # Returns a dictionary with keys 'pooled' and 'avg', (or 'seq' if using the --full_sequence_embed flag)
```

By default to save memory TAPE returns the average of the sequence embedding along with the pooled embedding generated through the pooling function. For some models (like UniRep), the pooled embedding is trained, and so can be used out of the box. For other models (like the transformer), the pooled embedding is not trained, and so the average embedding should be used. We will be looking into methods of self-supervised training the pooled embedding for all models in the future.

If you would like the full embedding rather than the average embedding, this can be specified to `tape-embed` by passing the `--full_sequence_embed` flag.

### Training a Model

Tape provides two commands for training, `tape-train` and `tape-train-distributed`. The first command uses standard pytorch data distribution to distributed across all available GPUs. The second one uses `torch.distributed.launch`-style multiprocessing to distributed across the number of specified GPUs (and could also be used for distributing across multiple nodes). We generally recommend using the second command, as it can provide a 10-15% speedup, but both will work.

To train the transformer on masked language modeling, for example, you could run this

```bash
tape-train-distributed transformer masked_language_modeling --batch_size BS --learning_rate LR --fp16 --warmup_steps WS --nproc_per_node NGPU --gradient-accumulation-steps NSTEPS
```

There are a number of features used in training:

    * Distributed training via multiprocessing
    * Half-precision training
    * Gradient accumulation
    * Gradient-allreduce post accumulation
    * Automatic batch by sequence length

The first feature you are likely to need is the `gradient_accumulation_steps`. TAPE specifies a relatively high batch size (1024) by default. This is the batch size that will be used *per backwards pass*. This number will be divided by the number of GPUs as well as the gradient accumulation steps. So with a batch size of 1024, 2 GPUs, and 1 gradient accumulation step, you will do 512 examples per GPU. If you run out of memory (and you likely will), TAPE provides a clear error message and will tell you to increase the gradient accumulation steps.

There are additional features as well that are not talked about here. See `tape-train-distributed --help` for a list of all commands.

### List of Models and Tasks

The available models are:

- `transformer` (pretrained available)
- `resnet`
- `lstm`
- `unirep` (pretrained available)
- `onehot` (no pretraining required)

The available standard tasks are:

- `language_modeling`
- `masked_language_modeling`
- `secondary_structure`
- `contact_prediction`
- `remote_homology`
- `fluorescence`
- `stability`

The available models and tasks can be found in `tape/datasets.py` and `tape/models/modeling*.py`.

### Adding New Models and Tasks

We have made some efforts to make the new repository easier to understand and extend. See the `examples` folder for an example on how to add a new model and a new task to TAPE. If there are other examples you would like or if there is something missing in the current examples, please open an issue.

## Data
Data should be placed in the `./data` folder, although you may also specify a different data directory if you wish.

The supervised data is around 120MB compressed and 2GB uncompressed.
The unsupervised Pfam dataset is around 7GB compressed and 19GB uncompressed. The data for training is hosted on AWS. By default we provide data as LMDB - see `tape/datasets.py` for examples on loading the data. If you wish to download all of TAPE, run `download_data.sh` to do so. We also provide links to each individual dataset below in both LMDB format and JSON format.

### LMDB Data

[Pretraining Corpus (Pfam)](http://s3.amazonaws.com/proteindata/data_pytorch/pfam.tar.gz) __|__ [Secondary Structure](http://s3.amazonaws.com/proteindata/data_pytorch/secondary_structure.tar.gz) __|__ [Contact (ProteinNet)](http://s3.amazonaws.com/proteindata/data_pytorch/proteinnet.tar.gz) __|__ [Remote Homology](http://s3.amazonaws.com/proteindata/data_pytorch/remote_homology.tar.gz) __|__ [Fluorescence](http://s3.amazonaws.com/proteindata/data_pytorch/fluorescence.tar.gz) __|__ [Stability](http://s3.amazonaws.com/proteindata/data_pytorch/stability.tar.gz)

### Raw Data

Raw data files are stored in JSON format for maximum portability. These are larger than the serialized TFRecord files (on average 3x larger). For all tasks except `proteinnet` we directly provide the output of our TFRecord parsing function on the file. For the `proteinnet` task we don't directly provide contact maps (as these massively increase the size of the files) and instead provide the 3D positions of all Carbon-alpha atoms. Note that this is in fact what is stored in the TFRecord files as well - our parsing function constructs the contact maps from this information on-the-fly.

[Pretraining Corpus (Pfam)](http://s3.amazonaws.com/proteindata/data_raw/pfam.tar.gz) __|__ [Secondary Structure](http://s3.amazonaws.com/proteindata/data_raw/secondary_structure.tar.gz) __|__ [Contact (ProteinNet)](http://s3.amazonaws.com/proteindata/data_raw/proteinnet.tar.gz) __|__ [Remote Homology](http://s3.amazonaws.com/proteindata/data_raw/remote_homology.tar.gz) __|__ [Fluorescence](http://s3.amazonaws.com/proteindata/data_raw/fluorescence.tar.gz) __|__ [Stability](http://s3.amazonaws.com/proteindata/data_raw/stability.tar.gz)


## Leaderboard

We will soon have a leaderboard available for tracking progress on the core five TAPE tasks, so check back for a link here. See the main tables in our paper for a sense of where performance stands at this point. Publication on the leaderboard will be contingent on meeting the following citation guidelines.

In the meantime, here's a temporary leaderboard for each task. All reported models on this leaderboard use unsupervised pretraining.

### Secondary Structure

| Ranking | Model | Accuracy (3-class) |
|:-:|:-:|:-:|
| 1. | One Hot + Alignment | 0.80 |
| 2. | LSTM | 0.75 |
| 2. | ResNet | 0.75 |
| 4. | Transformer | 0.73 |
| 4. | Bepler | 0.73 |
| 4. | Unirep | 0.73 |
| 7. | One Hot | 0.69 |

### Contact Prediction

| Ranking | Model | L/5 Medium + Long Range |
|:-:|:-:|:-:|
| 1. | One Hot + Alignment | 0.64 |
| 2. | Bepler | 0.40 |
| 3. | LSTM | 0.39 |
| 4. | Transformer | 0.36 |
| 5. | Unirep | 0.34 |
| 6. | ResNet | 0.29 |
| 6. | One Hot | 0.29 |

### Remote Homology Detection

| Ranking | Model | Top 1 Accuracy |
|:-:|:-:|:-:|
| 1. | LSTM | 0.26 |
| 2. | Unirep | 0.23 |
| 3. | Transformer | 0.21 |
| 4. | Bepler | 0.17 |
| 4. | ResNet | 0.17 |
| 6. | One Hot + Alignment | 0.09 |
| 6. | One Hot | 0.09 |

### Fluorescence

| Ranking | Model | Spearman's rho |
|:-:|:-:|:-:|
| 1. | Transformer | 0.68 |
| 2. | LSTM | 0.67 |
| 2. | Unirep | 0.67 |
| 4. | Bepler | 0.33 |
| 5. | ResNet | 0.21 |
| 6. | One Hot | 0.14 |

### Stability

| Ranking | Model | Spearman's rho |
|:-:|:-:|:-:|
| 1. | Transformer | 0.73 |
| 1. | Unirep | 0.73 |
| 1. | ResNet | 0.73 |
| 4. | LSTM | 0.69 |
| 5. | Bepler | 0.64 |
| 6. | One Hot | 0.19 |

## Citation Guidelines

If you find TAPE useful, please cite our corresponding paper. Additionally, __anyone using the datasets provided in TAPE must describe and cite all dataset components they use__. Producing these data is time and resource intensive, and we insist this be recognized by all TAPE users. For convenience,`data_refs.bib` contains all necessary citations. We also provide each individual citation below.

__TAPE (Our paper):__
```
@inproceedings{tape2019,
author = {Rao, Roshan and Bhattacharya, Nicholas and Thomas, Neil and Duan, Yan and Chen, Xi and Canny, John and Abbeel, Pieter and Song, Yun S},
title = {Evaluating Protein Transfer Learning with TAPE},
booktitle = {Advances in Neural Information Processing Systems}
year = {2019}
}
```

__Pfam (Pretraining):__
```
@article{pfam,
author = {El-Gebali, Sara and Mistry, Jaina and Bateman, Alex and Eddy, Sean R and Luciani, Aur{\'{e}}lien and Potter, Simon C and Qureshi, Matloob and Richardson, Lorna J and Salazar, Gustavo A and Smart, Alfredo and Sonnhammer, Erik L L and Hirsh, Layla and Paladin, Lisanna and Piovesan, Damiano and Tosatto, Silvio C E and Finn, Robert D},
doi = {10.1093/nar/gky995},
file = {::},
issn = {0305-1048},
journal = {Nucleic Acids Research},
keywords = {community,protein domains,tandem repeat sequences},
number = {D1},
pages = {D427--D432},
publisher = {Narnia},
title = {{The Pfam protein families database in 2019}},
url = {https://academic.oup.com/nar/article/47/D1/D427/5144153},
volume = {47},
year = {2019}
}
```
__SCOPe: (Remote Homology and Contact)__-
```
@article{scop,
  title={SCOPe: Structural Classification of Proteins—extended, integrating SCOP and ASTRAL data and classification of new structures},
  author={Fox, Naomi K and Brenner, Steven E and Chandonia, John-Marc},
  journal={Nucleic acids research},
  volume={42},
  number={D1},
  pages={D304--D309},
  year={2013},
  publisher={Oxford University Press}
}
```
__PDB: (Secondary Structure and Contact)__
```
@article{pdb,
  title={The protein data bank},
  author={Berman, Helen M and Westbrook, John and Feng, Zukang and Gilliland, Gary and Bhat, Talapady N and Weissig, Helge and Shindyalov, Ilya N and Bourne, Philip E},
  journal={Nucleic acids research},
  volume={28},
  number={1},
  pages={235--242},
  year={2000},
  publisher={Oxford University Press}
}
```

__CASP12: (Secondary Structure and Contact)__
```
@article{casp,
author = {Moult, John and Fidelis, Krzysztof and Kryshtafovych, Andriy and Schwede, Torsten and Tramontano, Anna},
doi = {10.1002/prot.25415},
issn = {08873585},
journal = {Proteins: Structure, Function, and Bioinformatics},
keywords = {CASP,community wide experiment,protein structure prediction},
pages = {7--15},
publisher = {John Wiley {\&} Sons, Ltd},
title = {{Critical assessment of methods of protein structure prediction (CASP)-Round XII}},
url = {http://doi.wiley.com/10.1002/prot.25415},
volume = {86},
year = {2018}
}
```

__NetSurfP2.0: (Secondary Structure)__
```
@article{netsurfp,
  title={NetSurfP-2.0: Improved prediction of protein structural features by integrated deep learning},
  author={Klausen, Michael Schantz and Jespersen, Martin Closter and Nielsen, Henrik and Jensen, Kamilla Kjaergaard and Jurtz, Vanessa Isabell and Soenderby, Casper Kaae and Sommer, Morten Otto Alexander and Winther, Ole and Nielsen, Morten and Petersen, Bent and others},
  journal={Proteins: Structure, Function, and Bioinformatics},
  year={2019},
  publisher={Wiley Online Library}
}
```

__ProteinNet: (Contact)__
```
@article{proteinnet,
  title={ProteinNet: a standardized data set for machine learning of protein structure},
  author={AlQuraishi, Mohammed},
  journal={arXiv preprint arXiv:1902.00249},
  year={2019}
}
```

__Fluorescence:__
```
@article{sarkisyan2016,
  title={Local fitness landscape of the green fluorescent protein},
  author={Sarkisyan, Karen S and Bolotin, Dmitry A and Meer, Margarita V and Usmanova, Dinara R and Mishin, Alexander S and Sharonov, George V and Ivankov, Dmitry N and Bozhanova, Nina G and Baranov, Mikhail S and Soylemez, Onuralp and others},
  journal={Nature},
  volume={533},
  number={7603},
  pages={397},
  year={2016},
  publisher={Nature Publishing Group}
}
```

__Stability:__
```
@article{rocklin2017,
  title={Global analysis of protein folding using massively parallel design, synthesis, and testing},
  author={Rocklin, Gabriel J and Chidyausiku, Tamuka M and Goreshnik, Inna and Ford, Alex and Houliston, Scott and Lemak, Alexander and Carter, Lauren and Ravichandran, Rashmi and Mulligan, Vikram K and Chevalier, Aaron and others},
  journal={Science},
  volume={357},
  number={6347},
  pages={168--175},
  year={2017},
  publisher={American Association for the Advancement of Science}
}
```
