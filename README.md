# Generalised Contextual Decomposition for Language Models

This repository contains the scripts to run the experiments described in the CoNLL 2019 paper _[Analysing Neural Language Models: Contextual Decomposition Reveals Default Reasoning in Number and Gender Assignment](https://arxiv.org/pdf/1909.08975.pdf)_.

The code is built upon the neural net analysis library [`diagnnose`](https://github.com/i-machine-think/diagnnose) that can be installed using `pip`.

The 4 scripts in this repo correspond to the 4 experiments that were performed in the paper.

Run `import_model.sh` first to import the language model of Gulordava et al. (2018). This script will download the model state dict and vocab file to the `model` directory.

After the model has been downloaded each script can be run as `python3 foo.py -c foo.json`.
