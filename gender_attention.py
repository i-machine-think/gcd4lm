import os
import shutil
from typing import Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions import DecomposerFactory
from diagnnose.downstream.winobias import create_winobias_classes
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.corpus import Corpus
from diagnnose.utils.midpoint import MidPointNorm
from diagnnose.vocab import get_vocab_from_config

TMP_DIR = "tmp"
plt.rcParams["figure.figsize"] = 8, 8


def calc_diff_scores(config, lm: LanguageModel) -> Dict[str, Dict[str, Tensor]]:
    scores = {}

    for corpus_type in ["unamb", "stereo"]:
        scores[corpus_type] = {}
        for condition in ["FM", "MF"]:
            corpus_name = f"{corpus_type}_{condition}.txt"
            corpus_path = os.path.join(config["corpus"]["path"], corpus_name)
            corpus: Corpus = import_corpus(
                corpus_path,
                vocab_path=get_vocab_from_config(config),
                header_from_first_line=True,
            )

            if config["activations"].get("activations_dir", None) is not None:
                activations_dir = os.path.join(
                    config["activations"]["activations_dir"],
                    corpus_type,
                    condition.lower(),
                )
            else:
                activations_dir = None

            sen_ids = slice(0, len(corpus))
            factory = DecomposerFactory(
                lm,
                activations_dir or TMP_DIR,
                create_new_activations=(activations_dir is None),
                corpus=corpus,
                sen_ids=sen_ids,
            )

            ref_types = [ex.ref_type for ex in corpus.examples]
            classes = create_winobias_classes(ref_types, corpus)

            decomposer = factory.create(sen_ids)
            lens = decomposer.final_index - 1

            final_hidden = decomposer.activation_dict[decomposer.toplayer, "hx"][
                range(len(corpus)), lens + 1
            ].unsqueeze(2)
            full_probs = torch.bmm(lm.decoder_w[classes], final_hidden).squeeze()
            full_probs += lm.decoder_b[classes]

            obj_idx_start = torch.tensor([ex.obj_idx_start - 1 for ex in corpus])
            obj_idx_end = torch.tensor([ex.obj_idx + 1 for ex in corpus])
            ranges = [
                (0, 2),
                (2, obj_idx_start),
                (obj_idx_start, obj_idx_end),
                (obj_idx_end, lens + 1),
            ]

            scores[corpus_type][condition] = torch.zeros(4)

            for i, (start, stop) in enumerate(ranges):
                config["decompose"].update({"start": start, "stop": stop})
                rel_dec = decomposer.decompose(**config["decompose"])["relevant"]
                final_rel_dec = rel_dec[range(len(corpus)), lens].unsqueeze(2)
                rel_probs = torch.bmm(lm.decoder_w[classes], final_rel_dec).squeeze()
                rel_probs /= full_probs

                prob_diffs = rel_probs[:, 0] - rel_probs[:, 1]
                scores[corpus_type][condition][i] = torch.mean(prob_diffs)
            print(corpus_type, condition, scores[corpus_type][condition])

    return scores


def plot_diff(scores: Dict[str, Dict[str, Tensor]]) -> None:
    cmin, cmax = -0.14, 0.12

    for corpus_type in scores:
        fig, axs = plt.subplots(1, 2)
        fig.set_facecolor("w")
        for n, (condition, score_arr) in enumerate(scores[corpus_type].items()):
            score_arr = score_arr.unsqueeze(1).numpy()
            axs[n].imshow(
                score_arr,
                cmap="PiYG",
                norm=MidPointNorm(vmin=cmin, vmax=cmax, midpoint=0),
            )
            axs[n].set_xticks(range(0))
            axs[n].set_yticks(range(4))
            axs[n].set_yticklabels(
                [f"subj$_{condition[0]}$", "[...]", f"obj$_{condition[1]}$", "[...]"],
                fontsize=35,
            )

            axs[n].set_title(condition, fontsize=35, weight="bold")

            for (j, i), label in np.ndenumerate(score_arr):
                # beta = np.round(label, 3)
                if (cmin / 1.5) < label < (cmax / 1.3):
                    axs[n].text(
                        i,
                        j,
                        f"{label:.3f}",
                        ha="center",
                        va="center",
                        fontsize=30,
                        color="black",
                    )
                else:
                    axs[n].text(
                        i,
                        j,
                        f"{label:.3f}",
                        ha="center",
                        va="center",
                        fontsize=30,
                        color="white",
                    )

            for p in range(4):
                axs[n].add_patch(
                    patches.Rectangle(
                        (-0.48, -0.48),
                        0.97,
                        0.97 + p,
                        linewidth=4 if p == 3 else 2,
                        edgecolor="black",
                        facecolor="none",
                    )
                )
        plt.show()


if __name__ == "__main__":
    arg_groups = {"model", "activations", "corpus", "init_states", "vocab", "decompose"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args, arg_groups)

    model: LanguageModel = import_model(config_dict)

    if "fix_shapley" not in config_dict["decompose"]:
        config_dict["decompose"]["fix_shapley"] = False

    diff_scores = calc_diff_scores(config_dict, model)
    plot_diff(diff_scores)

    if config_dict["activations"].get("activations_dir", None) is None:
        shutil.rmtree(TMP_DIR)
