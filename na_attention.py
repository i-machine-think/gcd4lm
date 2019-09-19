from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions.attention import CDAttention
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.corpus import Corpus
from diagnnose.vocab import get_vocab_from_config

if __name__ == "__main__":
    arg_groups = {
        "model",
        "activations",
        "corpus",
        "init_states",
        "vocab",
        "decompose",
        "plot_attention",
    }
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args, arg_groups)

    model: LanguageModel = import_model(config_dict)
    corpus: Corpus = import_corpus(
        vocab_path=get_vocab_from_config(config_dict), **config_dict["corpus"]
    )

    for idx in range(0, len(corpus.examples), 2):
        corpus.examples[idx].sen += [corpus.examples[idx + 1].sen[-1]]

    if "fix_shapley" not in config_dict["decompose"]:
        config_dict["decompose"]["fix_shapley"] = False

    attention = CDAttention(
        model,
        corpus,
        cd_config=config_dict["decompose"],
        plot_config=config_dict["plot_attention"],
    )

    print("Creating plot for SP case...")
    sen = ["The", "N$_{sing}$", "PREP", "the", "N$_{plur}$", "V$_{sing}$", "V$_{plur}$"]
    attention.plot_config["xtext"] = sen[1:]
    attention.plot_config["ytext"] = sen[:-2]
    attention.plot_by_sen_id(
        slice(1200, 2400, 2),
        avg_decs=True,
        activations_dir=config_dict["activations"].get("activations_dir", None),
        extra_classes=[-2],
    )

    print("Creating plot for PS case...")
    sen = ["The", "N$_{plur}$", "PREP", "the", "N$_{sing}$", "V$_{plur}$", "V$_{sing}$"]
    attention.plot_config["xtext"] = sen[1:]
    attention.plot_config["ytext"] = sen[:-2]
    attention.plot_by_sen_id(
        slice(2400, 3600, 2),
        avg_decs=True,
        activations_dir=config_dict["activations"].get("activations_dir", None),
        extra_classes=[-2],
    )

    print("Creating example plot")
    attention.plot_by_sen_id(
        [1500],
        activations_dir=config_dict["activations"].get("activations_dir", None),
        extra_classes=[-2],
    )
