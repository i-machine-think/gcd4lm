import shutil

from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.downstream.suite import DownstreamSuite
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel
from diagnnose.vocab import get_vocab_from_config

if __name__ == "__main__":
    arg_groups = {"model", "decompose", "init_states", "vocab", "downstream"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args, arg_groups)

    model: LanguageModel = import_model(config_dict)

    vocab_path = get_vocab_from_config(config_dict)
    assert vocab_path is not None, "vocab_path should be provided"

    if "fix_shapley" not in config_dict["decompose"]:
        config_dict["decompose"]["fix_shapley"] = False

    suite = DownstreamSuite(
        config_dict["downstream"]["config"],
        vocab_path,
        decompose_config=config_dict["decompose"],
        device=config_dict["model"].get("device", "cpu"),
        print_results=True,
    )

    print("\nIN only", end="")
    suite.run(model)
    print("Without dec. bias:")
    suite.run(model, add_dec_bias=False)

    print("\nNo intercept", end="")
    suite.decompose_config.update(
        {
            "rel_interactions": ["rel-rel", "rel-irrel"],
            "bias_bias_only_in_phrase": False,
        }
    )
    suite.run(model)
    print("Without dec. bias:")
    suite.run(model, add_dec_bias=False)

    print("\nIntercept only", end="")
    suite.decompose_config.update(
        {
            "start": -1,
            "stop": 6,
            "rel_interactions": ["rel-rel", "rel-b", "b-b", "rel-irrel"],
            "input_never_rel": True,
            "only_source_rel": True,
            "use_extracted_activations": False,
        }
    )
    suite.run(model)
    print("Without dec. bias:")
    suite.run(model, add_dec_bias=False)

    shutil.rmtree("lakretz_activations")
