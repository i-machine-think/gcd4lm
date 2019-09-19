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

    suite = DownstreamSuite(
        config_dict["downstream"]["config"],
        vocab_path,
        device=config_dict["model"].get("device", "cpu"),
        print_results=True,
    )

    config_dict["decompose"]["fix_shapley"] = False
    ignore_unk = False

    print("\nFull model", end="")
    suite.run(model)
    print("\nWithout dec. bias:")
    suite.run(model, add_dec_bias=False)

    print("\nSubject focus", end="")
    suite.decompose_config = {"start": 0, "stop": 2, **config_dict["decompose"]}
    suite.run(model)
    print("\nWithout dec. bias:")
    suite.run(model, add_dec_bias=False)

    print("\nObject focus", end="")
    suite.run(model, ignore_unk=ignore_unk, decomp_obj_idx=True)
    print("\nWithout dec. bias:")
    suite.run(model, add_dec_bias=False, ignore_unk=ignore_unk, decomp_obj_idx=True)

    print("\nIntercept only", end="")
    suite.decompose_config.update(
        {
            "start": -1,
            "stop": 10,
            "decomp_obj_idx": False,
            "rel_interactions": ["rel-rel", "rel-b", "b-b", "rel-irrel"],
            "only_source_rel": True,
            "input_never_rel": True,
            "init_states_rel": True,
            "use_extracted_activations": False,
        }
    )
    suite.run(model)
    print("\nWithout decoder bias:")
    suite.run(model, add_dec_bias=False)
