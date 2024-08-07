from typing import Dict, List
import inspect
from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.utils.handler.config import ConfigHandler, AttributeDict


def cinit(instance, config, **kwargs):
    """
    Instantiates a class by selecting the required args from a ConfigHandler. Omits wrong kargs
    @param instance:    class
    @param config:      ConfigHandler object contains class args
    @param kwargs:      kwargs besides/replacing ConfigHandler args
    @return:            class object
    """

    if isinstance(instance, type):
        instance_args = inspect.signature(instance.__init__)
        instance_keys = list(instance_args.parameters.keys())
        instance_keys.remove("self")
    else:
        instance_keys = inspect.getfullargspec(instance).args

    if isinstance(config, ConfigHandler) or isinstance(config, AttributeDict):
        config_dict = config.get_dict
    elif isinstance(config, Dict):
        config_dict = config
    elif isinstance(config, List):
        config_dict = {}
        for sub_conf in config:
            if isinstance(sub_conf, ConfigHandler) or isinstance(sub_conf, AttributeDict):
                config_dict.update(sub_conf.get_dict)
            elif isinstance(sub_conf, Dict):
                config_dict.update(sub_conf)
    else:
        raise UserWarning(
            f"cinit: Unknown config type. config must be Dict, AttributeDict or ConfigHandler but is {type(config)}")

    init_dict = {}

    for name, arg in kwargs.items():
        if name in instance_keys:
            init_dict[name] = arg

    for name, arg in config_dict.items():
        if name in instance_keys and name not in init_dict.keys():
            init_dict[name] = arg

    init_keys = list(init_dict.keys())
    missing_keys = list(set(instance_keys) - set(init_keys))
    if len(missing_keys) > 0:
        raise UserWarning(f"cinig: keys missing {missing_keys}")

    return instance(**init_dict)
