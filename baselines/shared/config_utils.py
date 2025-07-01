import os
import yaml
from typing import Dict, Any
from types import SimpleNamespace
import ruamel.yaml
import embodied

def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace recursively."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def load_config(argv=None, config_path=None):
    """Load configuration from YAML file with support for named configs."""
    if config_path is None:
        # Default to config.yaml in the current working directory
        config_path = "config.yaml"
    
    configs = ruamel.yaml.YAML(typ='safe').load(
        embodied.Path(config_path).read())
    
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config_dict = embodied.Config(configs['defaults'])

    for name in parsed.configs:
        config_dict = config_dict.update(configs[name])
    config_dict = embodied.Flags(config_dict).parse(other)
    
    # Convert to SimpleNamespace
    config = dict_to_namespace(config_dict)
    print(config)

    return config
