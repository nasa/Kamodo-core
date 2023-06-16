import hydra

from kamodo import yaml_loader
import yaml
from omegaconf import OmegaConf
import sys
from os import path

def config_override(cfg):
    """Overrides with user-supplied configuration

    kamodo will override its configuration using
    kamodo.yaml if it is in the current working directory
    or users can set an override config:
        config_override=path/to/myconfig.yaml
    """
    override_path = hydra.utils.to_absolute_path(cfg.config_override)
    if path.exists(override_path):
        override_conf = OmegaConf.load(override_path)
        # merge overrides first input with second
        cfg = OmegaConf.merge(cfg, override_conf)
    return cfg


@hydra.main(config_path='conf/rpc.yaml', strict = False)
def main(cfg):
    
    # cfg = config_override(cfg)
    print(cfg)
    
    rpc_path = hydra.utils.to_absolute_path(cfg.rpc_conf)

    if not path.exists(rpc_path):
        print('{} does not exist... exiting'.format(rpc_path))
        sys.exit()

    k = yaml.load(open(rpc_path, 'rb'),
        Loader=yaml_loader())

    try:
        print('serving {}'.format(k.detail()))
        k.serve(host=cfg.host, port=cfg.port)
    except AttributeError as m:
        print(k)
        raise

# entrypoint for package installer
def entry():
    main()

if __name__ == "__main__":
    main()
