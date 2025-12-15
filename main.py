import argparse
from omegaconf import OmegaConf
from src.config import Config, get_default_config
from src.train_lightning import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a yaml config file"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load default config
    cfg = OmegaConf.structured(get_default_config())

    # Load yaml config if provided
    if args.config is not None:
        yaml_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    # Apply CLI overrides
    cli_cfg = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Convert back to dataclass
    cfg: Config = OmegaConf.to_object(cfg)

    print(cfg)
    # 🔽 pass cfg into training code
    train(cfg)


if __name__ == "__main__":
    main()
