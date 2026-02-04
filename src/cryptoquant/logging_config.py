import logging.config
import yaml


def configure_logging(path: str) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    logging.config.dictConfig(config)
