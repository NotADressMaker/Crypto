import argparse
from cryptoquant.config import load_config
from cryptoquant.datasets.builder import build_dataset
from cryptoquant.logging_config import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config.monitoring.logging_config)
    build_dataset(config)


if __name__ == "__main__":
    main()
