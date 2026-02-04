import argparse
from cryptoquant.config import load_config
from cryptoquant.logging_config import configure_logging
from cryptoquant.models.router import train_router


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config.monitoring.logging_config)
    train_router(config)


if __name__ == "__main__":
    main()
