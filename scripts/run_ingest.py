import argparse
from cryptoquant.config import load_config
from cryptoquant.data.ingest import ingest_all
from cryptoquant.logging_config import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config.monitoring.logging_config)
    ingest_all(config)


if __name__ == "__main__":
    main()
