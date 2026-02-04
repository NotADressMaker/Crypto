import argparse
import uvicorn
from cryptoquant.config import load_config
from cryptoquant.logging_config import configure_logging
from cryptoquant.serve.signal_api import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config.monitoring.logging_config)
    app = create_app(config)
    uvicorn.run(app, host=config.serve.host, port=config.serve.port)


if __name__ == "__main__":
    main()
