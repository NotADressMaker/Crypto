import argparse
from cryptoquant.config import load_config
from cryptoquant.logging_config import configure_logging
from cryptoquant.personas import PERSONAS, run_persona, write_persona_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run persona-specific pipeline.")
    parser.add_argument("persona", choices=sorted(PERSONAS.keys()))
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config.monitoring.logging_config)
    report = run_persona(config, args.persona)
    path = write_persona_report(config, args.persona, report)
    print(f"Wrote persona report to {path}")


if __name__ == "__main__":
    main()
