import logging
import os


def get_run_logger():
    """Return a logger that writes to console and run.log once per run."""
    logger = logging.getLogger("run")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        file_path = os.path.join(os.path.dirname(__file__), "run.log")
        file_handler = logging.FileHandler(file_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def log_run_parameters(logger):
    """Log key runtime parameters to the provided logger."""
    import config

    scanner = config.SCANNER
    expected_policy = "fifo"
    logger.info("Runtime parameters:")
    logger.info(
        "  max_concurrent_trades: %s",
        scanner.get("max_concurrent_trades"),
    )
    logger.info("  tie_break_policy: %s", scanner.get("tie_break_policy"))
    if scanner.get("tie_break_policy") != expected_policy:
        if config.warn_non_fifo:
            logger.warning(
                "tie_break_policy '%s' differs from expected live policy '%s'",
                scanner.get("tie_break_policy"),
                expected_policy,
            )
    logger.info("  monte_carlo_runs: %s", scanner.get("monte_carlo_runs"))
    logger.info("  seed: %s", scanner.get("seed"))
    logger.info(
        "  fees: %s | slippage: %s",
        getattr(config, "FEES", 0.0),
        getattr(config, "SLIPPAGE", 0.0),
    )
    logger.info("  penalty_weights: %s", config.ROBUSTNESS)
    logger.info("  data_resampling: %s", getattr(config, "RESAMPLE", {}))
