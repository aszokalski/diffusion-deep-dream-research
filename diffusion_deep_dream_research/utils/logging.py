import sys
from loguru import logger


def setup_distributed_logging(global_rank: int):
    """
    Configures loguru to only log info/debug on the main process.
    Other processes will only log CRITICAL errors.
    """
    logger.remove()

    if global_rank == 0:
        logger.add(sys.stderr, level="INFO", format="<green>{time}</green> <level>{message}</level>")
        logger.add("experiment.log", level="DEBUG")

    else:
        logger.add(sys.stderr, level="CRITICAL")