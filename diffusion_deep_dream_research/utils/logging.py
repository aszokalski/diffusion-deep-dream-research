import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Redirects standard logging messages to Loguru.
    """

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical("Uncaught exception")


def setup_distributed_logging(global_rank: int):
    logger.remove()

    sys.excepthook = handle_exception

    if global_rank == 0:
        logger.add(
            sys.stderr, level="INFO", format="<green>{time}</green> <level>{message}</level>"
        )
        logger.add("experiment.log", level="DEBUG", enqueue=True)
    else:
        logger.add(sys.stderr, level="CRITICAL")
        logger.add("experiment.log", level="ERROR", enqueue=True)

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
