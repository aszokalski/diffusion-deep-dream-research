import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Redirects standard logging messages to Loguru.
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Use .opt(exception=...) to safely serialize the traceback for multiprocessing
    logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical("Uncaught exception")


def setup_distributed_logging(global_rank: int):
    # 1. Clear existing Loguru handlers
    logger.remove()

    # 2. Hook sys.excepthook to capture crashes
    sys.excepthook = handle_exception

    # 3. Configure Loguru Sinks (File + Console)
    if global_rank == 0:
        logger.add(
            sys.stderr, level="INFO", format="<green>{time}</green> <level>{message}</level>"
        )
        logger.add("experiment.log", level="DEBUG", enqueue=True)
    else:
        logger.add(sys.stderr, level="CRITICAL")
        logger.add("experiment.log", level="ERROR", enqueue=True)

    # 4. Redirect Standard Logging to Loguru
    # This captures warnings/logs from other libraries (e.g., PyTorch, Transformers)
    # and forces them to go through the sinks defined above.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
