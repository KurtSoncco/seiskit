import logging


def setup_basic_logger():
    """
    Sets up a basic root logger that prints to the console.
    """
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum level of messages to log
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Get a logger instance, it's conventional to use the module's name
    logger = logging.getLogger(__name__)
    return logger
