import logging


def create_logger(logger_name: str, log_file: str) -> logging.Logger:
    """
    Creates a logger for logging to a file.

    Parameters
    ----------
    logger_name : str
        The name of the logger.
    log_file : str
        The path of the file to which write the logs.

    Returns
    -------
    logging.Logger
        The logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )
    file_handler = logging.FileHandler(
        log_file,
        mode='a'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
