import logging

# Disable asyncio logging
# https://stackoverflow.com/questions/60503705/how-can-i-avoid-using-selector-epollselector-log-message-in-django
#logging.getLogger('asyncio').setLevel(logging.WARNING)
#logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def set_logging_level(level):
    """Set the logging level.

    Parameters
    ----------
    level : int
        The logging level. For example, logging.INFO.

    Returns
    -------
    None
    """
    logging.getLogger().setLevel(level)


def get_logging_level():
    """Get the logging level.

    Returns
    -------
    level : str
        The name of the logging level. For example, 'INFO'.
    """
    return logging.getLevelName(logging.getLogger().getEffectiveLevel())
