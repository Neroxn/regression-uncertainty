import logging

def pretty_dict_print(dictionary : dict):
    """
    Print a dictionary in a pretty way
    """
    print("--------------------")
    for key, value in dictionary.items():
        print('%s: %s' % (key, value))
    print("--------------------")

def set_logger():
    """
    Set the logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create error file handler and set level to error
    fh = logging.FileHandler("error.log")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create debug file handler and set level to debug
    fh = logging.FileHandler("debug.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger