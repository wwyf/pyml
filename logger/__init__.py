import logging

# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

fl = logging.FileHandler('temp.log')
fl.setLevel(logging.DEBUG)

# create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n%(filename)s - %(module)s - %(funcName)s - %(lineno)d \n%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
formatter = logging.Formatter('[%(levelname)8s] - [%(module)10s] - [%(lineno)3d] - [%(funcName)10s] - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)
fl.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fl)
