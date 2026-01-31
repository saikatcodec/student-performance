import logging
import os
from datetime import datetime

LOG_FILE_PATH = f'{datetime.now().strftime('%d-%m-%Y_%H_%M')}'
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE_PATH)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE = os.path.join(logs_path, LOG_FILE_PATH + '.log')

logging.basicConfig(
    format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p',
    level=logging.INFO,
    filename=LOG_FILE
)
