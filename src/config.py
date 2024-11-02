import os
import datetime

# Define the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define other directories relative to the root directory
LOG_DIR_ROOT = os.path.join(ROOT_DIR, 'logs')
RUN_NAME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

