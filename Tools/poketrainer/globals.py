import os
from datetime import datetime
import platform
from pathlib import Path

ROOT_PATH = Path.cwd().parent
time = datetime.now()
DATE_TIME = time.strftime('%Y%m%d%H')
INSTALLATION_INFO = platform.machine() + '_' + platform.system()
LOG_ROOT_PATH = str(ROOT_PATH) + '/logs/{}_{}'.format(DATE_TIME, INSTALLATION_INFO)
