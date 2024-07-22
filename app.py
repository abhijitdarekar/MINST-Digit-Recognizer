import os
from dotenv import load_dotenv

load_dotenv()

try:
    print(os.environ['ARTIFACTS_FOLDER'])
except Exception as e:
    print(e)