from collections import deque
from datetime import datetime
import os
import time
import requests
import json

import pickle
from pathlib import Path

import traceback

appreviews_req = requests.get(f"https://store.steampowered.com/appreviews/10?json=1")
appreview = appreviews_req.json()
print(appreview['query_summary'].get('total_reviews'))