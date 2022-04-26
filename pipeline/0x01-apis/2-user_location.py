#!/usr/bin/env python3
"""
Fetches location of a specific github user
via the GitHub API.
"""


import sys
import requests
from time import time

if __name__ == "__main__":

    if len(sys.argv) < 2:
        exit()

    path = sys.argv[1]

    info = requests.get(path)

    if info.status_code == 403:
        tm = int(info.headers["X-Ratelimit-Reset"])
        mins = round((tm - time()) / 60)
        print("Reset in {} min".format(mins))
    elif info.status_code == 404:
        print("Not found")
    else:
        loc = info.json()["location"]
        if loc:
            print(loc)
        else:
            print("Not found")
