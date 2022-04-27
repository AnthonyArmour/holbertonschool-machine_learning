#!/usr/bin/env python3
"""
Module contains script for fetching
rocket launch history data from the
spacex api.
"""


import requests


if __name__ == "__main__":

    base = "https://api.spacexdata.com/v4/"

    launches = requests.get(base+"launches/past").json()

    rockets = {}

    for info in launches:
        if info["rocket"] in rockets.keys():
            rockets[info["rocket"]] += 1
        else:
            rockets[info["rocket"]] = 1

    for id in rockets.keys():
        rocket = requests.get(base+"rockets/"+id).json()
        key = rocket["name"]
        rockets[key] = rockets.pop(id)

    ordered = sorted(rockets.items(), key=lambda x: (-x[1], x[0]))

    for name, val in ordered:
        print("{}: {}".format(name, val))
