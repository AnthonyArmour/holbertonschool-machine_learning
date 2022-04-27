#!/usr/bin/env python3
"""
Module is a script that fetches next launch
data from the spacex api.
"""


import requests


if __name__ == "__main__":

    base = "https://api.spacexdata.com/v4/"

    info = requests.get(base+"launches/next").json()

    date = info["date_local"]

    name = info["name"]

    launchpad_id = "launchpads/"+info["launchpad"]
    rocket_id = "rockets/"+info["rocket"]

    rocket_info = requests.get(base+rocket_id).json()

    launchpad_info = requests.get(base+launchpad_id).json()

    rocket = rocket_info["name"]
    launchpad = launchpad_info["name"]
    pad_location = launchpad_info["locality"]

    args = (name, date, rocket, launchpad, pad_location)
    result = "{} ({}) {} - {} ({})".format(*args)

    print(result)
