#!/usr/bin/env python3
"""
Module is a script that fetches next launch
data from the spacex api.
"""


import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    r = requests.get(url).json()

    dates = [launch['date_local'] for launch in r]
    next_launch_date = sorted(dates)[0]
    next_launch = r[dates.index(next_launch_date)]

    name = next_launch['name']
    date = next_launch['date_local']

    rocket = next_launch['rocket']
    rocket = requests.get(
        'https://api.spacexdata.com/v4/rockets/{}'.format(rocket)
    ).json()['name']

    launchpad = next_launch['launchpad']
    launchpad = requests.get(
        'https://api.spacexdata.com/v4/launchpads/{}'.format(launchpad)
    ).json()
    pad_name = launchpad['name']
    location = launchpad['locality']

    print('{} ({}) {} - {} ({})'.format(
        name, date, rocket, pad_name, location
    ))
