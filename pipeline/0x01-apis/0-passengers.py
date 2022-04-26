#!/usr/bin/env python3
"""
Module contains function for making
a request to the Swapi API to recieve
a list of ships that hold a given
number of passengers.
"""


import requests


def availableShips(passengerCount):
    """
    Hits the Swapi API to get a list of ships that can hold
    a given number of passengers.

    Args:
        passengerCount: min number of passengers for search.

    Return:
        List of ship names, else empty list.
    """

    criteria_met, nextP = [], True

    info = requests.get("https://swapi-api.hbtn.io/api/starships/")
    info = info.json()
    ships = info["results"]

    while nextP:

        for obj in ships:
            try:
                obj["passengers"] = obj["passengers"].replace(",", "")
                if int(obj["passengers"]) >= passengerCount:
                    criteria_met.append(obj["name"])
            except Exception:
                pass

        nextP = info["next"]
        if nextP:
            info = requests.get(nextP).json()
            ships = info["results"]

    return criteria_met


if __name__ == "__main__":
    ships = availableShips(4)

    for ship in ships:
        print(ship)
