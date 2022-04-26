#!/usr/bin/env python3
"""
Module contains function for making
a request to the Swapi API to recieve
a list of home planets of sentient
species.
"""


import requests


def sentientPlanets():
    """
    Uses the Swapi API to recieve a list of home planets
    of sentient species.

    Return:
        List of home planets.
    """

    criteria_met, nextP = [], True

    info = requests.get("https://swapi-api.hbtn.io/api/species/")
    info = info.json()
    species = info["results"]

    while nextP:

        for obj in species:
            try:
                world = obj["homeworld"]
                if world is None:
                    pass
                    # criteria_met.append("unknown")
                else:
                    world_info = requests.get(world).json()
                    criteria_met.append(world_info["name"])
            except Exception:
                pass

        nextP = info["next"]
        if nextP:
            info = requests.get(nextP).json()
            species = info["results"]

    return criteria_met


if __name__ == "__main__":
    planets = sentientPlanets()

    for planet in planets:
        print(planet)
