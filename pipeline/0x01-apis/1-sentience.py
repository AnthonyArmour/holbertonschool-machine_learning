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

    criteria_met = []

    info = requests.get("https://swapi-api.hbtn.io/api/species/")
    info = info.json()
    species = info["results"]

    for obj in species:
        try:
            world = obj["homeworld"]
            if world is None:
                criteria_met.append("unknown")
            else:
                world_info = requests.get(world).json()
                criteria_met.append(world_info["name"])
        except Exception:
            pass

    return criteria_met


if __name__ == "__main__":
    planets = sentientPlanets()

    for planet in planets:
        print(planet)
