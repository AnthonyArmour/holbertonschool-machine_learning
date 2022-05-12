#!/usr/bin/env python3
"""Module that contains the function update_topics."""

from pymongo import MongoClient


def update_topics(mongo_collection, name, topics):
    """Changes all topics of school document based on name."""
    mongo_collection.update_many({'name': name},
                                 {'$set': {'name': name, 'topics': topics}})
