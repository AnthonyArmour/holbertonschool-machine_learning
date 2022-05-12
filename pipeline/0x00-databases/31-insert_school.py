#!/usr/bin/env python3
"""Module contains function insert_school."""

from pymongo import MongoClient


def insert_school(mongo_collection, **kwargs):
    """Inserts a new document in a collection based on kwargs."""
    return mongo_collection.insert(kwargs)
