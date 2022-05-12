"""Contains function schools_by_topic."""

from pymongo import MongoClient


def schools_by_topic(mongo_collection, topic):
    """Gets list of schools with specific topic."""
    return mongo_collection.find({'topics': topic})
