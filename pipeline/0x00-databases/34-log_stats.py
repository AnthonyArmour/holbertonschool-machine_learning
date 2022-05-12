#!/usr/bin/env python3
"""Provides stats about Nginx logs stored in MongoDB."""


from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx
    doc_count = logs.count_documents({})
    print("{} logs".format(doc_count))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for m in methods:
        cnt = logs.count_documents({"method": m})
        print("\tmethod {}: {}".format(m, cnt))
    ftp = {"method": "GET", "path": "/status"}
    cnt = logs.count_documents(ftp)
    print("{} status check".format(cnt))
