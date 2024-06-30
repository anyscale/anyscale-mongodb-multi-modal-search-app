import pymongo
import os
import ray
import pandas as pd
import pyarrow as pa
import numpy as np
from pyarrow import csv
from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne
from pymongo.operations import SearchIndexModel, IndexModel


def read_data(path: str, nsamples: int) -> ray.data.Dataset:
    ds = ray.data.read_csv(
        path,
        parse_options=csv.ParseOptions(newlines_in_values=True),
        convert_options=csv.ConvertOptions(
            column_types={
                "id": pa.int64(),
                "name": pa.string(),
                "img": pa.string(),
                "asin": pa.string(),
                "price": pa.float64(),
                "mrp": pa.float64(),
                "rating": pa.float64(),
                "ratingTotal": pa.int64(),
                "discount": pa.int64(),
                "seller": pa.string(),
                "purl": pa.string(),
            }
        ),
        override_num_blocks=nsamples,
    )
    return ds

# see https://www.mongodb.com/docs/atlas/atlas-search/limitations/
SMALL_CLUSTER_SIZES = ["m0", "m2", "m5"]

def setup_collection(db_name, collection_name, cluster_size: str = "m0") -> None:
    """
    Create the collection with the necessary indices for the search:
    - A lexical search index on the "name" field with a standard lucene analyzer
    - A vector search index on both name and description embedding fields
    - Single field indices on the rest of the search fields with likely sorting order
    """
    mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
    db = mongo_client[db_name]
    db.drop_collection(collection_name)
    my_collection = db[collection_name]

    my_collection.create_indexes(
        [
            IndexModel([("rating", DESCENDING)]),
            IndexModel([("category", ASCENDING)]),
            IndexModel([("season", ASCENDING)]),
            IndexModel([("color", ASCENDING)]),
        ]
    )

    if cluster_size in SMALL_CLUSTER_SIZES:
        print(
            "For small cluster sizes, you need to use the UI to create the search indices"
        )
        return

    my_collection.create_search_index(
        SearchIndexModel(
            {
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "name": {
                                "type": "string",
                                "analyzer": "lucene.standard",
                            },
                        },
                    }
                },
                "name": "lexical_text_search_index",
            }
        )
    )

    my_collection.create_search_index(
        SearchIndexModel(
            {
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": [
                            {
                                "numDimensions": 1024,
                                "similarity": "cosine",
                                "type": "vector",
                                "path": "description_embedding",
                            },
                            {
                                "type": "filter",
                                "path": "category",
                            },
                            {
                                "type": "filter",
                                "path": "season",
                            },
                            {
                                "type": "filter",
                                "path": "color",
                            },
                            {
                                "type": "filter",
                                "path": "rating",
                            },
                            {
                                "type": "filter",
                                "path": "price",
                            },
                        ],
                    }
                },
                "name": "vector_search_index",
            }
        )
    )


def clear_data_in_collection(db_name: str, collection_name: str):
    mongo_client: pymongo.MongoClient = pymongo.MongoClient(
        os.environ["DB_CONNECTION_STRING"],
    )
    db = mongo_client[db_name]
    my_collection = db[collection_name]
    my_collection.delete_many({})


class MongoBulkUpdate:
    def __init__(self, db: str, collection: str) -> None:
        client = MongoClient(os.environ["DB_CONNECTION_STRING"])
        self.collection = client[db][collection]

    def __call__(self, batch_df: pd.DataFrame) -> dict[str, np.ndarray]:
        docs = batch_df.to_dict(orient="records")
        bulk_ops = [
            UpdateOne(filter={"_id": doc["_id"]}, update={"$set": doc}, upsert=True)
            for doc in docs
        ]
        self.collection.bulk_write(bulk_ops)
        return {}


class MongoBulkInsert:
    def __init__(self, db: str, collection: str) -> None:
        client = MongoClient(os.environ["DB_CONNECTION_STRING"])
        self.collection = client[db][collection]

    def __call__(self, batch_df: pd.DataFrame) -> dict[str, np.ndarray]:
        docs = batch_df.to_dict(orient="records")
        self.collection.insert_many(docs)
        return {}
