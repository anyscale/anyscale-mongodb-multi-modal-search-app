from typing import Optional
import openai
from motor.motor_asyncio import AsyncIOMotorClient
import os
from ray.serve import deployment, ingress
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI


@deployment
class EmbeddingModel:
    def __init__(self, model: str = "thenlper/gte-large") -> None:
        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=os.environ["ANYSCALE_API_KEY"],
        )
        self.model = model

    async def compute_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding


@deployment
class QueryLegacySearch:
    def __init__(
        self,
        database_name: str = "myntra",
        collection_name: str = "myntra-items",
    ) -> None:
        self.client = AsyncIOMotorClient(os.environ["DB_CONNECTION_STRING"])
        self.database_name = database_name
        self.collection_name = collection_name

    async def run(
        self,
        text_search: Optional[str],
        min_price: int,
        max_price: int,
        min_rating: float,
        n: int = 20,
    ) -> list[tuple[str, str]]:
        db = self.client[self.database_name]
        collection = db[self.collection_name]

        pipeline = []
        if text_search.strip():
            pipeline.append(
                {
                    "$search": {
                        "index": "name-index-search",
                        "text": {
                            "query": text_search,
                            "path": "name",
                        },
                    }
                }
            )
        pipeline.extend(
            [
                {
                    "$match": {
                        "price": {"$gte": min_price, "$lte": max_price},
                        "rating": {"$gte": min_rating},
                    }
                },
                {
                    "$limit": n,
                },
            ]
        )
        records = collection.aggregate(pipeline)
        results = [
            (record["img"].split(";")[-1].strip(), record["name"])
            async for record in records
        ]
        return results


@deployment
class QueryWithVectorSearch:
    def __init__(
        self,
        embedding_model: DeploymentHandle,
        database_name: str = "myntra",
        collection_name: str = "myntra-items",
    ) -> None:
        self.client = AsyncIOMotorClient(os.environ["DB_CONNECTION_STRING"])
        self.embedding_model = embedding_model
        self.database_name = database_name
        self.collection_name = collection_name

    async def run(
        self,
        text_search: str,
        min_price: int,
        max_price: int,
        min_rating: float,
        synthetic_categories: list[str],
        colors: list[str],
        seasons: list[str],
        n: int = 20,
        vector_search_index_name: str = "vector_index",
        vector_search_path: str = "description_embedding",
        text_search_index_name: str = "name-index-search",
        vector_penalty: int = 1,
        full_text_penalty: int = 10,
    ):
        db = self.client[self.database_name]
        collection = db[self.collection_name]

        if text_search.strip():
            embedding = await self.embedding_model.compute_embedding.remote(text_search)
            records = collection.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": vector_search_index_name,
                            "path": vector_search_path,
                            "queryVector": embedding,
                            "numCandidates": 100,
                            "limit": 20,
                            "filter": {
                                "price": {"$gte": min_price, "$lte": max_price},
                                "rating": {"$gte": min_rating},
                                "category": {"$in": synthetic_categories},
                                "color": {"$in": colors},
                                "season": {"$in": seasons},
                            },
                        }
                    },
                    {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                    {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                    {
                        "$addFields": {
                            "vs_score": {
                                "$divide": [1.0, {"$add": ["$rank", vector_penalty, 1]}]
                            }
                        }
                    },
                    {
                        "$project": {
                            "vs_score": 1,
                            "_id": "$docs._id",
                            "name": "$docs.name",
                            "img": "$docs.img",
                        }
                    },
                    {
                        "$unionWith": {
                            "coll": "myntra-items",
                            "pipeline": [
                                {
                                    "$search": {
                                        "index": text_search_index_name,
                                        "text": {
                                            "query": text_search,
                                            "path": "name",
                                        },
                                        # "phrase": {"query": text_search, "path": "name"},
                                    }
                                },
                                {
                                    # TODO - implement $match using pre-filters instead of a post-$search step
                                    "$match": {
                                        "price": {"$gte": min_price, "$lte": max_price},
                                        "rating": {"$gte": min_rating},
                                        "category": {"$in": synthetic_categories},
                                        "color": {"$in": colors},
                                        "season": {"$in": seasons},
                                    }
                                },
                                {"$limit": 20},
                                {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                                {
                                    "$unwind": {
                                        "path": "$docs",
                                        "includeArrayIndex": "rank",
                                    }
                                },
                                {
                                    "$addFields": {
                                        "fts_score": {
                                            "$divide": [
                                                1.0,
                                                {
                                                    "$add": [
                                                        "$rank",
                                                        full_text_penalty,
                                                        1,
                                                    ]
                                                },
                                            ]
                                        }
                                    }
                                },
                                {
                                    "$project": {
                                        "fts_score": 1,
                                        "_id": "$docs._id",
                                        "name": "$docs.name",
                                        "img": "$docs.img",
                                    }
                                },
                            ],
                        }
                    },
                    {
                        "$group": {
                            "_id": "$name",
                            "img": {"$first": "$img"},
                            "vs_score": {"$max": "$vs_score"},
                            "fts_score": {"$max": "$fts_score"},
                        }
                    },
                    {
                        "$project": {
                            "_id": 1,
                            "img": 1,
                            "vs_score": {"$ifNull": ["$vs_score", 0]},
                            "fts_score": {"$ifNull": ["$fts_score", 0]},
                        }
                    },
                    {
                        "$project": {
                            "score": {"$add": ["$fts_score", "$vs_score"]},
                            "name": "$_id",
                            "img": 1,
                            "vs_score": 1,
                            "fts_score": 1,
                        }
                    },
                    {"$sort": {"score": -1}},
                    {"$limit": n},
                ]
            )
        else:
            records = collection.aggregate(
                [
                    {
                        "$match": {
                            "price": {"$gte": min_price, "$lte": max_price},
                            "rating": {"$gte": min_rating},
                            "category": {"$in": synthetic_categories},
                            "color": {"$in": colors},
                            "season": {"$in": seasons},
                        }
                    },
                    {"$limit": n},
                ]
            )

        return [
            (record["img"].split(";")[-1].strip(), record["name"])
            async for record in records
        ]


fastapi = FastAPI()


@deployment
@ingress(fastapi)
class QueryApplication:

    def __init__(
        self,
        query_legacy: QueryLegacySearch,
        query_with_vector_search: QueryWithVectorSearch,
    ):
        self.query_legacy = query_legacy
        self.query_with_vector_search = query_with_vector_search

    @fastapi.get("/legacy")
    async def query_legacy_search(
        self,
        text_search: str,
        min_price: int,
        max_price: int,
        min_rating: float,
        num_results: int,
    ):
        return await self.query_legacy.run.remote(
            text_search=text_search,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            n=num_results,
        )

    @fastapi.get("/vector")
    async def query_vector_search(
        self,
        text_search: str,
        min_price: int,
        max_price: int,
        min_rating: float,
        synthetic_categories: list[str],
        colors: list[str],
        seasons: list[str],
        num_results: int,
    ):
        return await self.query_with_vector_search.run.remote(
            text_search=text_search,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            synthetic_categories=synthetic_categories,
            colors=colors,
            seasons=seasons,
            n=num_results,
        )


query_legacy = QueryLegacySearch.bind()
embedding_model = EmbeddingModel.bind()
query_with_vector_search = QueryWithVectorSearch.bind(embedding_model)
app = QueryApplication.bind(query_legacy, query_with_vector_search)
