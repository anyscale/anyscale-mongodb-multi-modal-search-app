import ray
import pymongo
 
df = ray.data.read_json(
    "s3://anyscale-public-materials/mongodb-demo/data_with_ai_v3/"
).to_pandas()


try:
    client = pymongo.MongoClient(
        # os.environ["MONGODB_CONN_STR"],
        "mongodb+srv://sarieddinemarwan:yLbV9diLKku0ieIm@mongodb-anyscale-demo-m.epezhiv.mongodb.net/?retryWrites=true&w=majority&appName=mongodb-anyscale-demo-marwan",
    )
except pymongo.errors.ConfigurationError as e:
    raise ValueError(
        "An Invalid URI host error was received. "
        "Is your Atlas host name correct in your connection string?"
    ) from e 
    

db = client.myDatabase
my_collection = db["myntra-items"]

df["_id"] = df["name"]

documents = df[["_id", "name", "img", "price"]].to_dict(orient="records")

try:
    result = my_collection.insert_many(documents)
except pymongo.errors.OperationFailure as e:
    raise ValueError(
        "An authentication error was received. "
        "Are you sure your database user is authorized to perform write operations?"
    ) from e
else:
    inserted_count = len(result.inserted_ids)
    print("I inserted %x documents." % (inserted_count))
    print("\n")
