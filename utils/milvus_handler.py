import sys

from pymilvus import FieldSchema, DataType, CollectionSchema, Collection, connections
from pymilvus.orm import utility

sys.path.append('../')
from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE
from logs import LOGGER


def create_collection(collection_name):
    if not utility.has_collection(collection_name):
        field1 = FieldSchema(name="id", dtype=DataType.INT6, is_primary=True)
        field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector",
                             dim=VECTOR_DIMENSION, is_primary=False)
        schema = CollectionSchema(fields=[field1, field2], description="collection description")
        return Collection(collection_name, schema=schema)
    return None


class MilvusHandler:
    def __init__(self, host, port, collection_name):
        # 连接
        connections.connect(host, port)

        # 如果存在就使用，不存在则创建
        if create_collection(collection_name) is None:
            self.collection = Collection(name=collection_name)
        else:
            self.collection = create_collection(collection_name)
            self.create_index(collection_name)

    def set_collection(self, collection_name):
        if utility.has_collection(collection_name):
            self.collection = Collection(name=collection_name)
        else:
            raise Exception(f"There has no collection named:{collection_name}")

    def create_index(self, collection_name):
        self.collection = self.set_collection(collection_name)
        default_index = {"index_type": "IVF_SQ8", "metric_type": METRIC_TYPE, "params": {"nlist": 16384}}
        status = self.collection.create_index(field_name="embedding", index_params=default_index)
        if not status.code:
            LOGGER.debug(
                f"Successfully create index in collection:{collection_name} with param:{default_index}")
            return status
        else:
            raise Exception(status.message)

    def insert(self, collection_name, vectors):
        try:
            self.collection = self.set_collection(collection_name)
            data = [vectors]    # data:二维数组
            milvus_result = self.collection.insert(data)
            ids = milvus_result.primary_keys

            self.collection.load()  # 当创建一个集合后，它默认处于未加载状态。通过调用collection.load()方法，你可以将集合加载到内存中，以便进行查询和其他操作
            LOGGER.debug(f"Insert vectors to Milvus in collection: {collection_name} with {len(vectors)} rows")
            return ids
        except Exception as e:
            LOGGER.error(f"Failed to insert data into Milvus: {e}")
