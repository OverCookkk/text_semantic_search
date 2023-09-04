import numpy as np
import pandas as pd
from towhee import pipe, ops
from towhee.datacollection import DataCollection
import csv

from logs import LOGGER

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

milvus_host = "172.31.175.230"
milvus_port = '19530'
milvus_collection_name = 'search_article_in_medium'


def create_milvus_collection(collection_name, dim):
    try:
        if utility.has_collection(collection_name):
            # return Collection(name=collection_name)
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            # FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            # FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="reading_time", dtype=DataType.INT64),
            # FieldSchema(name="publication", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="claps", dtype=DataType.INT64),
            FieldSchema(name="responses", dtype=DataType.INT64),
            FieldSchema(name="title_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)

        ]
        schema = CollectionSchema(fields=fields, description='search text')
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            'metric_type': "L2",
            'index_type': "IVF_SQ8",
            'params': {"nlist": 2048}
        }
        collection.create_index(field_name='title_vector', index_params=index_params)
        LOGGER.debug(f"create new collection")
        return collection
    except Exception as e:
        LOGGER.error(f"Error create_milvus_collection {e}")


def sentence_encode(data):
    try:
        text_embedding = (pipe.input('text')
                          .map('text', 'embedding', ops.sentence_embedding.sbert(model_name='paraphrase-mpnet-base-v2'))
                          .output('text', 'embedding')
                          )
        # data = ['Hello, world.']
        res = text_embedding(data).get()
        # print(len(res))
        return res[1]
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")


def batch_sentence_encode(batch_data):
    try:
        list = []
        for i in range(0, len(batch_data), 2):
            list.extend(sentence_encode(batch_data[i:i + 2]))
            # print(f"sentence encode size :{len(list)}")
            LOGGER.debug(f"sentence encode size :{len(list)}")
            if i == 4:
                return list

        # 如果列表长度为奇数，最后一个元素单独处理
        if len(batch_data) % 2 != 0:
            list.append(sentence_encode(batch_data[-1]))
        return list
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")


def insert_milvus(data1, data2, data3, data4, data5):
    try:
        insert_pipe = (
            pipe.input('id', 'title', 'reading_time', 'claps', 'responses')
            .map('title', 'vec', ops.sentence_embedding.sbert(model_name='paraphrase-mpnet-base-v2'))
            .map('id', 'out_id', lambda x: int(x))
            .map('reading_time', 'out_reading_time', lambda x: int(x))
            .map('claps', 'out_claps', lambda x: int(x))
            .map('responses', 'out_responses', lambda x: int(x))
            .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
            .map(('out_id', 'out_reading_time', 'out_claps', 'out_responses', 'vec'), 'res',
                 ops.ann_insert.milvus_client(host='127.0.0.1', port='19530',
                                              collection_name=milvus_collection_name))
            .output('res', 'id', 'title', 'reading_time', 'claps', 'responses')
        )
        res = insert_pipe(data1, data2, data3, data4, data5)
        LOGGER.debug(f"insert_milvus success, res: {res.get()}")
    except Exception as e:
        LOGGER.error(f"Error insert_milvus {e}")


def search_in_milvus(data):
    try:
        """
        ops.ann_search.milvus_client 参数在D:\Anaconda3\envs\bootcamp\Lib\site-packages\pymilvus\orm\collection.py目录search_iterator函数下
        搜索带有某些表达式的文本， expr='title like "Python%"'，搜索Python开头的文本
        """
        search_pipe = (pipe.input('query')
                       .map('query', 'vec', ops.sentence_embedding.sbert(model_name='paraphrase-mpnet-base-v2'))
                       .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
                       .flat_map('vec', ('id', 'score', 'reading_time', 'claps', 'responses'),
                                 ops.ann_search.milvus_client(host='127.0.0.1',
                                                              port='19530',
                                                              collection_name=milvus_collection_name,
                                                              limit=5,
                                                              output_fields=['reading_time', 'claps', 'responses']))
                       .output('query', 'id', 'score', 'reading_time', 'claps', 'responses')
                       )
        # 批量搜索
        # res = search_pipe.batch(['a',  'b'])
        # for re in res:
        #     DataCollection(re).show()

        # 单个搜索
        res = search_pipe(data)
        DataCollection(res).show()
        LOGGER.debug(f"search_in_milvus success, res: {res.get()}")
    except Exception as e:
        LOGGER.error(f"Error search_in_milvus {e}")


if __name__ == '__main__':
    try:
        connections.connect(host=milvus_host, port=milvus_port)

        collection = create_milvus_collection(milvus_collection_name, 768)
        with open('./New_Medium_Data.csv', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                insert_milvus(row[0], row[1], row[4], row[6], row[7])
        collection.load()
        LOGGER.debug(f"milvus has {collection.num_entities} entities")

        search_in_milvus("funny python demo")
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")
