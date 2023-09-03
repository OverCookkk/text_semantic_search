import numpy as np
import pandas as pd
from towhee import pipe, ops
from towhee.datacollection import DataCollection

from logs import LOGGER

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

milvus_host = "172.31.175.230"
milvus_port = '19530'
milvus_collection_name = 'search_article_in_medium'




def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        LOGGER.debug(f"hzh ....................drop_collection")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        # FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        # FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="reading_time", dtype=DataType.INT64),
        # FieldSchema(name="publication", dtype=DataType.VARCHAR, max_length=500),
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
    return collection

connections.connect(host=milvus_host, port=milvus_port)
collection = create_milvus_collection(milvus_collection_name, 768)


def load_data(file_dir):
    try:
        data = pd.read_csv(file_dir)
        # d_list = data.values.tolist()
        # id = data['id'][0:6].tolist()
        # title = data['title'][0:6].tolist()
        # # link = data['link'][0:6].tolist()
        # reading_time = data['reading_time'][0:6].tolist()
        # # publication = data['publication'][0:6].tolist()
        # claps = data['claps'][0:6].tolist()
        # responses = data['responses'][0:6].tolist()
        # return [id, tile, link, reading_time, publication, claps, responses]
        return data
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")


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


def my_fun1(x):
    list = []
    for value in x:
        list.append(value[1])
    return list


def my_fun2(x, y):
    list = []
    for i in range(len(x)):
        data = []
        for j in range(len(x[i])):
            if j == 1:
                continue
            data.append(x[i][j])
        data.append(y[1][i].tolist())
        list.append(data)
    LOGGER.debug(f"list ========:{list}")
    return list

def my_fun3(x):
    # y = x.values.tolist()
    # LOGGER.debug(f"y ========:{y}")
    return int(x)

def insertMilvus(data):

    d = data.values.tolist()
    # r = my_fun2(d, [0.215, 0.5644])
    # d = [0, [0.215, 0.5644]]


    # insert_pipe = (pipe.input('d')
    #                .map('d', 'title', my_fun1)
    #                .map('title', 'embedding', text_embedding)
    #                .map(('d', 'embedding'), 'l', my_fun2)
    #                # .flat_map('df', 'e', my_fun3)
    #                .map('l', 'res', ops.ann_insert.milvus_client(host='127.0.0.1',
    #                                                               port='19530',
    #                                                               collection_name='search_article_in_medium'))
    #                .output('res')
    #                )
    insert_pipe = (
        pipe.input('id', 'title', 'reading_time', 'claps', 'responses')
        .map('title', 'vec', ops.sentence_embedding.sbert(model_name='paraphrase-mpnet-base-v2'))
        .map('id', 'g1', my_fun3)
        .map('reading_time', 'g2', my_fun3)
        .map('claps', 'g3', my_fun3)
        .map('responses', 'g4', my_fun3)
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
        .map(('g1', 'g2', 'g3', 'g4', 'vec'), 'insert_status',
             ops.ann_insert.milvus_client(host='127.0.0.1', port='19530', collection_name='search_article_in_medium'))
        .output()
    )

    import csv
    with open('./New_Medium_Data.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            insert_pipe(*row)

        # res = insert_pipe(d)
        # DataCollection(res).show()
        # LOGGER.debug(f"data = {data}")
        # LOGGER.debug(f"res = {res}")


if __name__ == '__main__':
    try:
        data_list = load_data("./New_Medium_Data.csv")
        # title_list = [row[1] for row in data_list[0:4]]
        # embeddings = batch_sentence_encode(title_list)
        #
        # m_list = []
        # i = 0
        # for d in data_list[0:4]:
        #     l = [d[0], d[4], d[6], d[7], d[7]]  # embeddings[i]
        #     i += 1
        #     m_list.append(l)

        insertMilvus(data_list)
        collection.load()
        LOGGER.debug(f"milvus has {collection.num_entities} entities")

    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")
