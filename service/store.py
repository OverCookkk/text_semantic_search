import sys

import pandas as pd

sys.path.append('../')
from config import DEFAULT_TABLE
from logs import LOGGER


def extract_features(file_dir, encode_model):
    try:
        data = pd.read_csv(file_dir)
        title_data = data['title'].tolist()
        text_data = data['text'].tolist()
        sentence_embeddings = encode_model.sentence_encode(title_data)
        return title_data, text_data, sentence_embeddings
    except Exception as e:
        LOGGER.error(f" Error with extracting feature from question {e}")


def format_data(ids, title_data, text_data):
    data = []
    for i in range(len(ids)):
        value = str(ids[i]) + title_data[i] + text_data[i]
        data.append(value)
    return data


def do_store(collection_name, file_dir, milvus_client, mysql_client, encode_model):
    if not collection_name:
        collection_name = DEFAULT_TABLE

    # embedding
    title_data, text_data, sentence_embeddings = extract_features(file_dir, encode_model)
    # 插入milvus
    ids = milvus_client.insert(collection_name, sentence_embeddings)
    # 插入数据库
    mysql_client.create_mysql_table(collection_name)
    mysql_client.insert_data_to_mysql(collection_name, format_data(ids, title_data, text_data))

    return len(ids)
