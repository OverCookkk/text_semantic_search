import sys

import pandas as pd

sys.path.append('../')
# from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE
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


def do_store(collection_name, file_dir, milvus_client, encode_model):
    # 先转embedding
    title_data, text_data, sentence_embeddings = extract_features(file_dir, encode_model)
    # 插入milvus
    ids = milvus_client.insert(collection_name, sentence_embeddings)
    # 插入数据库

    return len(ids)
