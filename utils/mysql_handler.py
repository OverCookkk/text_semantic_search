import sys

import pymysql

sys.path.append('../')
# from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE
from logs import LOGGER


class MySQLHandler():
    def __init__(self, host, port, username, password, database):
        self.conn = pymysql.connect(host=host, port=port, user=username, password=password, local_infile=True)
        self.curor = self.conn.cursor()
        try:
            self.conn.ping()
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with mysql host:{host}, port:{port}")
            sys.exit(1)

    def create_mysql_table(self, table_name):
        # Create mysql table if not exists
        sql = "create table if not exists " + table_name + "(milvus_id TEXT, title TEXT ,text TEXT);"
        try:
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL create table: {table_name} with sql: {sql}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def insert_data_to_mysql(self, table_name, data):
        sql = "insert into " + table_name + " ((milvus_id,title,text) values (%s,%s,%s);"
        try:
            self.cursor.executemany(sql, data)  # 执行批量插入
            self.conn.commit()
            LOGGER.debug(f"MYSQL loads data to table: {table_name} successfully")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            # sys.exit(1)
