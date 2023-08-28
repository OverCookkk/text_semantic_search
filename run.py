import os
from utils.milvus_handler import MilvusHandler
from fastapi import FastAPI, UploadFile, File
import uvicorn
from logs import LOGGER
from service.store import do_store

app = FastAPI()
milvus_client = MilvusHandler()


@app.post('/store_test')
async def store_test(file: UploadFile = File(...), collection_name: str = None):
    # 加载文件
    try:
        text = await file.read()
        file_name = file.filename
        dirs = "data"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        file_path = os.path.join(os.getcwd(), os.path.join(dirs, file_name))
        with open(file_path, 'wb') as f:
            f.write(text)
    except Exception:
        return {'status': -1, 'msg': 'Failed to load data.'}

    try:
        count = do_store(collection_name, milvus_client)
        LOGGER.info(f"Successfully loaded data, total count: {count}")
        return {'status': 0, 'msg': 'store data success.'}
    except Exception as e:
        LOGGER.error(e)
        return {'status': -1, 'msg': "store text data failed."}


if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=8000)
