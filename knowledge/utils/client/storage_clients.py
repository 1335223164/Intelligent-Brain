import threading
import json
from typing import Optional
from typing import TypeVar, Optional
import logging

logger = logging.getLogger(__name__)

from minio import Minio
from pymilvus import MilvusClient
from pymongo import MongoClient
from pymongo.database import Database
from dotenv import load_dotenv
from knowledge.utils.client.base import BaseClientManager

load_dotenv()


class StorageClients(BaseClientManager):
    """存储类客户端：MinIO、Milvus"""

    _minio_client: Optional[Minio] = None
    _minio_lock = threading.Lock()

    _milvus_client: Optional[MilvusClient] = None
    _milvus_lock = threading.Lock()

    _mongo_db: Optional[Database] = None
    _mongo_lock = threading.Lock()

    # ── MinIO ──

    @classmethod
    def get_minio_client(cls) -> Minio:
        return cls._get_or_create("_minio_client", cls._minio_lock, cls._create_minio_client)

    @classmethod
    def _create_minio_client(cls) -> Minio:
        try:
            endpoint = cls._require_env("MINIO_ENDPOINT")
            access_key = cls._require_env("MINIO_ACCESS_KEY")
            secret_key = cls._require_env("MINIO_SECRET_KEY")
            bucket_name = cls._require_env("MINIO_BUCKET_NAME")

            client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"MinIO bucket '{bucket_name}' 已自动创建")
            else:
                logger.info(f"MinIO bucket '{bucket_name}' 已存在")

            # 图片以直链(path-style GET)形式写入答案，桶必须允许匿名只读，否则浏览器加载 403
            cls._ensure_bucket_public_read(client, bucket_name)

            logger.info(f"MinIO 客户端初始化成功 (endpoint={endpoint})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MinIO 客户端创建失败: {e}")
            raise ConnectionError(f"MinIO 连接失败: {e}") from e

    @staticmethod
    def _ensure_bucket_public_read(client: Minio, bucket_name: str) -> None:
        """为桶设置匿名只读策略（仅 s3:GetObject），使图片直链可被浏览器公开访问。"""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
                }
            ],
        }
        try:
            client.set_bucket_policy(bucket_name, json.dumps(policy))
            logger.info(f"MinIO bucket '{bucket_name}' 已设置为匿名只读")
        except Exception as e:
            logger.warning(f"MinIO bucket '{bucket_name}' 设置只读策略失败: {e}")

    # ── Milvus ──
    @classmethod
    def get_milvus_client(cls) -> MilvusClient:
        return cls._get_or_create("_milvus_client", cls._milvus_lock, cls._create_milvus_client)

    @classmethod
    def _create_milvus_client(cls) -> MilvusClient:
        try:

            milvus_uri = cls._require_env('MILVUS_URL')

            milvus_client = MilvusClient(uri=milvus_uri)

            return milvus_client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"Milvus 客户端创建失败: {e}")
            raise ConnectionError(f"Milvus 连接失败: {e}") from e

    # ── MongoDB ──

    @classmethod
    def get_mongo_db(cls) -> Database:
        return cls._get_or_create("_mongo_db", cls._mongo_lock, cls._create_mongo_db)

    @classmethod
    def _create_mongo_db(cls) -> Database:
        try:
            mongo_url = cls._require_env("MONGO_URL")
            db_name = cls._require_env("MONGO_DB_NAME")

            # 1. 实例化客户端
            client = MongoClient(mongo_url)

            # 2. 根据客户端获取数据库对象
            db = client[db_name]

            logger.info(f"MongoDB 客户端初始化成功 (db={db_name})")

            # 3. 返回数据库对象
            return db
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MongoDB 客户端创建失败: {e}")
            raise ConnectionError(f"MongoDB 连接失败: {e}") from e


if __name__ == '__main__':
    print(StorageClients.get_milvus_client())
