import logging
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
_neo4j_driver = None


def get_neo4j_driver() -> GraphDatabase:

    global _neo4j_driver
    try:
        if _neo4j_driver is None:
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")

            logger.info(f"正在初始化 Neo4j 驱动,连接URI:{uri}")

            _neo4j_driver = GraphDatabase.driver(
                uri=uri,
                auth=(username,password)
            )

            _neo4j_driver.verify_connectivity()

            logger.info(f"Neo4j 驱动初始化完成")

        return _neo4j_driver

    except Exception as e:
        logger.error(f"初始化 Neo4j失败:{str(e)}",exc_info=True)
        return None