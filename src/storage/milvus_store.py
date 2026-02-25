"""
CodeG - 向量存储、检索模块
封装集合创建、插入、检索
"""
from typing import List
import logging
import json

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from src.parser.method_extractor import MethodExtractor, MethodInfo
from src.vectorizer.method_vectorizer import MethodVectorizer

# 配置日志
logger = logging.getLogger(__name__)


class MilvusStore:
    """
    向量存储、检索模块
    """

    def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "methods",
                 model_name: str = "microsoft/codebert-base", device: str = None):
        self.collection = None
        self.method_extractor = MethodExtractor()
        self.method_vectorizer = MethodVectorizer(model_name=model_name, device=device)
        self.connections = connections.connect(host=host, port=port)
        self.create_collection(collection_name=collection_name)
        self.create_index()

    def create_collection(self, collection_name: str = "methods", dim: int = 768):
        """
        创建向量集合
        :param collection_name: 向量集合名称
        :param dim: 向量维度
        :return:
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, description="id", is_primary=True, auto_id=True),
            FieldSchema(name="code", dtype=DataType.VARCHAR, description="code", max_length=65535),
            FieldSchema(name="name", dtype=DataType.VARCHAR, description="method_name", max_length=65535),
            FieldSchema(name="parameters", dtype=DataType.VARCHAR, description="parameters", max_length=65535),
            FieldSchema(name="return_type", dtype=DataType.VARCHAR, description="return_type", max_length=65535),
            FieldSchema(name="start_line", dtype=DataType.INT64, description="start_line"),
            FieldSchema(name="end_line", dtype=DataType.INT64, description="end_line"),
            FieldSchema(name="docstring", dtype=DataType.VARCHAR, description="docstring", max_length=65535),
            FieldSchema(name="annotation", dtype=DataType.VARCHAR, description="annotation", max_length=65535),
            FieldSchema(name="code_embedding", dtype=DataType.FLOAT_VECTOR, description="code_embedding", dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description="methods")
        self.collection = Collection(name=collection_name, schema=schema)
        logger.info(f"Collection {collection_name} created successfully")

    def create_index(self):
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="code_embedding", index_params=index_params)
        logger.info("Index created successfully")

    def insert(self, code: str) -> int:
        """
        插入代码中的方法到向量库
        
        Args:
            code: Java代码字符串
            
        Returns:
            int: 成功插入的方法数量
        """
        if not code or not isinstance(code, str):
            logger.warning("Empty or invalid code provided")
            return 0
            
        method_infos: List[MethodInfo] = self.method_extractor.extract_methods_from_code(code)
        
        if not method_infos:
            logger.info("No methods found in the provided code")
            return 0
            
        method_entities = []
        successful_inserts = 0
        
        for method_info in method_infos:
            try:
                method_vector = self.method_vectorizer.encode_method(method_info)
                method_entity = method_info.to_entity()
                # 转换parameters为JSON字符串
                method_entity["parameters"] = json.dumps(method_entity["parameters"])
                # 转换numpy数组为list格式（Milvus要求）
                method_entity["code_embedding"] = method_vector.tolist()
                method_entities.append(method_entity)
                successful_inserts += 1
            except Exception as e:
                logger.error(f"Failed to process method {method_info.name}: {e}")
                continue
                
        if method_entities:
            try:
                # Milvus要求按列插入：每个字段一个列表
                insert_data = [
                    [e["code"] for e in method_entities],
                    [e["name"] for e in method_entities],
                    [e["parameters"] for e in method_entities],
                    [e["return_type"] if e["return_type"] else "" for e in method_entities],
                    [e["start_line"] for e in method_entities],
                    [e["end_line"] for e in method_entities],
                    [e["docstring"] if e["docstring"] else "" for e in method_entities],
                    [e["annotation"] if e["annotation"] else "" for e in method_entities],
                    [e["code_embedding"] for e in method_entities],
                ]
                self.collection.insert(insert_data)
                logger.info(f"Successfully inserted {successful_inserts} methods")
                return successful_inserts
            except Exception as e:
                logger.error(f"Failed to insert methods into Milvus: {e}")
                return 0
                
        return 0

    def search(self, code: str, top_k: int = 10) -> List[MethodInfo]:
        """
        搜索相似的方法
        
        Args:
            code: 查询代码
            top_k: 返回结果数量
            
        Returns:
            List[MethodInfo]: 相似方法列表
        """
        if not code or not isinstance(code, str):
            logger.warning("Empty or invalid query code provided")
            return []
            
        try:
            self.collection.load()
            method_vector = self.method_vectorizer.encode(code)
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 128}
            }
            
            results = self.collection.search(
                data=[method_vector[0].tolist()],
                anns_field="code_embedding", 
                param=search_params,
                limit=top_k,
                output_fields=["name", "code", "parameters", "return_type", "start_line", "end_line", "docstring", "annotation"]
            )
            
            # 正确解析搜索结果
            search_results = []
            if results and len(results) > 0:
                for hits in results:
                    for hit in hits:
                        try:
                            # 反序列化parameters
                            parameters = hit.entity.get("parameters", "[]")
                            try:
                                parameters = json.loads(parameters) if isinstance(parameters, str) else []
                            except:
                                parameters = []
                            
                            method_info = MethodInfo(
                                name=hit.entity.get("name", ""),
                                parameters=parameters,
                                return_type=hit.entity.get("return_type", ""),
                                code=hit.entity.get("code", ""),
                                start_line=hit.entity.get("start_line", 0),
                                end_line=hit.entity.get("end_line", 0),
                                docstring=hit.entity.get("docstring", ""),
                                annotation=hit.entity.get("annotation", "")
                            )
                            search_results.append(method_info)
                        except Exception as e:
                            logger.error(f"Failed to parse search result: {e}")
                            continue
                            
            logger.info(f"Search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []