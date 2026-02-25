"""
CodeG - 代码向量化模块
使用 CodeBERT 将代码转换为向量表示
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import numpy as np
from src.parser.method_extractor import MethodInfo


class MethodVectorizer:
    """
    MethodVectorizer - 方法向量化类
    使用 CodeBERT 将方法代码转换为向量表示
    """

    def __init__(self, model_name: str = "microsoft/codebert-base", device: str = None):
        self.model_name = model_name
        # 修复：明确的设备处理逻辑
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size

    def encode(self, code: Union[str, List[str]], batch_size: int = 8) -> np.ndarray:
        """
        encode - 将代码转换为向量表示

        Parameters:
        - code: Union[str, List[str]] - 代码或代码列表
        - batch_size: int - 批量处理大小

        Returns:
        - np.ndarray - 向量表示
        """
        if isinstance(code, str):
            code = [code]

        # 严格处理空列表情况
        if len(code) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(code), batch_size):
                batch = code[i:i + batch_size]
                inputs = (self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt",
                                         max_length=512)
                          .to(self.device))
                outputs = self.model(**inputs)
                # 使用[CLS] token的表示（严格按照规范）
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def encode_method(self, method_info: MethodInfo) -> np.ndarray:
        """
        encode_method - 将方法代码转换为向量表示

        Parameters:
        - method_info: MethodInfo - 方法信息对象

        Returns:
        - np.ndarray - 向量表示
        """
        if not isinstance(method_info, MethodInfo):
            raise TypeError("method_info must be MethodInfo instance")
        
        if not method_info.code:
            raise ValueError("method_info.code cannot be empty")
            
        return self.encode([method_info.code])[0]

    def similarity(self, code1: str, code2: str) -> float:
        """
        similarity - 计算两个代码的相似度

        Parameters:
        - code1: str - 代码1
        - code2: str - 代码2

        Returns:
        - float - 相似度 (-1.0 到 1.0)
        """
        if not code1 or not code2:
            return 0.0
            
        embedding1 = self.encode([code1])
        embedding2 = self.encode([code2])

        # 严格的余弦相似度计算
        vec1 = embedding1[0]
        vec2 = embedding2[0]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # 严格的零向量处理
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        
        # 确保返回值在有效范围内
        return float(max(-1.0, min(1.0, similarity)))