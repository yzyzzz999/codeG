"""
CodeG - 代码向量化模块
使用 CodeBERT 将代码转换为向量表示
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import numpy as np


class CodeEmbedder:
    """代码向量化器"""
    
    def __init__(self, model_name: str = "microsoft/codebert-base", device: str = None):
        """
        初始化向量化器
        
        Args:
            model_name: 预训练模型名称
            device: 计算设备 (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        print(f"模型加载完成，向量维度: {self.embedding_dim}")
    
    def encode(self, code: Union[str, List[str]], batch_size: int = 8) -> np.ndarray:
        """
        将代码编码为向量
        
        Args:
            code: 代码字符串或代码列表
            batch_size: 批处理大小
            
        Returns:
            np.ndarray: 向量表示，shape (n, embedding_dim)
        """
        if isinstance(code, str):
            code = [code]
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(code), batch_size):
                batch = code[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # 获取模型输出
                outputs = self.model(**inputs)
                
                # 使用 [CLS] token 的表示作为代码向量
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def encode_method(self, method_code: str) -> np.ndarray:
        """编码单个方法"""
        return self.encode(method_code)[0]
    
    def similarity(self, code1: str, code2: str) -> float:
        """
        计算两段代码的相似度
        
        Returns:
            float: 余弦相似度，范围 [-1, 1]
        """
        emb1 = self.encode(code1)
        emb2 = self.encode(code2)
        
        # 余弦相似度
        similarity = np.dot(emb1, emb2.T) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        return float(similarity[0][0])


if __name__ == '__main__':
    # 测试代码
    embedder = CodeEmbedder()
    
    code1 = """
    public void printHello() {
        System.out.println("Hello World");
    }
    """
    
    code2 = """
    public void sayHi() {
        System.out.println("Hi there");
    }
    """
    
    code3 = """
    public int calculateSum(int a, int b) {
        return a + b;
    }
    """
    
    emb1 = embedder.encode(code1)
    print(f"Code1 向量 shape: {emb1.shape}")
    
    sim_12 = embedder.similarity(code1, code2)
    sim_13 = embedder.similarity(code1, code3)
    
    print(f"相似代码相似度: {sim_12:.4f}")
    print(f"不相似代码相似度: {sim_13:.4f}")
