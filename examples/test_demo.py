"""
测试演示脚本
展示如何使用 MethodVectorizer 和 MilvusStore
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.vectorizer.method_vectorizer import MethodVectorizer
from src.storage.milvus_store import MilvusStore
from src.parser.method_extractor import MethodInfo
import numpy as np


def demo_method_vectorizer():
    """演示 MethodVectorizer 功能"""
    print("=== MethodVectorizer 演示 ===")
    
    # 创建向量化器
    vectorizer = MethodVectorizer(device="cpu")
    print(f"✓ 向量化器创建成功，维度: {vectorizer.embedding_dim}")
    
    # 测试代码样本
    code_samples = [
        "public void printHello() { System.out.println(\"Hello\"); }",
        "public void sayHi() { System.out.println(\"Hi there\"); }",
        "public int add(int a, int b) { return a + b; }"
    ]
    
    # 编码测试
    print("\n1. 编码测试:")
    embeddings = vectorizer.encode(code_samples)
    print(f"   输入数量: {len(code_samples)}")
    print(f"   输出形状: {embeddings.shape}")
    
    # 相似度测试
    print("\n2. 相似度测试:")
    similarity_1_2 = vectorizer.similarity(code_samples[0], code_samples[1])
    similarity_1_3 = vectorizer.similarity(code_samples[0], code_samples[2])
    print(f"   相似方法相似度: {similarity_1_2:.4f}")
    print(f"   不同方法相似度: {similarity_1_3:.4f}")
    
    # 方法信息编码测试
    print("\n3. 方法信息编码测试:")
    method_info = MethodInfo(
        name="testMethod",
        parameters=[{"name": "x", "type": "int"}],
        return_type="void",
        code="public void testMethod(int x) { }",
        start_line=1,
        end_line=1,
        docstring="Test method",
        annotation="@Test"
    )
    
    method_embedding = vectorizer.encode_method(method_info)
    print(f"   方法向量形状: {method_embedding.shape}")
    print(f"   向量范数: {np.linalg.norm(method_embedding):.4f}")


def demo_milvus_store():
    """演示 MilvusStore 功能（需要Milvus服务）"""
    print("\n=== MilvusStore 演示 ===")
    
    try:
        # 创建存储实例
        store = MilvusStore(
            host="localhost",
            port="19530",
            collection_name="test_methods_demo"
        )
        print("✓ MilvusStore 创建成功")
        
        # 插入测试数据
        print("\n1. 插入测试数据:")
        test_code = '''
        public class MathUtils {
            public int factorial(int n) {
                if (n <= 1) return 1;
                return n * factorial(n - 1);
            }
            
            public boolean isPrime(int num) {
                if (num < 2) return false;
                for (int i = 2; i <= Math.sqrt(num); i++) {
                    if (num % i == 0) return false;
                }
                return true;
            }
        }
        '''
        
        store.insert(test_code)
        print("   ✓ 数据插入完成")
        
        # 搜索测试
        print("\n2. 搜索测试:")
        query_code = "public int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }"
        results = store.search(query_code, top_k=2)
        
        print(f"   找到 {len(results)} 个相似方法:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.name} - {result.return_type}")
            print(f"      行数: {result.start_line}-{result.end_line}")
            
    except Exception as e:
        print(f"✗ Milvus连接失败: {e}")
        print("   请确保Milvus服务正在运行 (docker-compose up -d)")


def main():
    """主函数"""
    print("CodeG 测试演示")
    print("=" * 50)
    
    # 运行向量化器演示
    demo_method_vectorizer()
    
    # 运行Milvus演示（可选）
    demo_milvus_store()
    
    print("\n" + "=" * 50)
    print("演示完成!")


if __name__ == "__main__":
    main()