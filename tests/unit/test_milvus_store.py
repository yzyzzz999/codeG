"""
MilvusStore 模块单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from src.storage.milvus_store import MilvusStore
from src.parser.method_extractor import MethodInfo


class TestMilvusStore:
    """测试 MilvusStore 类"""
    
    @pytest.fixture
    def mock_collection(self):
        """创建mock的Milvus集合"""
        collection = Mock()
        collection.name = "test_collection"
        return collection
    
    @pytest.fixture
    def mock_connections(self):
        """创建mock的连接"""
        with patch('src.storage.milvus_store.connections') as mock_conn:
            mock_conn.connect.return_value = Mock()
            yield mock_conn
    
    @patch('src.storage.milvus_store.MethodExtractor')
    @patch('src.storage.milvus_store.MethodVectorizer')
    @patch('src.storage.milvus_store.Collection')
    def test_init(self, mock_collection_class, mock_vectorizer_class, mock_extractor_class, mock_connections):
        """测试初始化"""
        # 设置mock
        mock_collection_instance = Mock()
        mock_collection_class.return_value = mock_collection_instance
        mock_vectorizer_instance = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer_instance
        mock_extractor_instance = Mock()
        mock_extractor_class.return_value = mock_extractor_instance
        
        # 创建实例
        store = MilvusStore(
            host="localhost",
            port="19530",
            collection_name="test_methods",
            model_name="test-model"
        )
        
        # 验证调用
        mock_connections.connect.assert_called_once_with(host="localhost", port="19530")
        mock_collection_class.assert_called_once()
        mock_vectorizer_class.assert_called_once_with(model_name="test-model", device=None)
        mock_extractor_class.assert_called_once()
        
        assert store.collection == mock_collection_instance
        assert store.method_extractor == mock_extractor_instance
        assert store.method_vectorizer == mock_vectorizer_instance
    
    @patch('src.storage.milvus_store.FieldSchema')
    @patch('src.storage.milvus_store.CollectionSchema')
    @patch('src.storage.milvus_store.DataType')
    @patch('src.storage.milvus_store.connections')
    def test_create_collection(self, mock_connections, mock_datatype, mock_schema, mock_field_schema):
        """测试创建集合"""
        # Mock连接
        mock_conn = Mock()
        mock_connections.connect.return_value = mock_conn
        
        store = MilvusStore.__new__(MilvusStore)  # 不调用__init__
        store.collection = Mock()
        
        # 调用被测试方法
        store.create_collection(collection_name="test_collection", dim=512)
        
        # 验证FieldSchema被正确调用
        assert mock_field_schema.call_count >= 9  # 至少9个字段
    
    @patch('src.storage.milvus_store.MethodExtractor')
    @patch('src.storage.milvus_store.MethodVectorizer')
    def test_insert_single_method(self, mock_vectorizer_class, mock_extractor_class, mock_connections):
        """测试插入单个方法"""
        # 设置mock
        mock_extractor = Mock()
        mock_vectorizer = Mock()
        mock_collection = Mock()
        
        mock_extractor_class.return_value = mock_extractor
        mock_vectorizer_class.return_value = mock_vectorizer
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        
        # 创建测试数据
        test_method = MethodInfo(
            name="testMethod",
            parameters=[{"name": "param1", "type": "String"}],
            return_type="void",
            code="public void testMethod(String param1) { }",
            start_line=1,
            end_line=3,
            docstring="Test method",
            annotation="@Test"
        )
        
        mock_extractor.extract_methods_from_code.return_value = [test_method]
        mock_vectorizer.encode_method.return_value = np.array([0.1, 0.2, 0.3])  # 3维向量
        
        with patch('src.storage.milvus_store.Collection', mock_collection):
            store = MilvusStore(collection_name="test_collection")
            
            # 测试插入
            test_code = "public class Test { public void testMethod(String param1) { } }"
            store.insert(test_code)
            
            # 验证调用
            mock_extractor.extract_methods_from_code.assert_called_once_with(test_code)
            mock_vectorizer.encode_method.assert_called_once_with(test_method)
            mock_collection_instance.insert.assert_called_once()
    
    @patch('src.storage.milvus_store.MethodExtractor')
    @patch('src.storage.milvus_store.MethodVectorizer')
    def test_insert_multiple_methods(self, mock_vectorizer_class, mock_extractor_class, mock_connections):
        """测试插入多个方法"""
        # 设置mock
        mock_extractor = Mock()
        mock_vectorizer = Mock()
        mock_collection = Mock()
        
        mock_extractor_class.return_value = mock_extractor
        mock_vectorizer_class.return_value = mock_vectorizer
        
        # 创建多个测试方法
        methods = [
            MethodInfo(name=f"method{i}", parameters=[], return_type="void",
                      code=f"public void method{i}() {{ }}", start_line=i, end_line=i,
                      docstring=None, annotation=None)
            for i in range(3)
        ]
        
        mock_extractor.extract_methods_from_code.return_value = methods
        mock_vectorizer.encode_method.return_value = np.array([0.1, 0.2, 0.3])
        
        with patch('src.storage.milvus_store.Collection', mock_collection):
            store = MilvusStore(collection_name="test_collection")
            
            test_code = "multiple methods code"
            store.insert(test_code)
            
            # 验证多次调用
            assert mock_vectorizer.encode_method.call_count == 3
            mock_collection_instance = mock_collection.return_value
            mock_collection_instance.insert.assert_called_once()
    
    @patch('src.storage.milvus_store.MethodVectorizer')
    def test_search(self, mock_vectorizer_class, mock_connections):
        """测试搜索功能"""
        # 设置mock
        mock_vectorizer = Mock()
        mock_collection = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer
        
        # 模拟搜索结果 - 修复：确保返回正确的结果格式
        mock_hit = Mock()
        mock_hit.entity = Mock()
        mock_hit.entity.get.side_effect = lambda x, default=None: {
            'name': 'foundMethod',
            'parameters': [],
            'return_type': 'void',
            'code': 'public void foundMethod() { }',
            'start_line': 1,
            'end_line': 1,
            'docstring': None,
            'annotation': None
        }.get(x, default)
        
        mock_search_result = Mock()
        mock_search_result.__iter__ = Mock(return_value=iter([mock_hit]))
        mock_search_result.__len__ = Mock(return_value=1)
        
        mock_collection_instance = Mock()
        mock_collection_instance.search.return_value = [[mock_search_result]]  # 修正返回格式
        mock_collection.return_value = mock_collection_instance
        
        mock_vectorizer.encode.return_value = np.array([[0.5, 0.5, 0.5]])  # 查询向量
        
        with patch('src.storage.milvus_store.Collection', mock_collection):
            store = MilvusStore(collection_name="test_collection")
            
            # 执行搜索
            query_code = "search query code"
            results = store.search(query_code, top_k=5)
            
            # 验证结果
            assert len(results) == 1
            assert isinstance(results[0], MethodInfo)
            assert results[0].name == 'foundMethod'
            
            # 验证调用
            mock_vectorizer.encode.assert_called_once_with(query_code)
            mock_collection_instance.load.assert_called_once()
            mock_collection_instance.search.assert_called_once()
    
    def test_empty_insert(self, mock_connections):
        """测试插入空代码"""
        with patch('src.storage.milvus_store.MethodExtractor') as mock_extractor_class, \
             patch('src.storage.milvus_store.MethodVectorizer') as mock_vectorizer_class, \
             patch('src.storage.milvus_store.Collection') as mock_collection_class:
            
            mock_extractor = Mock()
            mock_extractor.extract_methods_from_code.return_value = []  # 空方法列表
            mock_extractor_class.return_value = mock_extractor
            
            mock_collection_instance = Mock()
            mock_collection_class.return_value = mock_collection_instance
            
            store = MilvusStore()
            
            # 插入不包含方法的代码
            store.insert("public class EmptyClass { }")
            
            # 验证没有调用插入操作
            mock_collection_instance.insert.assert_not_called()
    
    @patch('src.storage.milvus_store.MethodVectorizer')
    def test_search_empty_results(self, mock_vectorizer_class, mock_connections):
        """测试搜索返回空结果"""
        mock_vectorizer = Mock()
        mock_vectorizer.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_vectorizer_class.return_value = mock_vectorizer
        
        mock_collection_instance = Mock()
        mock_collection_instance.search.return_value = [[]]  # 空搜索结果
        mock_collection = Mock(return_value=mock_collection_instance)
        
        with patch('src.storage.milvus_store.Collection', mock_collection):
            store = MilvusStore()
            
            results = store.search("query code")
            assert len(results) == 0


class TestMilvusStoreIntegration:
    """集成测试"""
    
    @pytest.mark.skip(reason="需要运行的Milvus服务")
    def test_real_milvus_integration(self):
        """真实的Milvus集成测试（需要实际服务）"""
        # 这个测试需要实际的Milvus服务运行
        try:
            store = MilvusStore(host="localhost", port="19530")
            
            # 插入测试数据
            test_code = """
            public class TestClass {
                public void hello() {
                    System.out.println("Hello");
                }
                
                public int add(int a, int b) {
                    return a + b;
                }
            }
            """
            
            store.insert(test_code)
            
            # 搜索测试
            query_code = "public void hello() { System.out.println(\"Hello\"); }"
            results = store.search(query_code, top_k=2)
            
            assert len(results) > 0
            assert any("hello" in result.name.lower() for result in results)
            
        except Exception as e:
            pytest.skip(f"Milvus服务不可用: {e}")