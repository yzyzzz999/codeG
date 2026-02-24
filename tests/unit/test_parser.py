"""
代码解析模块单元测试
"""

import pytest
from src.parser.java_parser import JavaCodeParser


class TestJavaParser:
    """测试 Java 代码解析器"""
    
    @pytest.fixture
    def parser(self):
        return JavaCodeParser()
    
    def test_extract_methods(self, parser):
        """测试方法提取"""
        code = '''
public class Test {
    public void hello() {
        System.out.println("Hello");
    }
    
    public int add(int a, int b) {
        return a + b;
    }
}
'''
        methods = parser.extract_methods(code)
        assert len(methods) == 2
        assert methods[0]['name'] == 'hello'
        assert methods[1]['name'] == 'add'
    
    def test_extract_imports(self, parser):
        """测试导入提取"""
        code = '''
import java.util.List;
import java.io.File;

public class Test {}
'''
        imports = parser.extract_imports(code)
        assert len(imports) == 2
        assert 'java.util.List' in imports[0]
    
    def test_code_metrics(self, parser):
        """测试代码指标"""
        code = '''
public class Test {
    public void method1() {}
    public void method2() {}
}
'''
        metrics = parser.extract_code_metrics(code)
        assert metrics['methods_count'] == 2
        assert metrics['lines'] == 5
