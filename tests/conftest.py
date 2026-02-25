"""
测试配置文件
"""

import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_java_code():
    """提供示例Java代码"""
    return '''
public class Calculator {
    
    /**
     * 计算两个数的和
     * @param a 第一个数
     * @param b 第二个数
     * @return 两数之和
     */
    public int add(int a, int b) {
        return a + b;
    }
    
    /**
     * 计算两个数的乘积
     * @param x 第一个数
     * @param y 第二个数
     * @return 两数乘积
     */
    public double multiply(double x, double y) {
        return x * y;
    }
    
    @Deprecated
    public void oldMethod() {
        System.out.println("This is deprecated");
    }
}
'''


@pytest.fixture
def mock_method_info():
    """创建mock的方法信息"""
    from src.parser.method_extractor import MethodInfo
    return MethodInfo(
        name="testMethod",
        parameters=[{"name": "param1", "type": "String"}],
        return_type="void",
        code="public void testMethod(String param1) { }",
        start_line=1,
        end_line=1,
        docstring="Test method documentation",
        annotation="@Test"
    )


# 环境变量设置
def pytest_configure(config):
    """测试配置"""
    # 设置测试环境变量
    os.environ.setdefault('TESTING', 'true')
    os.environ.setdefault('MILVUS_HOST', 'localhost')
    os.environ.setdefault('MILVUS_PORT', '19530')


def pytest_unconfigure(config):
    """测试结束清理"""
    # 清理测试环境变量
    for key in ['TESTING', 'MILVUS_HOST', 'MILVUS_PORT']:
        os.environ.pop(key, None)