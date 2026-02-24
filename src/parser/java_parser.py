"""
CodeG - 代码解析模块
使用 Tree-sitter 解析 Java 代码，提取 AST 特征
"""

from typing import List, Dict, Optional, Tuple
import tree_sitter_java as ts_java
from tree_sitter import Language, Parser, Tree, Node


# 初始化 Java 语言解析器
JAVA_LANGUAGE = Language(ts_java.language())


class JavaCodeParser:
    """Java 代码解析器"""
    
    def __init__(self):
        self.parser = Parser(JAVA_LANGUAGE)
    
    def parse(self, code: str) -> Tree:
        """解析代码字符串，返回 AST"""
        return self.parser.parse(code.encode('utf-8'))
    
    def extract_methods(self, code: str) -> List[Dict]:
        """
        提取代码中的所有方法定义
        
        Returns:
            List[Dict]: 方法列表，每个方法包含 name, start_line, end_line, code
        """
        tree = self.parse(code)
        methods = []
        
        def traverse(node: Node):
            if node.type == 'method_declaration':
                method_info = self._extract_method_info(node, code)
                if method_info:
                    methods.append(method_info)
            for child in node.children:
                traverse(child)
        
        traverse(tree.root_node)
        return methods
    
    def _extract_method_info(self, node: Node, source_code: str) -> Optional[Dict]:
        """从方法节点提取信息"""
        method_name = None
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        # 查找方法名
        for child in node.children:
            if child.type == 'identifier':
                method_name = child.text.decode('utf-8')
                break
        
        if not method_name:
            return None
        
        # 提取方法代码
        lines = source_code.split('\n')
        method_code = '\n'.join(lines[start_line-1:end_line])
        
        return {
            'name': method_name,
            'start_line': start_line,
            'end_line': end_line,
            'code': method_code,
            'ast_node': node
        }
    
    def extract_imports(self, code: str) -> List[str]:
        """提取导入语句"""
        tree = self.parse(code)
        imports = []
        
        def traverse(node: Node):
            if node.type == 'import_declaration':
                import_text = code[node.start_byte:node.end_byte]
                imports.append(import_text)
            for child in node.children:
                traverse(child)
        
        traverse(tree.root_node)
        return imports
    
    def get_ast_depth(self, node: Node) -> int:
        """计算 AST 节点深度"""
        if not node.children:
            return 1
        return 1 + max(self.get_ast_depth(child) for child in node.children)
    
    def extract_code_metrics(self, code: str) -> Dict:
        """
        提取代码度量指标
        
        Returns:
            Dict: 包含 lines, methods_count, avg_method_length, ast_depth
        """
        tree = self.parse(code)
        methods = self.extract_methods(code)
        
        lines = len(code.split('\n'))
        methods_count = len(methods)
        avg_method_length = sum(
            m['end_line'] - m['start_line'] + 1 for m in methods
        ) / methods_count if methods_count > 0 else 0
        
        return {
            'lines': lines,
            'methods_count': methods_count,
            'avg_method_length': round(avg_method_length, 2),
            'ast_depth': self.get_ast_depth(tree.root_node)
        }


if __name__ == '__main__':
    # 测试代码
    sample_code = '''
public class Test {
    public void hello() {
        System.out.println("Hello");
    }
    
    public int add(int a, int b) {
        return a + b;
    }
}
'''
    
    parser = JavaCodeParser()
    methods = parser.extract_methods(sample_code)
    print(f"找到 {len(methods)} 个方法:")
    for method in methods:
        print(f"  - {method['name']}: 第 {method['start_line']}-{method['end_line']} 行")
    
    metrics = parser.extract_code_metrics(sample_code)
    print(f"\n代码指标: {metrics}")
