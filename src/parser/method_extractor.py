"""
方法提取器 - 从Java文件中提取详细的方法信息（修复版）
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import tree_sitter_java as ts_java
from tree_sitter import Language, Parser, Tree, Node

JAVA_LANGUAGE = Language(ts_java.language())


@dataclass
class MethodInfo:
    name: str
    parameters: List[Dict]
    return_type: Optional[str]
    code: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    annotation: Optional[str]


class MethodExtractor:
    """方法提取器"""

    def __init__(self):
        """初始化 Tree-sitter解析器"""
        self.parser = Parser(JAVA_LANGUAGE)

    def extract_methods_from_code(self, code: str) -> List[MethodInfo]:
        """从代码字符串中提取所有方法"""
        tree = self.parser.parse(code.encode('utf-8'))
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

    def _extract_method_info(self, node: Node, source_code: str) -> Optional[MethodInfo]:
        """从AST节点解析方法信息"""
        method_name = self._extract_method_name(node)
        if not method_name:
            return None

        method_code = self._extract_method_code(node, source_code)
        parameters = self._extract_parameters(node)
        return_type = self._extract_return_type(node)
        docstring = self._extract_docstring(node)
        annotation = self._extract_annotation(node)

        return MethodInfo(
            name=method_name,
            parameters=parameters,
            return_type=return_type,
            code=method_code,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            annotation=annotation
        )

    def _extract_method_name(self, node: Node) -> Optional[str]:
        """提取方法名"""
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8')
        return None

    def _extract_parameters(self, node: Node) -> List[Dict]:
        """提取参数列表"""
        parameters = []
        for child in node.children:
            if child.type == 'formal_parameters':
                for param in child.children:
                    if param.type == 'formal_parameter':
                        parameter = self._extract_parameter(param)
                        if parameter:
                            parameters.append(parameter)
                break
        return parameters

    def _extract_parameter(self, param: Node) -> Optional[Dict]:
        """提取单个参数信息"""
        param_name = None
        param_type = None
        for child in param.children:
            if child.type == 'identifier':
                param_name = child.text.decode('utf-8')
            elif child.type == 'type_identifier':
                param_type = child.text.decode('utf-8')
        return {
            'name': param_name,
            'type': param_type
        }

    def _extract_return_type(self, node: Node) -> Optional[str]:
        """提取返回类型"""
        for i, child in enumerate(node.children):
            if child.type == 'identifier':  # 找到方法名
                # 查找方法名之前的类型
                for j in range(i):
                    prev_child = node.children[j]
                    if (prev_child.type.endswith('_type') or
                        prev_child.type in ['type_identifier', 'void_type']):
                        return prev_child.text.decode('utf-8')
        return None

    def _extract_docstring(self, node: Node) -> Optional[str]:
        """提取文档注释"""
        parent = node.parent
        if parent:
            for i, child in enumerate(parent.children):
                if child == node:  # 找到当前节点位置
                    # 向前查找最近的文档注释
                    for j in range(i-1, -1, -1):
                        prev_sibling = parent.children[j]
                        if prev_sibling.type == 'block_comment':
                            comment_text = prev_sibling.text.decode('utf-8').strip()
                            if comment_text.startswith('/**'):
                                return comment_text.replace('/**', '').replace('*/', '').strip()
                            elif comment_text.startswith('/*'):
                                return comment_text.replace('/*', '').replace('*/', '').strip()
                            elif comment_text.startswith('//'):
                                return comment_text.replace('//', '').strip()
                        elif prev_sibling.type not in ['block_comment', 'line_comment']:
                            break
        return None

    def _extract_annotation(self, node: Node) -> Optional[str]:
        """提取注解"""
        annotations = []
        for child in node.children:
            if child.type == 'marker_annotation':
                annotations.append(child.text.decode('utf-8'))
            elif child.type == 'modifiers':
                for modifier_child in child.children:
                    if modifier_child.type == 'marker_annotation':
                        annotations.append(modifier_child.text.decode('utf-8'))
        return ", ".join(annotations) if annotations else None

    def _extract_method_code(self, node: Node, source_code: str) -> str:
        lines = source_code.split('\n')
        method_code = '\n'.join(lines[node.start_point[0]:node.end_point[0]+1]).strip()
        return method_code


def main():
    extractor = MethodExtractor()

    test_code = '''
public class Calculator {

/**
已弃置
*/
    @Deprecated
    public int add(int a, int b) {
        return a + b;
    }
    
    /**
    乘法
    */
    public double multiply(double x, double y) {
        return x * y;
    }
}
'''

    methods = extractor.extract_methods_from_code(test_code)
    print(f"找到 {len(methods)} 个方法:")
    for i, method in enumerate(methods, 1):
        print(f"\n--- 方法 {i} ---")
        print(f"名称: {method.name}")
        print(f"返回类型: {method.return_type}")
        print(f"注解: {method.annotation}")
        print(f"文档注释: {method.docstring if method.docstring else '无'}")
        print("参数:")
        for param in method.parameters:
            print(f"  - {param['name']}: {param['type']}")
        print("代码预览:", method.code.replace('\n', '\\n'))


if __name__ == '__main__':
    main()