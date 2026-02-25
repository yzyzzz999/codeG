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
        """提取方法名 - 修复：使用安全解码"""
        for child in node.children:
            if child.type == 'identifier':
                return self._safe_decode(child.text)
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
        """提取单个参数信息 - 支持泛型、数组等复杂类型"""
        param_name = None
        param_type = None
        for child in param.children:
            if child.type == 'identifier':
                param_name = self._safe_decode(child.text)
            elif child.type == 'type_identifier' or child.type.endswith('_type'):
                param_type = self._extract_type_string(child)
        return {
            'name': param_name,
            'type': param_type
        }

    def _extract_type_string(self, node: Node) -> str:
        """递归提取类型字符串，支持泛型和数组"""
        if node.type == 'type_identifier':
            return self._safe_decode(node.text)
        elif node.type == 'void_type':
            return 'void'
        elif node.type == 'array_type':
            # 数组类型: type + dimensions
            element_type = None
            dimensions = ''
            for child in node.children:
                if child.type in ['type_identifier', 'generic_type']:
                    element_type = self._extract_type_string(child)
                elif child.type == 'dimensions' or child.type.endswith('_type'):
                    dimensions += self._safe_decode(child.text)
            return f"{element_type}{dimensions}" if element_type else dimensions
        elif node.type == 'generic_type':
            # 泛型: List<String>, Map<K, V> 等
            base_type = None
            type_args = []
            for child in node.children:
                if child.type == 'type_identifier':
                    base_type = self._safe_decode(child.text)
                elif child.type == 'type_arguments':
                    for arg in child.children:
                        if arg.type in ['type_identifier', 'generic_type', 'wildcard']:
                            type_args.append(self._extract_type_string(arg))
            if base_type and type_args:
                return f"{base_type}<{', '.join(type_args)}>"
            return base_type or self._safe_decode(node.text)
        elif node.type == 'wildcard':
            # 通配符: ? extends T, ? super T
            return self._safe_decode(node.text)
        return self._safe_decode(node.text)

    def _extract_return_type(self, node: Node) -> Optional[str]:
        """提取返回类型 - 修复：支持泛型和数组返回类型"""
        for i, child in enumerate(node.children):
            if child.type == 'identifier':  # 找到方法名
                # 查找方法名之前的类型
                for j in range(i):
                    prev_child = node.children[j]
                    if prev_child.type == 'type_identifier' or prev_child.type.endswith('_type'):
                        return self._extract_type_string(prev_child)
        return None

    def _safe_decode(self, text) -> str:
        """安全解码，处理 None 情况"""
        if text is None:
            return ""
        if isinstance(text, bytes):
            return text.decode('utf-8')
        return str(text)

    def _extract_docstring(self, node: Node) -> Optional[str]:
        """提取文档注释 - 修复：在方法声明前查找 Javadoc 风格的块注释"""
        # 方法声明的父节点通常是 class_body 或 interface_body
        parent = node.parent
        if not parent:
            return None
        
        # 找到当前方法在父节点中的索引
        try:
            node_index = parent.children.index(node)
        except ValueError:
            return None
        
        # 向前查找最近的块注释
        for i in range(node_index - 1, -1, -1):
            prev_sibling = parent.children[i]
            
            # 跳过空白节点（某些 parser 版本可能有）
            if prev_sibling.type == 'line_comment':
                # 单行注释通常不算文档注释，继续向前找
                continue
            elif prev_sibling.type == 'block_comment':
                comment_text = self._safe_decode(prev_sibling.text)
                if comment_text:
                    comment_text = comment_text.strip()
                    # Javadoc 风格: /**
                    if comment_text.startswith('/**'):
                        return self._clean_docstring(comment_text)
                    # 普通块注释: /*
                    elif comment_text.startswith('/*') and not comment_text.startswith('/**'):
                        return self._clean_docstring(comment_text)
                # 找到注释就停止（不继续向前找更老的注释）
                break
            else:
                # 遇到非注释节点就停止
                break
        
        return None
    
    def _clean_docstring(self, comment: str) -> str:
        """清理文档注释，移除 /* */ 和每行的 * 前缀"""
        lines = comment.split('\n')
        cleaned_lines = []
        for line in lines:
            # 移除开头的 /* 或 /**
            line = line.strip()
            if line.startswith('/**'):
                line = line[3:]
            elif line.startswith('/*'):
                line = line[2:]
            # 移除结尾的 */
            if line.endswith('*/'):
                line = line[:-2]
            # 移除每行开头的 *
            line = line.lstrip('*').strip()
            if line:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def _extract_annotation(self, node: Node) -> Optional[str]:
        """提取注解 - 修复：支持带参数的注解如 @RequestMapping(\"/api\")"""
        annotations = []
        
        def parse_annotation(annotation_node: Node) -> str:
            """解析单个注解节点"""
            if annotation_node.type == 'marker_annotation':
                # 无参注解: @Override
                return self._safe_decode(annotation_node.text)
            elif annotation_node.type == 'annotation':
                # 带参数的注解: @RequestMapping("/api")
                name = None
                arguments = None
                for child in annotation_node.children:
                    if child.type == 'identifier':
                        name = self._safe_decode(child.text)
                    elif child.type == 'annotation_argument_list':
                        arguments = self._safe_decode(child.text)
                if name and arguments:
                    return f"@{name}{arguments}"
                elif name:
                    return f"@{name}"
            return self._safe_decode(annotation_node.text)
        
        for child in node.children:
            if child.type in ['marker_annotation', 'annotation']:
                annotations.append(parse_annotation(child))
            elif child.type == 'modifiers':
                for modifier_child in child.children:
                    if modifier_child.type in ['marker_annotation', 'annotation']:
                        annotations.append(parse_annotation(modifier_child))
        
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
     * 已弃置
     * @deprecated 请使用 addV2
     */
    @Deprecated
    public int add(int a, int b) {
        return a + b;
    }
    
    /**
     * 乘法运算
     * @param x 第一个数
     * @param y 第二个数
     * @return 乘积
     */
    public double multiply(double x, double y) {
        return x * y;
    }
    
    // 带泛型和数组的测试
    @RequestMapping("/api/calculate")
    public List<String> processItems(int[] items, Map<String, Object> config) {
        return new ArrayList<>();
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
        print(f"文档注释:")
        if method.docstring:
            for line in method.docstring.split('\n'):
                print(f"  {line}")
        else:
            print("  无")
        print("参数:")
        for param in method.parameters:
            print(f"  - {param['name']}: {param['type']}")
        print(f"代码行数: {method.end_line - method.start_line + 1}")


if __name__ == '__main__':
    main()