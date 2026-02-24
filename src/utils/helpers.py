"""
CodeG - 工具函数模块
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_str: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    配置日志
    
    Args:
        level: 日志级别
        format_str: 日志格式
        log_file: 日志文件路径
        
    Returns:
        logging.Logger: 配置好的 logger
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=handlers
    )
    
    return logging.getLogger("CodeG")


def read_java_file(file_path: str) -> str:
    """读取 Java 文件内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def batch_iterator(items: list, batch_size: int):
    """批量迭代器"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
