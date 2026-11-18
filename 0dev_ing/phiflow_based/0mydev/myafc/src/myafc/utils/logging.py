# -*- coding: utf-8 -*-
"""
myafc.utils.logging - 统一日志配置模块

为 myafc 包提供统一的日志记录功能。所有模块使用同一个 logger，
支持控制台输出和文件输出。

用法:
    from myafc.utils import get_logger, setup_file_logging
    
    # 获取 logger（所有模块共用）
    logger = get_logger()
    logger.info("这是一条日志")
    
    # 配置文件输出（通常在 SolverRunner 初始化时调用）
    setup_file_logging('/path/to/run.log')
    
    # 后续所有日志将同时输出到控制台和文件
    logger.info("这条日志会同时写入文件")

特性:
    - 单例 logger: 整个 myafc 包使用同一个 logger 实例
    - 懒初始化: 首次调用 get_logger() 时配置控制台处理器
    - 文件输出: 调用 setup_file_logging() 后启用
    - 避免重复: 自动检测并跳过重复的处理器
"""
import os
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# ============================================================
# 模块级常量
# ============================================================

# 日志器名称（myafc 包的根 logger）
LOGGER_NAME = 'myafc'

# 默认日志格式
DEFAULT_FORMAT = '[%(asctime)s][%(levelname)s] %(message)s'
DEFAULT_DATE_FORMAT = '%H:%M:%S'

# 文件日志格式（包含模块名，便于追踪）
FILE_FORMAT = '[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
FILE_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 默认日志级别
DEFAULT_LEVEL = logging.INFO

# 模块级状态
_logger: Optional[logging.Logger] = None
_console_handler_added = False


# ============================================================
# 公开 API
# ============================================================

def get_logger() -> logging.Logger:
    """
    获取 myafc 包的统一 logger
    
    首次调用时会初始化 logger 并添加控制台处理器。
    后续调用返回同一个 logger 实例。
    
    Returns:
        logging.Logger: myafc 包的 logger 实例
    
    Example:
        >>> logger = get_logger()
        >>> logger.info("初始化完成")
        [12:34:56][INFO] 初始化完成
    """
    global _logger, _console_handler_added
    
    if _logger is None:
        _logger = logging.getLogger(LOGGER_NAME)
        _logger.setLevel(DEFAULT_LEVEL)
        # 阻止日志传播到 root logger，避免重复输出
        _logger.propagate = False
    
    # 确保控制台处理器只添加一次
    if not _console_handler_added:
        _add_console_handler(_logger)
        _console_handler_added = True
    
    return _logger


def setup_file_logging(
    log_path: str,
    level: int = logging.INFO,
    append: bool = True
) -> None:
    """
    为 logger 添加文件输出
    
    调用后，所有日志将同时输出到控制台和指定文件。
    自动创建日志文件所在目录。
    
    Args:
        log_path: 日志文件路径
        level: 文件日志级别（默认 INFO）
        append: 是否追加模式（默认 True）。
                False 时会覆盖已有文件。
    
    Note:
        - 同一路径的文件处理器只会添加一次
        - 可多次调用以添加多个日志文件
    
    Example:
        >>> setup_file_logging('./outputs/run.log')
        >>> get_logger().info("这条日志会写入文件")
    """
    logger = get_logger()
    log_path = os.path.abspath(log_path)
    
    # 检查是否已添加相同路径的处理器
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if handler.baseFilename == log_path:
                # 已存在，跳过
                return
    
    # 创建目录
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 创建文件处理器
    mode = 'a' if append else 'w'
    file_handler = logging.FileHandler(log_path, mode=mode, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATE_FORMAT))
    
    logger.addHandler(file_handler)
    logger.info(f"日志文件: {log_path}")


def set_log_level(level: str) -> None:
    """
    设置日志级别
    
    Args:
        level: 日志级别字符串
            可选值: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    
    Example:
        >>> set_log_level('DEBUG')  # 显示所有日志
        >>> set_log_level('WARNING')  # 只显示警告及以上
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    
    if not isinstance(level, str):
        get_logger().warning(f"日志级别应为字符串，收到: {type(level)}，使用默认 INFO")
        level = 'INFO'
    
    level_upper = level.upper()
    if level_upper not in level_map:
        get_logger().warning(f"未知日志级别: {level}，使用默认 INFO")
        level_upper = 'INFO'
    
    log_level = level_map[level_upper]
    
    logger = get_logger()
    logger.setLevel(log_level)
    
    # 同时更新所有处理器的级别
    for handler in logger.handlers:
        handler.setLevel(log_level)


# ============================================================
# 内部函数
# ============================================================

def _add_console_handler(logger: logging.Logger) -> None:
    """添加控制台处理器（内部使用）"""
    # 检查是否已有控制台处理器
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            return

    # 在 Windows 上尽量将标准流编码切换为 UTF-8，降低中文日志乱码概率
    if os.name == 'nt':
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    # 使用原始 stdout 作为控制台日志流，避免后续 stdout/stderr tee 影响 logger 输出
    console_stream = getattr(sys, '__stdout__', None) or sys.stdout
    console_handler = logging.StreamHandler(stream=console_stream)
    console_handler.setLevel(DEFAULT_LEVEL)
    console_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT))
    logger.addHandler(console_handler)


class _TeeStream:
    """将写入复制到多个流的轻量级文件对象。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    @property
    def encoding(self):
        first = self._streams[0] if self._streams else None
        return getattr(first, 'encoding', None)

    @property
    def errors(self):
        first = self._streams[0] if self._streams else None
        return getattr(first, 'errors', None)


@contextmanager
def tee_console_output(log_path: str, append: bool = True):
    """将 stdout/stderr 复制到指定日志文件，同时保留原始控制台输出。

    适用场景:
        - tqdm 进度条
        - print() 输出
        - 直接写 sys.stdout / sys.stderr 的第三方库输出

    Args:
        log_path: 目标日志文件
        append: True=追加写入；False=覆盖写入
    """
    path = Path(log_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    mode = 'a' if append else 'w'
    with open(path, mode, encoding='utf-8') as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeStream(original_stdout, log_file)
        sys.stderr = _TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            try:
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = original_stdout
            sys.stderr = original_stderr
