"""
LGFF 通用日志工具，提供轻量化的 logger 初始化函数。

- get_logger(name, log_file=None): 获取一个带终端输出的 logger。
- setup_logger(log_file, name="lgff.train"): 在脚本入口调用，统一配置。

特性：
1. 防止重复添加 Handler。
2. 阻断日志冒泡，防止多层级 Logger 导致重复打印。
3. 自动创建日志目录。
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    获取一个名为 `name` 的 logger，如果首次创建则绑定终端和文件输出。
    """
    logger = logging.getLogger(name)

    # 1. 如果该 logger 已经有 handler，直接返回，避免重复添加
    if logger.handlers:
        return logger

    # 2. 基础设置
    logger.setLevel(logging.INFO)
    # [关键修改] 禁止日志向父级传播，防止在父子 logger 都有 handler 时重复打印
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3. 终端输出 (Stdout)
    # 使用 sys.stdout 显式指定，防止部分环境输出到 stderr
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 4. 文件输出（可选）
    if log_file is not None:
        path = Path(log_file)
        # 稳健性：即使多进程并发创建目录也不会报错
        path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_logger(
    log_file: str,
    name: str = "lgff.train",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    在脚本入口调用一次，用于初始化主 logger。
    """
    logger = get_logger(name, log_file=log_file)
    logger.setLevel(level)
    return logger


__all__ = ["get_logger", "setup_logger"]