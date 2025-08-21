"""
API 路由模块
包含各种 API 端点的实现
"""

from .openai_compat import register_openai_routes
from .management import register_management_routes
from .load_balanced import register_load_balanced_routes

__all__ = [
    'register_openai_routes',
    'register_management_routes', 
    'register_load_balanced_routes'
]