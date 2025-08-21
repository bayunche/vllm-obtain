"""
API兼容性集成测试
"""

import pytest
import requests
import json
import time


class TestAPICompatibility:
    """API兼容性测试类"""
    
    BASE_URL = "http://127.0.0.1:8001"
    API_BASE = f"{BASE_URL}/v1"
    
    def test_service_health(self):
        """测试服务健康状态"""
        response = requests.get(f"{self.BASE_URL}/health")
        # 503是正常的，因为没有加载模型
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "status" in data
        
        print(f"服务状态: {data['status']} (状态码: {response.status_code})")
        if "issues" in data:
            print(f"已知问题: {data['issues']}")
            
        # 验证基本响应结构
        assert "manager_initialized" in data
        assert "engines" in data
    
    def test_models_list(self):
        """测试模型列表接口"""
        response = requests.get(f"{self.API_BASE}/models")
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        
        print(f"发现 {len(data['data'])} 个模型")
        return data["data"]
    
    def test_invalid_chat_completion(self):
        """测试无效的聊天补全请求"""
        # 测试不存在的模型
        payload = {
            "model": "non-existent-model", 
            "messages": [{"role": "user", "content": "test"}]
        }
        
        response = requests.post(f"{self.API_BASE}/chat/completions", json=payload)
        
        # 应该返回错误
        assert response.status_code >= 400
        data = response.json()
        assert "error" in data
        
        print(f"错误处理正确: {data['error']}")
    
    def test_missing_parameters(self):
        """测试缺少必需参数"""
        # 缺少 messages 参数
        payload = {"model": "test-model"}
        
        response = requests.post(f"{self.API_BASE}/chat/completions", json=payload)
        assert response.status_code >= 400
        
        data = response.json() 
        assert "error" in data
        
        print(f"参数验证正确: {data['error']}")
    
    def test_text_completion_invalid(self):
        """测试无效的文本补全"""
        payload = {
            "model": "non-existent-model",
            "prompt": "test"
        }
        
        response = requests.post(f"{self.API_BASE}/completions", json=payload)
        assert response.status_code >= 400
        
        data = response.json()
        assert "error" in data
        
        print(f"文本补全错误处理: {data['error']}")
    
    def test_cors_headers(self):
        """测试CORS头部"""
        response = requests.get(f"{self.API_BASE}/models")
        
        # 检查基本CORS头部
        assert "Access-Control-Allow-Origin" in response.headers
        
        print(f"CORS Origin设置: {response.headers.get('Access-Control-Allow-Origin')}")
        print("基本CORS配置正确")
    
    def test_management_endpoints(self):
        """测试管理接口"""
        # 测试系统状态
        response = requests.get(f"{self.API_BASE}/system/status")
        print(f"系统状态接口: {response.status_code}")
        
        # 测试配置获取
        response = requests.get(f"{self.API_BASE}/config")
        if response.status_code == 200:
            config = response.json()
            print(f"当前配置: {config.get('inference_engine', 'unknown')}")
        
        # 测试模型管理
        response = requests.get(f"{self.API_BASE}/models/list")
        print(f"模型管理接口: {response.status_code}")
    
    def test_api_response_format(self):
        """测试API响应格式"""
        # 测试模型列表响应格式
        response = requests.get(f"{self.API_BASE}/models")
        assert response.status_code == 200
        
        data = response.json()
        
        # 验证OpenAI兼容格式
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        
        for model in data["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"
        
        print("OpenAI格式兼容性正确")


def main():
    """直接运行测试"""
    test = TestAPICompatibility()
    
    print("=== API兼容性测试 ===")
    
    try:
        print("\n1. 测试服务健康状态...")
        test.test_service_health()
        
        print("\n2. 测试模型列表...")
        test.test_models_list()
        
        print("\n3. 测试错误处理...")
        test.test_invalid_chat_completion()
        test.test_missing_parameters()
        test.test_text_completion_invalid()
        
        print("\n4. 测试CORS配置...")
        test.test_cors_headers()
        
        print("\n5. 测试管理接口...")
        test.test_management_endpoints()
        
        print("\n6. 测试响应格式...")
        test.test_api_response_format()
        
        print("\n=== 所有测试完成 ===")
        print("OK - API兼容性测试通过")
        
    except Exception as e:
        print(f"\nERROR - 测试失败: {e}")
        raise


if __name__ == "__main__":
    main()