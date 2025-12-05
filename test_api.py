#!/usr/bin/env python3
"""
测试 rmserver API 接口
接口地址:  http://10.102.98.149:4997/v1/step_rewards
"""

import requests
import json
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


class RMServerTester:
    def __init__(self, base_url: str = " http://10.102.98.149:4997"):
        self.base_url = base_url
        self.step_rewards_url = f"{base_url}/v1/step_rewards"
        self.health_url = f"{base_url}/health"
    
    def health_check(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ 健康检查通过: {data}")
                return True
            else:
                print(f"✗ 健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ 健康检查异常: {e}")
            return False
    
    def test_single_request(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """测试单个请求"""
        payload = {
            "messages": messages
        }
        
        print(f"\n发送请求:")
        print(f"  URL: {self.step_rewards_url}")
        print(f"  Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        try:
            start_time = time.time()
            response = requests.post(
                self.step_rewards_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            elapsed_time = time.time() - start_time
            
            print(f"\n响应状态码: {response.status_code}")
            print(f"响应时间: {elapsed_time:.2f}秒")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ 请求成功")
                print(f"响应数据:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return {"success": True, "data": result, "time": elapsed_time}
            else:
                print(f"✗ 请求失败")
                print(f"响应内容: {response.text}")
                return {"success": False, "error": response.text, "time": elapsed_time}
                
        except requests.exceptions.Timeout:
            print(f"✗ 请求超时")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            print(f"✗ 请求异常: {e}")
            return {"success": False, "error": str(e)}
    
    def test_batch_request(self, messages_list: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """测试批量请求"""
        payload = {
            "messages": messages_list
        }
        
        print(f"\n发送批量请求:")
        print(f"  URL: {self.step_rewards_url}")
        print(f"  批量大小: {len(messages_list)}")
        
        try:
            start_time = time.time()
            response = requests.post(
                self.step_rewards_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            elapsed_time = time.time() - start_time
            
            print(f"\n响应状态码: {response.status_code}")
            print(f"响应时间: {elapsed_time:.2f}秒")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ 批量请求成功")
                print(f"返回的 step_rewards 数量: {len(result.get('step_rewards', []))}")
                return {"success": True, "data": result, "time": elapsed_time}
            else:
                print(f"✗ 批量请求失败")
                print(f"响应内容: {response.text}")
                return {"success": False, "error": response.text, "time": elapsed_time}
                
        except Exception as e:
            print(f"✗ 批量请求异常: {e}")
            return {"success": False, "error": str(e)}
    
    def test_concurrent_requests(self, messages: List[Dict[str, Any]], num_requests: int = 10) -> Dict[str, Any]:
        """测试并发请求"""
        print(f"\n发送 {num_requests} 个并发请求:")
        print(f"  URL: {self.step_rewards_url}")
        
        def send_request(i):
            payload = {"messages": messages}
            try:
                start_time = time.time()
                response = requests.post(
                    self.step_rewards_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                elapsed_time = time.time() - start_time
                return {
                    "success": response.status_code == 200,
                    "time": elapsed_time,
                    "request_id": i,
                    "status_code": response.status_code
                }
            except Exception as e:
                return {
                    "success": False,
                    "time": 0,
                    "request_id": i,
                    "error": str(e)
                }
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(send_request, i) for i in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        total_time = time.time() - start_time
        
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        
        print(f"\n并发测试结果:")
        print(f"  总请求数: {num_requests}")
        print(f"  成功: {len(successful)}")
        print(f"  失败: {len(failed)}")
        print(f"  总耗时: {total_time:.2f}秒")
        
        if successful:
            times = [r["time"] for r in successful]
            print(f"  平均响应时间: {sum(times)/len(times):.2f}秒")
            print(f"  最快响应: {min(times):.2f}秒")
            print(f"  最慢响应: {max(times):.2f}秒")
            print(f"  吞吐量: {len(successful)/total_time:.2f} 请求/秒")
        
        if failed:
            print(f"  失败请求详情:")
            for f in failed[:5]:  # 只显示前5个失败
                print(f"    请求 {f['request_id']}: {f.get('error', 'Unknown error')}")
        
        return {
            "success": len(successful) > 0,
            "total_requests": num_requests,
            "successful": len(successful),
            "failed": len(failed),
            "total_time": total_time,
            "results": results
        }


def main():
    """主测试函数"""
    print("=" * 60)
    print("RMServer API 测试工具")
    print("=" * 60)
    
    tester = RMServerTester()
    
    # 1. 健康检查
    print("\n[1] 健康检查")
    print("-" * 60)
    if not tester.health_check():
        print("服务器不可用，退出测试")
        return
    
    # 2. 测试单个请求 - 简单数学问题
    print("\n[2] 测试单个请求 - 简单数学问题")
    print("-" * 60)
    messages1 = [
        {
            "role": "user",
            "content": "计算 2 + 2"
        },
        {
            "role": "assistant",
            "content": [
                "计算 2 + 2",
                "答案是 4"
            ]
        }
    ]
    result1 = tester.test_single_request(messages1)
    
    # 3. 测试单个请求 - 复杂数学问题
    print("\n[3] 测试单个请求 - 复杂数学问题")
    print("-" * 60)
    messages2 = [
        {
            "role": "user",
            "content": "求解方程 x^2 + 5x + 6 = 0"
        },
        {
            "role": "assistant",
            "content": [
                "求解方程 x^2 + 5x + 6 = 0",
                "使用因式分解: (x+2)(x+3) = 0",
                "所以 x = -2 或 x = -3"
            ]
        }
    ]
    result2 = tester.test_single_request(messages2)
    
    # 4. 测试批量请求
    print("\n[4] 测试批量请求")
    print("-" * 60)
    batch_messages = [
        [
            {
                "role": "user",
                "content": "计算 1+1"
            },
            {
                "role": "assistant",
                "content": ["计算 1+1", "答案是 2"]
            }
        ],
        [
            {
                "role": "user",
                "content": "计算 3*4"
            },
            {
                "role": "assistant",
                "content": ["计算 3*4", "答案是 12"]
            }
        ],
        [
            {
                "role": "user",
                "content": "计算 10/2"
            },
            {
                "role": "assistant",
                "content": ["计算 10/2", "答案是 5"]
            }
        ]
    ]
    result3 = tester.test_batch_request(batch_messages)
    
    # 5. 性能测试
    print("\n[5] 性能测试 - 连续发送多个请求")
    print("-" * 60)
    test_messages = [[
        {
            "role": "user",
            "content": "求解方程 x^2 + 5x + 6 = 0"
        },
        {
            "role": "assistant",
            "content": [
                "求解方程 x^2 + 5x + 6 = 0",
                "使用因式分解: (x+2)(x+3) = 0",
                "所以 x = -2 或 x = -3"
            ]
        }
    ] for _ in range(768)]
    
    times = []
    for i, msg in enumerate(test_messages, 1):
        print(f"\n请求 {i}/768...")
        result = tester.test_single_request(msg)
        if result.get("success"):
            times.append(result.get("time", 0))
    
    if times:
        print(f"\n性能统计:")
        print(f"  总请求数: {len(times)}")
        print(f"  平均响应时间: {sum(times)/len(times):.2f}秒")
        print(f"  最快响应: {min(times):.2f}秒")
        print(f"  最慢响应: {max(times):.2f}秒")
    
    # 6. 并发测试 - 测试负载均衡
    print("\n[6] 并发测试 - 测试负载均衡和GPU并发能力")
    print("-" * 60)
    concurrent_messages = [
        {
            "role": "user",
            "content": "计算 2 + 2"
        },
        {
            "role": "assistant",
            "content": ["计算 2 + 2", "答案是 4"]
        }
    ]
    # 获取GPU数量（从health接口）
    try:
        health_response = requests.get(tester.health_url, timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            num_gpus = health_data.get('num_gpus', 6)  # 默认6张卡
            print(f"检测到 {num_gpus} 张GPU，将发送 {768} 个并发请求进行测试")
            num_requests = 768
        else:
            num_requests = 20
    except:
        num_requests = 20
    
    concurrent_result = tester.test_concurrent_requests(concurrent_messages, num_requests=num_requests)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

