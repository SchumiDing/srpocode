#!/usr/bin/env python3
"""
Concurrent stress test for /v1/step_rewards
Usage:
    python stress_step_reward.py \
        --host 127.0.0.1 \
        --port 4997 \
        -c 64 \
        -n 1000
"""
import argparse, asyncio, json, time, sys
from typing import List, Dict
import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
import random

# --------------------  payload  --------------------
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
QUERY  = ("Sue lives in a fun neighborhood…")   # 太长，省略，用你上面那段即可
RESPONSE = [
    "To find out how many more pink plastic flamingos…",
    "On Saturday, they take back one third…",
    "On Sunday, the neighbors add another 18…",
    "To find the difference…"
]

def build_payload() -> Dict:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUERY},
        {"role": "assistant", "content": "<extra_0>".join(RESPONSE) + "<extra_0>"}
    ]
    return {"messages": [messages]}     # 注意接口是 List[conversation]

# --------------------  client  --------------------
async def request_one(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> float:
    """返回耗时（秒）"""
    payload = build_payload()
    start = time.perf_counter()
    async with sem:
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    print(await resp.text())
                    return -1.0
                js = await resp.json()
                # ---- 随机 5 % 打印返回内容，验证正确性 ----
                if random.random() < 0.1:
                    print("[sample]", js.get("step_rewards"))
                return time.perf_counter() - start
        except Exception as e:
            print(e)
            return -1.0

async def stress_test(host: str, port: int, concurrency: int, total: int):
    url = f"http://{host}:{port}/v1/step_rewards"
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, ssl=False),
            timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        tasks = [request_one(session, url, sem) for _ in range(total)]
        latencies = []
        ok = 0
        for coro in tqdm.as_completed(tasks, total=total):
            lat = await coro
            if lat > 0:
                latencies.append(lat)
                ok += 1

    # --------------------  report  --------------------
    lat = np.array(latencies) * 1000          # ms
    print(f"\nSuccessful : {ok}/{total}")
    print(f"QPS        : {len(lat)/lat.sum()*1000:.2f}")
    print(f"Mean       : {lat.mean():.2f} ms")
    for p in [50, 90, 95, 99]:
        print(f"P{p}        : {np.percentile(lat, p):.2f} ms")

# --------------------  main  --------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="100.97.108.71")
    parser.add_argument("--port", type=int, default=4997)
    parser.add_argument("-c", "--concurrency", type=int, default=64)
    parser.add_argument("-n", "--number", type=int, default=1000)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(stress_test(args.host, args.port, args.concurrency, args.number))