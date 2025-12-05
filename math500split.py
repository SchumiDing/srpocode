hfdatasetPath = "math500split"

from datasets import load_dataset

dataset = load_dataset(hfdatasetPath)

print(dataset)

import pandas as pd
import numpy as np

np.random.seed(42)

import re
pattern1 = r'\\boxed\{(.*?)\}'
pattern2 = r'$\\boxed (.*?)$'
data_source = "math500split"
# print(pattern1)
# print(pattern2)
train_data = []
val_data = []
datas = []
import time
from openai import OpenAI
import json
import asyncio
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import queue


train_dataset = dataset["train"]
val_dataset = dataset["test"]

def process_row(args):
    i, row_data = args
    problem = row_data["problem"]
    solution = row_data["solution"]
    ground_truth = row_data["answer"]
    instruction_following = 'Let\'s think step by step and put the final answer in the \\boxed{} tag.'
    data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": problem+"\n"+instruction_following,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "answer": solution,
                "question": problem,
                "subject": row_data["subject"],
                "level": row_data["level"],
            },
        }
    return data

def progress_monitor(progress_queue, total):
    """监控进度并更新进度条"""
    pbar = tqdm(total=total, desc="Processing", unit="item", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    completed = 0
    start_time = time.time()
    
    while completed < total:
        try:
            # 从队列中获取完成的任务
            result = progress_queue.get(timeout=1)
            if result is None:  # 结束信号
                break
            completed += 1
            pbar.update(1)
            
            # 更新进度条显示信息
            elapsed = time.time() - start_time
            if completed > 0:
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                pbar.set_postfix({
                    'rate': f'{rate:.1f}it/s',
                    'eta': f'{eta:.0f}s'
                })
                
        except queue.Empty:
            continue
    
    pbar.close()

def main(dataset):
    print(f"Processing {len(dataset)} items, using {mp.cpu_count()} CPU cores")
    
    # 准备数据
    data_args = [(i, dataset[i]) for i in range(len(dataset))]
    
    # 创建进度队列
    progress_queue = queue.Queue()
    
    # 启动进度监控线程
    progress_thread = threading.Thread(
        target=progress_monitor, 
        args=(progress_queue, len(dataset))
    )
    progress_thread.daemon = True  # 设置为守护线程
    progress_thread.start()
    
    results = []
    completed_count = 0
    
    # 使用ProcessPoolExecutor进行并行处理
    # 限制最大工作进程数，避免过多进程导致资源竞争
    max_workers = min(mp.cpu_count(), 8)  # 最多使用8个进程
    print(f"Using {max_workers*16} parallel processes")
    
    with ProcessPoolExecutor(max_workers=max_workers*16) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(process_row, args): i 
            for i, args in enumerate(data_args)
        }
        
        # 收集结果
        for future in as_completed(future_to_index):
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                progress_queue.put(1)  # 通知进度更新
                
                    
            except Exception as e:
                print(f"Error processing item {future_to_index[future]}: {e}")
                completed_count += 1
                progress_queue.put(1)  # 即使出错也要更新进度
    
    # 发送结束信号
    progress_queue.put(None)
    progress_thread.join()
    
    print(f"处理完成，共处理 {len(results)} 个有效结果")
    return results
if __name__ == "__main__":
    
    train_data = main(train_dataset)
    val_data = main(val_dataset)
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_df.to_parquet("data/math500split_train.parquet")
    val_df.to_parquet("data/math500split_val.parquet")
    
    json.dump(train_data, open("data/math500split_train.json", "w"), indent=4)
    json.dump(val_data, open("data/math500split_val.json", "w"), indent=4)