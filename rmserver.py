from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import gc
import os
from concurrent.futures import ThreadPoolExecutor
import threading

class QwenPRMService:
    def __init__(self, model_path, device_id):
        self.model_name = model_path
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()  # 线程锁，确保并发安全
        self._load_model()
    
    def _load_model(self):
        print(f"[GPU {self.device_id}] Loading tokenizer from {self.model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        print(f"[GPU {self.device_id}] Loading model from {self.model_name}...", flush=True)
        
        # 每张卡独立加载完整模型，不使用FSDP
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        
        self.model.eval()
        print(f"[GPU {self.device_id}] Model loaded successfully", flush=True)
        
    def make_step_rewards(self,logits, token_masks):
        # print(logits.shape, flush=True)
        # print(logits, flush=True)
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
    
    def process_request(self, messages):
        """处理单个请求"""
        # 使用线程锁确保并发安全
        with self.lock:
            # 复制messages避免修改原始数据
            messages = [msg.copy() if isinstance(msg, dict) else msg for msg in messages]
            # print(messages, flush=True)
            messages[-1]['content'] = "<extra_0>".join(messages[-1]['content'])+"<extra_0>"
            conversation_str = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )

            input_ids = self.tokenizer.encode(
                conversation_str, 
                return_tensors="pt", 
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)

            step_sep_id = self.tokenizer.encode("<extra_0>")[0]
            token_masks = (input_ids == step_sep_id)
            
            # 处理模型输出：可能是tuple或BaseModelOutput
            if hasattr(outputs, 'last_hidden_state'):
                logits = outputs.last_hidden_state
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            step_reward = self.make_step_rewards(logits, token_masks)
            return step_reward
    
    def process_batch(self, messages_list):
        """批量处理多个请求，提高效率"""
        if not messages_list:
            return []
        
        # 使用线程锁确保并发安全
        with self.lock:
            # 准备所有输入（复制避免修改原始数据）
            conversation_strs = []
            for messages in messages_list:
                # 深拷贝messages避免修改原始数据
                messages_copy = [msg.copy() if isinstance(msg, dict) else msg for msg in messages]
                messages_copy[-1]['content'] = "<extra_0>".join(messages_copy[-1]['content'])+"<extra_0>"
                conversation_str = self.tokenizer.apply_chat_template(
                    messages_copy, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                conversation_strs.append(conversation_str)
            
            # 批量tokenize，使用padding
            encoded = self.tokenizer(
                conversation_strs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096  # 根据需要调整
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            step_sep_id = self.tokenizer.encode("<extra_0>")[0]
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 处理模型输出：可能是tuple或BaseModelOutput
            if hasattr(outputs, 'last_hidden_state'):
                logits = outputs.last_hidden_state
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # 处理每个样本的step rewards
            all_step_rewards = []
            for i in range(input_ids.size(0)):
                token_masks = (input_ids[i] == step_sep_id)
                step_reward = self.make_step_rewards(logits[i:i+1], token_masks.unsqueeze(0))
                all_step_rewards.extend(step_reward)
            
            return all_step_rewards


class FlaskApp:
    def __init__(self, model_service, device_id):
        self.app = Flask(__name__)
        self.model_service = model_service
        self.device_id = device_id
        self._setup_cors()
        self._register_routes()
    
    def _setup_cors(self):
        CORS(self.app, resources={
            r"/*": {
                "origins": ["*"],
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "allow_headers": ["*"]
            }
        })

        @self.app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Proxy-Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response
    
    def _register_routes(self):
        @self.app.route('/v1/step_rewards', methods=['POST'])
        def step_rewards():
            data = request.get_json()
            messages = data.get('messages', [])
            
            if not messages:
                return jsonify({"step_rewards": []})
            
            # 使用批处理提高效率
            # 如果messages是单个message的列表，直接批处理
            # 如果messages是多个message的列表，也批处理
            if isinstance(messages[0], list):
                # messages是多个message的列表
                step_reward = self.model_service.process_batch(messages)
            elif isinstance(messages[0], dict) and 'content' in messages[0]:
                # messages是单个conversation的messages列表
                # 检查是否应该批处理多个conversations
                if 'batch' in data and isinstance(data['batch'], list):
                    # 批量处理多个conversations
                    step_reward = self.model_service.process_batch(data['batch'])
                else:
                    # 单个conversation，使用批处理接口（即使只有一个）
                    step_reward = self.model_service.process_batch([messages])
            else:
                # 兼容旧格式：单个message
                step_reward = []
                for message in messages:
                    step_reward += self.model_service.process_request(message)
            
            # 定期清理显存
            gc.collect()
            torch.cuda.empty_cache()
            
            return jsonify({"step_rewards": step_reward})
            

        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy', 'device_id': self.device_id})
    
    def run(self, host='0.0.0.0', port=4997):
        print(f"[GPU {self.device_id}] Starting server on port {port}...", flush=True)
        self.app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


class LoadBalancer:
    """负载均衡器，轮询分发请求到不同的GPU服务"""
    def __init__(self, model_path, base_port=4997):
        self.model_path = model_path
        self.base_port = base_port
        self.num_gpus = torch.cuda.device_count()
        self.services = []
        self.current_gpu = 0
        self.lock = threading.Lock()
        
        # 为每张GPU创建服务
        print(f"Initializing {self.num_gpus} GPU services...", flush=True)
        for i in range(self.num_gpus):
            service = QwenPRMService(model_path, i)
            self.services.append(service)
        print("All GPU services initialized", flush=True)
    
    def get_next_service(self):
        """轮询获取下一个GPU服务"""
        with self.lock:
            service = self.services[self.current_gpu]
            self.current_gpu = (self.current_gpu + 1) % self.num_gpus
            return service


def run_single_gpu(device_id, model_path, port):
    """在单张GPU上运行服务器"""
    model_service = QwenPRMService(model_path, device_id)
    flask_app = FlaskApp(model_service, device_id)
    flask_app.run(port=port)


def main():
    model_path = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/models"
    base_port = 4997
    num_gpus = torch.cuda.device_count()
    
    print(f"Starting {num_gpus} GPU servers...", flush=True)
    print(f"Each GPU will run on ports {base_port} to {base_port + num_gpus - 1}", flush=True)
    
    # 创建负载均衡器（主服务器）
    lb = LoadBalancer(model_path, base_port)
    
    # 创建主Flask应用，使用负载均衡
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/v1/step_rewards', methods=['POST'])
    def step_rewards():
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({"step_rewards": []})
        
        # 获取下一个GPU服务
        service = lb.get_next_service()
        
        # 使用批处理提高效率
        if isinstance(messages[0], list):
            step_reward = service.process_batch(messages)
        elif isinstance(messages[0], dict) and 'content' in messages[0]:
            if 'batch' in data and isinstance(data['batch'], list):
                step_reward = service.process_batch(data['batch'])
            else:
                step_reward = service.process_batch([messages])
        else:
            step_reward = []
            for message in messages:
                step_reward += service.process_request(message)
        
        return jsonify({"step_rewards": step_reward})
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'num_gpus': num_gpus,
            'base_port': base_port
        })
    
    print(f"Starting main server on port {base_port}...", flush=True)
    app.run(host='0.0.0.0', port=base_port, debug=False, use_reloader=False, threaded=True)


if __name__ == '__main__':
    print(f"Starting Server")
    main()