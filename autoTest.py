import json
import torch
import os
import sys
# 加入model文件夹到Python路径，确保能导入自定义模型
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model_minimind import MiniMindModel  # 假设自定义模型类为MiniMindModel
from model_lora import load_lora_weights   # 假设LoRA加载函数为load_lora_weights

# ===================== 基础配置（适配你的目录结构） =====================
MODEL_WEIGHT_PATH = "./out/full_sft_512.pth"  # 非HF格式的权重文件
LORA_WEIGHT_PATH = "./out/lora/lora_huatuo_512.pth"  # LoRA权重文件
TOKENIZER_CONFIG_PATH = "./model/tokenizer_config.json"  # 自定义tokenizer配置
TOKENIZER_VOCAB_PATH = "./model/tokenizer.json"  # 自定义tokenizer词表
TEST_DATA_PATH = "dental_choice_questions.json"
OUTPUT_REPORT_PATH = "dental_model_test_report.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 加载自定义Tokenizer（适配非HF格式） =====================
class CustomTokenizer:
    """适配非HuggingFace格式的自定义Tokenizer"""
    def __init__(self, vocab_path, config_path):
        self.vocab = self._load_vocab(vocab_path)
        self.config = self._load_config(config_path)
        # 补充基础token定义（需和你的模型训练时一致）
        self.eos_token = "<|im_end|>"
        self.pad_token = "<|endoftext|>"
        self.eos_token_id = self.vocab.get(self.eos_token, 2)
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.vocab_size = len(self.vocab)
        # 反向映射（id->token）
        self.id2token = {v: k for k, v in self.vocab.items()}

    def _load_vocab(self, vocab_path):
        """加载自定义词表（JSON格式）"""
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # 兼容不同格式的词表（dict/列表）
        if isinstance(vocab, list):
            return {token: idx for idx, token in enumerate(vocab)}
        elif isinstance(vocab, dict):
            return vocab
        else:
            raise ValueError("词表格式不支持，需为JSON dict/列表")

    def _load_config(self, config_path):
        """加载tokenizer配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def encode(self, text, return_tensors="pt", truncation=True, max_length=1024, padding=True):
        """自定义编码逻辑（模拟HF Tokenizer的输入输出格式）"""
        # 简单分词（需替换为你模型实际的分词逻辑，如BPE/字粒度）
        tokens = list(text)  # 示例：字粒度分词（根据你的模型调整）
        # 截断
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        # 转token id
        input_ids = [self.vocab.get(token, self.vocab.get("<unk>", 1)) for token in tokens]
        # padding
        if padding:
            padding_length = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * padding_length
        # 转tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        # 模拟HF的返回格式
        return {"input_ids": input_ids.to(DEVICE), "attention_mask": (input_ids != self.pad_token_id).to(DEVICE)}

    def decode(self, input_ids, skip_special_tokens=True):
        """自定义解码逻辑"""
        # 处理tensor/list
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy().tolist()
        # 展平（处理batch维度）
        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        # 解码token id到文本
        tokens = []
        for idx in input_ids:
            token = self.id2token.get(idx, "<unk>")
            # 跳过特殊token
            if skip_special_tokens and token in [self.eos_token, self.pad_token]:
                continue
            tokens.append(token)
        return "".join(tokens)

def load_custom_tokenizer():
    """加载自定义Tokenizer"""
    try:
        tokenizer = CustomTokenizer(TOKENIZER_VOCAB_PATH, TOKENIZER_CONFIG_PATH)
        return tokenizer
    except Exception as e:
        print(f"Tokenizer加载失败：{str(e)}")
        raise

# ===================== 加载自定义PyTorch模型（非HF格式） =====================
def load_custom_model():
    """加载自定义MiniMind模型 + 权重"""
    try:
        # 1. 初始化模型（需和model_minimind.py中模型定义一致）
        model = MiniMindModel(
            vocab_size=load_custom_tokenizer().vocab_size,
            hidden_size=512,  # 替换为你模型的实际参数
            num_layers=8,     # 替换为你模型的实际参数
            num_heads=8       # 替换为你模型的实际参数
        ).to(DEVICE)
        
        # 2. 加载基础权重（非HF格式的.pth文件）
        state_dict = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)
        # 兼容多GPU保存的权重（去除module.前缀）
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        
        # 3. 加载LoRA权重（若有）
        if os.path.exists(LORA_WEIGHT_PATH):
            model = load_lora_weights(model, LORA_WEIGHT_PATH, DEVICE)
        
        # 4. 模型设为评估模式
        model.eval()
        return model
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        print("请检查：1.权重文件路径 2.模型初始化参数和训练时一致 3.LoRA加载逻辑")
        raise

# ===================== 构建测试Prompt（复用逻辑） =====================
def build_choice_prompt(question: str, options: dict) -> str:
    """构建标准化选择题Prompt"""
    prompt = f"""你是专业的牙科咨询机器人，请回答以下选择题，仅需输出正确选项的字母（如A、B、C、D），不要输出其他内容。

问题：{question}
选项：
"""
    for letter, content in options.items():
        prompt += f"{letter}：{content}\n"
    prompt += "\n答案："
    return prompt.strip()

# ===================== 模型推理（适配自定义模型） =====================
@torch.no_grad()
def get_model_answer(model, tokenizer, prompt: str) -> str:
    """调用自定义模型获取回答"""
    try:
        # 1. 编码Prompt
        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        
        # 2. 模型推理（自定义generate逻辑，适配非HF模型）
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        max_new_tokens = 2  # 仅生成答案字母
        
        # 逐token生成（模拟HF的generate）
        generated_ids = input_ids
        for _ in range(max_new_tokens):
            # 前向传播（需和你模型的forward逻辑一致）
            outputs = model(
                input_ids=generated_ids,
                attention_mask=attention_mask
            )
            # 获取最后一个token的logits
            next_token_logits = outputs.logits[:, -1, :]
            # 贪心解码（取概率最大的token）
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            # 拼接生成的token
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            # 更新attention_mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=DEVICE)
            ], dim=-1)
        
        # 3. 解码并提取答案
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        answer_part = response.split("答案：")[-1].strip().upper()
        model_answer = answer_part[0] if answer_part and answer_part[0] in ["A","B","C","D"] else "未知"
        return model_answer
    except Exception as e:
        print(f"推理失败：{str(e)}")
        return "未知"

# ===================== 批量测试（复用逻辑） =====================
def run_batch_test():
    """执行批量测试"""
    # 1. 加载模型和Tokenizer
    model = load_custom_model()
    tokenizer = load_custom_tokenizer()
    
    # 2. 加载测试数据
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_questions = json.load(f)
    
    # 3. 初始化测试结果
    test_results = {
        "total": len(test_questions),
        "correct": 0,
        "incorrect": 0,
        "accuracy": 0.0,
        "details": []
    }
    
    # 4. 遍历测试
    for idx, q in enumerate(test_questions):
        q_id = q.get("question_id", idx+1)
        question = q.get("question", "")
        options = q.get("options", {})
        correct_ans = q.get("correct_answer", "").upper()
        
        if not question or not options or not correct_ans:
            test_results["incorrect"] += 1
            continue
        
        # 构建Prompt并推理
        prompt = build_choice_prompt(question, options)
        model_ans = get_model_answer(model, tokenizer, prompt)
        
        # 统计结果
        is_correct = model_ans == correct_ans
        if is_correct:
            test_results["correct"] += 1
        else:
            test_results["incorrect"] += 1
        
        # 记录详情
        test_results["details"].append({
            "question_id": q_id,
            "question": question,
            "options": options,
            "model_answer": model_ans,
            "correct_answer": correct_ans,
            "is_correct": is_correct
        })
    
    # 5. 计算准确率
    if test_results["total"] > 0:
        test_results["accuracy"] = round(test_results["correct"] / test_results["total"] * 100, 2)
    
    # 6. 保存报告
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=4)
    
    return test_results

# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        results = run_batch_test()
        print(f"\n测试完成！")
        print(f"总题数：{results['total']} | 正确：{results['correct']} | 错误：{results['incorrect']}")
        print(f"准确率：{results['accuracy']}%")
        print(f"报告已保存至：{OUTPUT_REPORT_PATH}")
    except Exception as e:
        print(f"测试失败：{str(e)}")
        import traceback
        traceback.print_exc()