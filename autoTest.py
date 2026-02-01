import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Tuple

# ===================== 基础配置 =====================
MODEL_PATH = "./out/full_sft_512.pth"  # 黑盒蒸馏/微调后的模型路径
LORA_PATH = "./out/lora/lora_huatuo_512.pth"  # 如果用了LoRA微调，填写LoRA权重路径，否则留空
TEST_DATA_PATH = "dental_choice_questions.json"
OUTPUT_REPORT_PATH = "dental_model_test_report.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 加载模型和Tokenizer =====================
def load_model_and_tokenizer():
    """加载测试用的模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 如果使用了LoRA微调，加载LoRA权重
    if LORA_PATH:
        model = PeftModel.from_pretrained(model, LORA_PATH)
    
    model.eval()  # 切换到评估模式
    return model, tokenizer

# ===================== 构建测试Prompt =====================
def build_choice_prompt(question: str, options: Dict[str, str]) -> str:
    """
    构建标准化的选择题Prompt，引导模型仅输出选项字母（A/B/C/D）
    """
    prompt = f"""
你是专业的牙科咨询机器人，请回答以下选择题，仅需输出正确选项的字母（如A、B、C、D），不要输出其他内容。

问题：{question}
选项：
"""
    # 拼接选项
    for option_letter, option_content in options.items():
        prompt += f"{option_letter}：{option_content}\n"
    
    prompt += "\n答案："
    return prompt.strip()

# ===================== 调用模型获取回答 =====================
@torch.no_grad()  # 禁用梯度计算，节省显存
def get_model_answer(model, tokenizer, prompt: str) -> str:
    """调用模型获取回答，并提取选项字母"""
    # 编码Prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(DEVICE)
    
    # 生成回答（限制仅输出1个字符，确保只返回选项字母）
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,  # 仅生成1个字符
        temperature=0.0,   # 确定性输出，避免随机
        top_p=1.0,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 解码并提取答案
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取"答案："后的内容
    answer = response.split("答案：")[-1].strip().upper()  # 转为大写，兼容小写输出
    return answer

# ===================== 批量测试并计算指标 =====================
def run_batch_test() -> Dict:
    """执行批量选择题测试，返回测试结果和指标"""
    # 1. 加载模型和数据
    model, tokenizer = load_model_and_tokenizer()
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_questions = json.load(f)
    
    # 2. 初始化测试结果
    test_results = {
        "total_questions": len(test_questions),
        "correct_count": 0,
        "incorrect_count": 0,
        "accuracy": 0.0,
        "detailed_results": []
    }
    
    # 3. 遍历测试题目
    for idx, q in enumerate(test_questions):
        print(f"测试第 {idx+1}/{len(test_questions)} 题（ID：{q['Question']}）...")
        
        # 构建Prompt
        prompt = build_choice_prompt(q["Question"], q["Options"])
        
        # 获取模型回答
        model_answer = get_model_answer(model, tokenizer, prompt)
        
        # 验证答案
        is_correct = model_answer == q["Answer"]
        if is_correct:
            test_results["correct_count"] += 1
        else:
            test_results["incorrect_count"] += 1
        
        # 记录详细结果
        test_results["detailed_results"].append({
            "question_id": q["question_id"],
            "question": q["question"],
            "options": q["options"],
            "model_answer": model_answer,
            "correct_answer": q["correct_answer"],
            "is_correct": is_correct,
            "knowledge_point": q["knowledge_point"]
        })
    
    # 4. 计算准确率
    test_results["accuracy"] = round(
        test_results["correct_count"] / test_results["total_questions"] * 100,
        2
    )
    
    # 5. 保存测试报告
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=4)
    
    return test_results

# ===================== 生成可视化测试报告（可选） =====================
def generate_human_report(test_results: Dict):
    """生成易读的人类友好型测试报告"""
    report = f"""
# 牙科聊天机器人选择题性能测试报告
## 测试概况
- 总测试题目数：{test_results['total_questions']}
- 正确数：{test_results['correct_count']}
- 错误数：{test_results['incorrect_count']}
- 整体准确率：{test_results['accuracy']}%

## 错误题目详情
"""
    # 提取错误题目
    wrong_questions = [r for r in test_results["detailed_results"] if not r["is_correct"]]
    if wrong_questions:
        for r in wrong_questions:
            report += f"""
### 题目ID：{r['question_id']}
知识点：{r['knowledge_point']}
问题：{r['question']}
选项：
"""
            for opt_letter, opt_content in r["optioOptionsns"].items():
                report += f"- {opt_letter}：{opt_content}\n"
            report += f"模型回答：{r['model_answer']}\n"
            report += f"正确答案：{r['correct_answer']}\n\n"
    else:
        report += "无错误题目，模型回答全部正确！\n"
    
    # 保存文本报告
    with open("dental_test_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("测试报告已保存为 dental_test_report.txt")

# ===================== 执行测试 =====================
if __name__ == "__main__":
    # 运行批量测试
    test_results = run_batch_test()
    
    # 打印核心指标
    print(f"\n===== 测试结果汇总 =====")
    print(f"总题目数：{test_results['total_questions']}")
    print(f"正确数：{test_results['correct_count']}")
    print(f"错误数：{test_results['incorrect_count']}")
    print(f"准确率：{test_results['accuracy']}%")
    
    # 生成人类友好型报告
    generate_human_report(test_results)