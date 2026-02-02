import json
import torch
import os
import sys
import argparse

# 加入项目根目录和model目录到Python路径（适配代码仓结构）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))

# 复用eval_llm_medical.py中用到的核心导入（与代码仓保持一致）
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed
# ===================== 基础配置（与eval_llm_medical.py对齐，适配代码仓） =====================
DEFAULT_CONFIG = {
    "model_path": "./model",  # 模型结构/分词器配置目录
    "checkpoint_path": "./out/full_sft_512.pth",  # 微调权重文件
    "lora_checkpoint_path": "./out/lora/lora_medical_mental_512.pth",  # LoRA权重（可选）
    "test_data_path": "./dataset/mental.jsonl",
    "output_report_path": "dental_model_test_report.json",
    "max_seq_len": 1024,
    "batch_size": 1,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

 
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(args.device), tokenizer


def build_choice_prompt(question: str, options: str | dict) -> str:
    """
    构建标准化选择题Prompt，兼容dict和字符串格式的选项，支持A-E选项
    :param question: 选择题题干
    :param options: 选项（dict格式：{"A": "xxx"} 或 字符串格式："A xxx\nB xxx"）
    :return: 格式化后的Prompt
    """
    # 核心：统一将options转为标准的「字母：内容」格式字符串
    standard_options = ""
    
    # 场景1：options是dict（键为A/B/C/D/E，值为选项内容）
    if isinstance(options, dict):
        for opt_letter, opt_content in sorted(options.items()):  # sorted保证A-E顺序不乱
            standard_options += f"{opt_letter}：{opt_content.strip()}\n"
    
    # 场景2：options是字符串（如你给出的原始格式）
    elif isinstance(options, str):
        # 按行拆分字符串，逐行处理
        option_lines = options.strip().split("\n")
        for line in option_lines:
            line = line.strip()
            if not line:
                continue
            # 提取选项字母（开头第一个字符，如A/B/C/E）
            opt_letter = line[0].upper()
            # 提取选项内容（去除字母后的部分，处理空格/顿号）
            opt_content = line[1:].strip().lstrip("：").lstrip(".").lstrip(" ").strip()
            standard_options += f"{opt_letter}：{opt_content}\n"
    
    # 构建最终Prompt，优化引导语（明确支持A-E）
    prompt = f"""请回答以下选择题，仅需输出正确选项的字母（如A、B、C、D、E），不要输出其他内容，无需额外解释。
    问题：{question.strip()}
    选项：
    {standard_options.strip()}

    答案："""
    
    return prompt.strip()

# ===================== 复用eval_llm_medical.py推理逻辑：获取模型回答 =====================
@torch.no_grad()
def get_model_answer(model, tokenizer, prompt, config):
    """
    复用eval_llm_medical.py的贪心推理逻辑，适配自定义模型和BPE Tokenizer
    """
    # 1. BPE编码（与eval_llm_medical.py一致，使用自定义tokenizer的encode）
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=config["max_seq_len"]
    )
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(config["device"])
    attention_mask = torch.ones_like(input_ids).to(config["device"])

    # 2. 贪心生成（仅生成1-2个字符，确保只返回选项字母，与eval_llm_medical.py推理逻辑对齐）
    generated_ids = input_ids
    max_new_tokens = 2  # 防止截断，兼容轻微冗余输出
    for _ in range(max_new_tokens):
        # 前向传播（复用模型的forward逻辑，与eval_llm_medical.py一致）
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            labels=None
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # 取最后一个token的logits，贪心解码
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # 终止条件：生成eos token则停止
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # 拼接生成的token
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), dtype=torch.long, device=config["device"])
        ], dim=-1)

    # 3. BPE解码（与eval_llm_medical.py一致，跳过特殊token）
    response = tokenizer.decode(
        generated_ids[0].cpu().numpy(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False  # 保留BPE解码原始结果，避免字母错乱
    )

    # 4. 提取有效答案（仅保留A/B/C/D）
    answer_part = response.split("答案：")[-1].strip().upper()
    model_answer = ""
    for char in answer_part:
        if char in ["A", "B", "C", "D","E"]:
            model_answer = char
            break
    return model_answer if model_answer else "未知"

 # ===================== 2. 加载测试数据（适配 jsonl 文件） =====================
 # 文件是对话聊天格式
def batch_extract_qa_from_jsonl(jsonl_path: str, output_path: str = "extracted_qa.jsonl"):
    """
    批量处理JSONL文件，提取每一行的question和answer并保存
    """
    with open(jsonl_path, "r", encoding="utf-8") as in_f, open(output_path, "w", encoding="utf-8") as out_f:
        for line_num, line in enumerate(in_f, 1):
            line = line.strip()
            if not line:
                continue
            qa_result = extract_qa_from_conversation(line)
            # 保存批量提取结果
            out_f.write(json.dumps(qa_result, ensure_ascii=False) + "\n")
    print(f"✅ 批量提取完成，结果保存至：{output_path}")

# 调用批量处理函数
batch_extract_qa_from_jsonl("./sft_mini_512_with_choice_train.jsonl")

 # ===================== 2. 加载测试数据（适配 jsonl 文件） =====================

 # 文件是cmexam 里的Question。Answer 这种格式
def jsonload(jsonl_path):
    if not os.path.exists(config["test_data_path"]):
        raise FileNotFoundError(f"测试数据文件不存在：{config['test_data_path']}")

    # 初始化测试题目列表
    test_questions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        # 逐行读取 jsonl 文件
        for line_num, line in enumerate(f, 1):
            # 去除行首尾空白字符（空格、换行、制表符等）
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            try:
                # 解析单行 JSON 对象
                question_obj = json.loads(line)
                # 将解析后的对象加入列表
                test_questions.append(question_obj)
            except json.JSONDecodeError as e:
                # 捕获单行解析错误，给出友好提示，不中断整体加载
                print(f"⚠️  第 {line_num} 行 JSON 格式错误，跳过该行：{e}")

    # 验证是否加载到有效数据
    if not test_questions:
    raise ValueError(f"jsonl 文件中无有效测试数据：{jsonl_path}")

    print(f"✅ 测试数据加载完成，共 {len(test_questions)} 道有效题目")
    return test_questions
# ===================== 批量测试（保留原业务逻辑，适配新的模型/Tokenizer） =====================
def run_batch_test(config,args):
    """
    执行批量选择题测试，生成统计报告
    """
    # 1. 加载模型和Tokenizer（复用验证过的逻辑）
    model, tokenizer = init_model(args)   
    test_questions = jsonload(config)
    # 3. 初始化测试结果
    test_results = {
        "total_questions": len(test_questions),
        "correct_count": 0,
        "incorrect_count": 0,
        "accuracy": 0.0,
        "detailed_results": []
    }

    # 4. 遍历执行测试
    for idx, q in enumerate(test_questions):
        # 提取题目字段（兼容json字段，提高容错性）
        q_id = q.get("Question", idx + 1)
        question = q.get("Question", "")
        options = q.get("Options", {})
        correct_answer = q.get("Answer", "").upper()

        # 跳过无效题目
        if not question or not options or not correct_answer or correct_answer not in ["A", "B", "C", "D","E"]:
            print(f"⚠️  跳过第 {idx+1} 题（ID：{q_id}）：字段缺失或无效")
            test_results["incorrect_count"] += 1
            continue

        # 打印进度
        print(f"📝 测试第 {idx+1}/{len(test_questions)} 题（ID：{q_id}）")

        # 5. 构建Prompt并获取模型回答
        prompt = build_choice_prompt(question, options)
        model_answer = get_model_answer(model, tokenizer, prompt, config)

        # 6. 统计结果
        is_correct = (model_answer == correct_answer) and (model_answer != "未知")
        if is_correct:
            test_results["correct_count"] += 1
        else:
            test_results["incorrect_count"] += 1

        # 7. 记录详细结果
        test_results["detailed_results"].append({
            "question_id": q_id,
            "question": question,
            "options": options,
            "model_answer": model_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        })

    # 8. 计算准确率（避免除零错误）
    if test_results["total_questions"] > 0:
        test_results["accuracy"] = round(
            (test_results["correct_count"] / test_results["total_questions"]) * 100,
            2
        )

    # 9. 保存测试报告
    with open(config["output_report_path"], "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=4)
    print(f"✅ 测试报告已保存至：{config['output_report_path']}")

    return test_results

# ===================== 生成人类友好型报告（保留原逻辑，修复字段错误） =====================
def generate_human_report(test_results, config):
    """
    生成易读的文本格式测试报告
    """
    report_path = "dental_test_report.txt"
    report = f"""# 牙科聊天机器人选择题性能测试报告
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
            report += f"""### 题目ID：{r['question_id']}
问题：{r['question']}
选项：
"""

            report += f"模型回答：{r['model_answer']}\n"
            report += f"正确答案：{r['correct_answer']}\n\n"
    else:
        report += "无错误题目，模型回答全部正确！\n"

    # 保存文本报告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ 人类友好型报告已保存至：{report_path}")

# ===================== 主函数（适配代码仓风格，支持命令行参数） =====================
if __name__ == "__main__":
    # 构建命令行参数（与eval_llm_medical.py风格一致）
    parser = argparse.ArgumentParser(description="牙科模型选择题批量测试（复用eval_llm_medical.py逻辑）")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument("--model_path", type=str, default=DEFAULT_CONFIG["model_path"], help="模型结构目录")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CONFIG["checkpoint_path"], help="基础权重文件路径")
    parser.add_argument("--lora_checkpoint_path", type=str, default=DEFAULT_CONFIG["lora_checkpoint_path"], help="LoRA权重文件路径")
   
   
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='lora_medical_mental', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
   
   
   
    parser.add_argument("--test_data_path", type=str, default=DEFAULT_CONFIG["test_data_path"], help="测试数据JSON路径")
    parser.add_argument("--output_report_path", type=str, default=DEFAULT_CONFIG["output_report_path"], help="输出JSON报告路径")
   
   
    args = parser.parse_args()

    # 合并配置
    run_config = DEFAULT_CONFIG.copy()
    run_config.update(vars(args))

    try:
        # 执行批量测试
        test_results = run_batch_test(run_config,args)

        # 生成可视化报告
        generate_human_report(test_results, run_config)

        # 打印汇总结果
        print("\n" + "="*50)
        print("📊 测试结果汇总")
        print("="*50)
        print(f"总题目数：{test_results['total_questions']}")
        print(f"正确数：{test_results['correct_count']}")
        print(f"错误数：{test_results['incorrect_count']}")
        print(f"整体准确率：{test_results['accuracy']}%")
        print("="*50)

    except Exception as e:
        print(f"❌ 测试执行失败：{str(e)}")
        import traceback
        traceback.print_exc()