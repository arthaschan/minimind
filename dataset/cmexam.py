import json

# 转换cmexam的数据格式为本项目的格式。
# cmexam 的数据是选择题，格式为： {"Question":"上消化道出血可单纯表现为呕血或黑便，也可两者兼有，这取决于","Options":"A 原发病\nB 出血部位\nC 出血量\nD 在胃内停留时间\nE 以上均非","Answer":"C","Explanation":"上消化道出血表现为呕血还是便血主要取决于出血的速度和出血量的多少（C对），而出血部位（B错）高低、原发病部位（A错）是相对次要的。血液在胃内停留的时间（D错）主要决定呕血或黑便的颜色深浅，时间越久，颜色越深。"}
# 转换为：
# 定义输入输出文件路径# "conversations": [
#     {
#         "role": "user",
#         "content": question_content
#     },
#     {
#         "role": "assistant",
#         "content": answer_content
#     }
import json
import os

def convert_choice_to_sft(choice_data_path, output_sft_path, sft_format="conversations"):
    """
    将口腔选择题数据集转换为SFT格式的JSONL文件
    :param choice_data_path: 原始选择题数据路径（json/jsonl）
    :param output_sft_path: 转换后的SFT JSONL输出路径
    :param sft_format: SFT数据格式（conversations / instruction）
    """
    # 1. 加载原始选择题数据
    choice_questions = []
    if choice_data_path.endswith(".json"):
        with open(choice_data_path, "r", encoding="utf-8") as f:
            choice_questions = json.load(f)
    elif choice_data_path.endswith(".jsonl"):
        with open(choice_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                choice_questions.append(json.loads(line))
    else:
        raise ValueError("仅支持json或jsonl格式的原始选择题数据")
    
    # 2. 转换为SFT格式
    sft_samples = []
    for q in choice_questions:
        # 提取核心字段（容错处理）
        question = q.get("Question", "").strip()
        options = q.get("Options", {})
        #correct_answer = q.get ("Answer", "").strip().upper()

        # 处理correct_answer：核心包装，避免strip()异常
        correct_answer = q.get("Answer", "")
        # 步骤1：非字符串类型转为字符串（None→空字符串，数字/其他→转为对应字符串）
        if not isinstance(correct_answer, str):
            correct_answer = str(correct_answer) if correct_answer is not None else ""
        # 步骤2：去空白 + 转大写（适配A-E选项字母）
        correct_answer = correct_answer.strip().upper()
        
        # 3. 过滤无效样本（任一核心字段为空则跳过）
        if not question or not options or not correct_answer:
            print(f"⚠️  跳过无效样本：缺少核心字段，question=“{question}")
            continue
        
        # 3. 构建用户输入（Prompt格式，与测试时对齐）
        user_content = """请回答以下选择题，仅需输出正确选项的字母（如A、B、C、D、E），不要输出其他内容，无需额外解释。
问题：{question}
选项：
{options_str}"""
        
        # 构建标准化选项字符串
        options_str = ""
        if isinstance(options, dict):
            for opt_letter, opt_content in sorted(options.items()):
                options_str += f"{opt_letter}：{opt_content.strip()}\n"
        elif isinstance(options, str):
            option_lines = options.strip().split("\n")
            for line in option_lines:
                line = line.strip()
                if not line:
                    continue
                opt_letter = line[0].upper()
                opt_content = line[1:].strip().lstrip("：").lstrip(".").lstrip(" ").strip()
                options_str += f"{opt_letter}：{opt_content}\n"
        else:
            continue
        
        # 填充用户输入内容
        user_content = user_content.format(question=question, options_str=options_str.strip())
        
        # 4. 构建不同格式的SFT样本
        if sft_format == "conversations":
            sft_sample = {
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": correct_answer}
                ]
            }
        elif sft_format == "instruction":
            sft_sample = {
                "instruction": "回答牙科选择题，仅输出正确选项的字母（A、B、C、D、E），无需其他内容",
                "input": user_content.replace("你是专业的牙科咨询机器人，请回答以下选择题，仅需输出正确选项的字母（如A、B、C、D、E），不要输出其他内容，无需额外解释。\n", ""),
                "output": correct_answer
            }
        else:
            raise ValueError("仅支持 conversations 或 instruction 格式")
        
        sft_samples.append(sft_sample)
    
    # 5. 保存为JSONL文件
    with open(output_sft_path, "w", encoding="utf-8") as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"✅ 转换完成！共生成 {len(sft_samples)} 条SFT格式选择题样本")
    print(f"✅ 输出文件路径：{output_sft_path}")

# 运行转换（根据你的实际文件路径修改）
if __name__ == "__main__":
    convert_choice_to_sft(
        choice_data_path="./dental.jsonl",  # 你的原始口腔选择题数据
        output_sft_path="./dental_sft.jsonl",  # 转换后的选择题SFT数据
        sft_format="conversations"  # 对齐你sft_mini_512.jsonl的格式
    )