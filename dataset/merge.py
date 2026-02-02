# 合并2个jsonl 文件为一个jsonl 文件
# 输入output.jsonl", "lora_medical.jsonl"
# 输出 merge.jsonl
import json

def process_single_json(input_file):
    """
    适配标准JSONL文件（每行一个JSON对象）的辅助函数
    """
    data_list = []
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                raw_obj = json.loads(line)
                # 统一格式为列表项（兼容行内是单个对象或数组）
                items = raw_obj if isinstance(raw_obj, list) else [raw_obj]
                data_list.extend(items)
        return data_list
    except FileNotFoundError:
        print(f"警告：未找到文件 {input_file}，跳过该文件处理")
        return []
    except json.JSONDecodeError as e:
        print(f"警告：{input_file} 某行JSON格式非法 - {e}，跳过该行处理")
        return data_list

# 定义输入文件路径（两个需要合并的文件）和输出文件路径
input_files = ["huatuo.jsonl","./dental/dental_sft_train.jsonl"]  # 两个输入完整JSON文件
output_file_path = "lora_huatuo_dental_sft_train.jsonl"

# 收集两个文件的所有有效数据
all_data_items = []
for file in input_files:
    file_data = process_single_json(file)
    all_data_items.extend(file_data)  # 合并两个文件的数据列表

if not all_data_items:
    print("错误：未获取到任何有效数据，无法生成输出文件")
else:
    # 将合并后的数据转换为标准JSONL格式并写入b.jsonl
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for item in all_data_items:
            
        
            # 按JSONL规范写入（每行一个独立JSON对象，保留非ASCII字符）
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write("\n")
    
    print(f"合并转换完成！已生成输出文件：{output_file_path}")
    print(f"共处理 {len(all_data_items)} 条数据，写入 {output_file_path}")