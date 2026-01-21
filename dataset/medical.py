import json

# 转换FreedomIntelligenc/Huatuo26M-Lite 数据为 本项目格式
# 定义输入输出文件路径
input_file_path = "huatuo26M.jsonl"  # 备注：若文件后缀仍为g.jsonl，仅需修改此处为"g.jsonl"即可
output_file_path = "huatuo.jsonl"

try:
    # 1. 读取完整的JSON文件（一次性加载整个文件，适配完整JSON格式）
    with open(input_file_path, "r", encoding="utf-8") as infile:
        # 加载整个JSON数据（支持根节点是数组或单个对象，与你的示例格式匹配）
        raw_data = json.load(infile)
    
    # 2. 统一处理数据格式：无论根节点是数组还是单个对象，都转为列表便于遍历
    # 若根节点是单个JSON对象，包装为长度1的列表；若是数组，直接使用
    data_items = raw_data if isinstance(raw_data, list) else [raw_data]
    
    # 3. 构建目标格式并写入JSONL文件
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for item in data_items:
            # 提取question和answer字段（兼容字段缺失，避免KeyError）
            question_content = item.get("question", "")
            answer_content = item.get("answer", "")
            
            # 构建要求的对话格式结构
            target_conversation = {
                "conversations": [
                    {
                        "role": "user",
                        "content": question_content
                    },
                    {
                        "role": "assistant",
                        "content": answer_content
                    }
                ]
            }
            
            # 4. 按JSONL规范写入（每行一个独立JSON对象，不格式化输出）
            json.dump(target_conversation, outfile, ensure_ascii=False)
            outfile.write("\n")  # 每行结束换行，符合JSONL格式要求
    
    print(f"转换完成！已生成输出文件：{output_file_path}")

except FileNotFoundError:
    print(f"错误：未找到输入文件 {input_file_path}，请检查文件路径是否正确")
except json.JSONDecodeError as e:
    print(f"错误：解析JSON文件失败，格式不合法 - {e}")
except Exception as e:
    print(f"错误：转换过程中出现异常 - {e}")