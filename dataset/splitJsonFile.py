import json
import os
import random

def split_jsonl_file(
    input_jsonl_path: str,
    output_dir: str = "./split_data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    将单个JSONL文件按指定比例拆分为训练集、验证集、测试集三个JSONL文件
    :param input_jsonl_path: 输入原始JSONL文件路径
    :param output_dir: 输出拆分后文件的目录
    :param train_ratio: 训练集比例（默认0.8）
    :param val_ratio: 验证集比例（默认0.1）
    :param test_ratio: 测试集比例（默认0.1）
    :param random_seed: 随机种子（保证拆分结果可复现）
    """
    # 1. 校验参数
    if not os.path.exists(input_jsonl_path):
        raise FileNotFoundError(f"输入文件不存在：{input_jsonl_path}")
    if not input_jsonl_path.endswith(".jsonl"):
        raise ValueError("输入文件必须是.jsonl格式")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("三个数据集比例之和必须为1.0")
    
    # 2. 创建输出目录（若不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 加载并过滤原始JSONL数据（保留有效行）
    all_samples = []
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 解析单行JSON，跳过无效格式行
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_num} 行JSON格式错误，跳过该行：{e}")
    
    total_samples = len(all_samples)
    if total_samples == 0:
        raise ValueError("原始JSONL文件中无有效数据")
    print(f"✅ 成功加载 {total_samples} 条有效数据")
    
    # 4. 打乱数据顺序（保证拆分后数据分布均匀）
    random.seed(random_seed)
    random.shuffle(all_samples)
    
    # 5. 计算各数据集的样本数量
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    # 处理四舍五入误差，确保测试集数量 = 总数量 - 训练集 - 验证集（保证无数据丢失）
    test_size = total_samples - train_size - val_size
    
    print(f"📊 拆分比例确认：")
    print(f"   - 训练集：{train_size} 条（{train_ratio*100}%）")
    print(f"   - 验证集：{val_size} 条（{val_ratio*100}%）")
    print(f"   - 测试集：{test_size} 条（{test_ratio*100}%）")
    
    # 6. 按比例切片拆分数据
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size+val_size]
    test_samples = all_samples[train_size+val_size:]
    
    # 7. 定义输出文件路径
    input_filename = os.path.basename(input_jsonl_path)
    filename_prefix = os.path.splitext(input_filename)[0]
    train_output_path = os.path.join(output_dir, f"{filename_prefix}_train.jsonl")
    val_output_path = os.path.join(output_dir, f"{filename_prefix}_val.jsonl")
    test_output_path = os.path.join(output_dir, f"{filename_prefix}_test.jsonl")
    
    # 8. 保存拆分后的JSONL文件
    def save_jsonl(samples, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"✅ 已保存 {len(samples)} 条数据至：{output_path}")
    
    save_jsonl(train_samples, train_output_path)
    save_jsonl(val_samples, val_output_path)
    save_jsonl(test_samples, test_output_path)
    
    print("\n🎉 所有数据拆分完成！")
    print(f"📁 输出目录：{os.path.abspath(output_dir)}")

# 运行示例
if __name__ == "__main__":
    split_jsonl_file(
        input_jsonl_path="./dental_sft.jsonl",  # 你的原始JSONL文件路径
        output_dir="./dental",  # 拆分后文件的输出目录
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42  # 固定种子，确保每次拆分结果一致
    )