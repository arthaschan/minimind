# 一段完整的黑盒蒸馏的代码,独立的，不是和本项目集成的代码

# 导入所需库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm  # 进度条可视化

# ====================== 关键配置（可按需修改） ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU/CPU
TEACHER_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # 黑盒教师模型
STUDENT_HIDDEN_SIZE = 128  # 学生模型隐藏层大小（轻量化）
NUM_CLASSES = 2  # 分类类别数（正面/负面情感）
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
TEMPERATURE = 2.0  # 蒸馏温度系数（核心参数）
ALPHA = 0.7  # 蒸馏损失权重（核心参数，硬标签损失权重=1-ALPHA）
MAX_SEQ_LEN = 32  # 文本最大长度

# ====================== 1. 自定义数据集类（含黑盒教师数据生成） ======================
class SentimentDataset(Dataset):
    def __init__(self, tokenizer, sample_num=1000):
        self.tokenizer = tokenizer
        self.sample_num = sample_num
        # 生成模拟文本数据（也可替换为真实数据集，如IMDB、SST-2）
        self.texts = self._generate_simulate_texts()
        # 黑盒获取教师模型的软标签 + 真实硬标签
        self.teacher_soft_labels, self.hard_labels = self._get_teacher_outputs()

    def _generate_simulate_texts(self):
        """生成模拟情感文本（正面/负面各500条）"""
        positive_texts = [
            f"I love this product, it's {adj}!" for adj in ["great", "amazing", "excellent", "wonderful", "fantastic"]
        ] * 100
        negative_texts = [
            f"I hate this product, it's {adj}!" for adj in ["terrible", "awful", "bad", "horrible", "worse"]
        ] * 100
        all_texts = positive_texts + negative_texts
        np.random.shuffle(all_texts)
        return all_texts[:self.sample_num]

    def _get_teacher_outputs(self):
        """黑盒调用教师模型，获取软标签（无模型内部访问权限，仅输入输出）"""
        # 加载教师模型（实际黑盒场景可替换为API调用，无需本地加载）
        teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME).to(DEVICE)
        teacher_model.eval()  # 评估模式，关闭dropout

        teacher_soft_labels = []
        hard_labels = []

        # 批量处理文本，获取教师输出
        text_loader = DataLoader(self.texts, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():  # 关闭梯度计算，节省资源
            for batch_texts in tqdm(text_loader, desc="黑盒生成教师软标签"):
                # 文本预处理
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    return_tensors="pt"
                ).to(DEVICE)

                # 黑盒获取教师logits，转换为软标签
                teacher_logits = teacher_model(**inputs).logits
                # 温度系数平滑，生成软标签（核心：蒸馏的知识载体）
                soft_label = nn.functional.softmax(teacher_logits / TEMPERATURE, dim=-1)
                teacher_soft_labels.append(soft_label.cpu())

                # 获取真实硬标签（模拟标注，实际场景可替换为真实标签）
                batch_hard_labels = []
                for text in batch_texts:
                    if "love" in text or "great" in text or "amazing" in text:
                        batch_hard_labels.append(1)  # 正面情感
                    else:
                        batch_hard_labels.append(0)  # 负面情感
                hard_labels.extend(batch_hard_labels)

        # 拼接所有软标签
        teacher_soft_labels = torch.cat(teacher_soft_labels, dim=0)
        hard_labels = torch.tensor(hard_labels, dtype=torch.long)

        return teacher_soft_labels, hard_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 文本编码
        inputs = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        )
        # 提取input_ids和attention_mask（去除batch维度）
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        soft_label = self.teacher_soft_labels[idx]
        hard_label = self.hard_labels[idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "soft_label": soft_label,
            "hard_label": hard_label
        }

# ====================== 2. 构建轻量化学生模型（PyTorch原生） ======================
class LightweightStudentModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes, max_seq_len):
        super(LightweightStudentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # 词嵌入层
        self.dropout = nn.Dropout(0.1)  # dropout防止过拟合
        self.encoder = nn.Linear(hidden_size, hidden_size)  # 简单编码器（轻量化）
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 平均池化，压缩序列维度
        self.classifier = nn.Linear(hidden_size, num_classes)  # 分类头
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, attention_mask=None):
        # 1. 词嵌入
        embeds = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        # 2. 应用attention_mask（屏蔽padding部分）
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeds.size())
            embeds = embeds * mask_expanded
        # 3. 编码器层
        encoder_output = self.encoder(embeds)  # [batch_size, seq_len, hidden_size]
        encoder_output = torch.relu(encoder_output)  # 激活函数
        encoder_output = self.dropout(encoder_output)
        # 4. 池化层（将序列维度压缩为1）
        encoder_output = encoder_output.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        pooled_output = self.pooling(encoder_output).squeeze(-1)  # [batch_size, hidden_size]
        # 5. 分类层
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        return logits

# ====================== 3. 蒸馏损失函数（蒸馏损失 + 硬标签损失） ======================
def distillation_loss(student_logits, teacher_soft_labels, hard_labels, temperature, alpha):
    """
    黑盒蒸馏损失计算
    :param student_logits: 学生模型原始输出
    :param teacher_soft_labels: 教师模型软标签
    :param hard_labels: 真实硬标签
    :param temperature: 蒸馏温度
    :param alpha: 蒸馏损失权重
    :return: 总损失
    """
    # 1. 蒸馏损失（KL散度：匹配学生与教师的软标签分布）
    # 学生输出先做log_softmax，教师输出做softmax（KL散度输入要求）
    student_soft_logits = nn.functional.log_softmax(student_logits / temperature, dim=-1)
    distill_loss = nn.KLDivLoss(reduction="batchmean")(student_soft_logits, teacher_soft_labels)
    # 温度系数平方缩放（保持损失量级稳定，官方推荐）
    distill_loss = distill_loss * (temperature ** 2)

    # 2. 硬标签损失（交叉熵：保证学生模型的分类准确性）
    hard_loss = nn.CrossEntropyLoss()(student_logits, hard_labels)

    # 3. 总损失：加权求和
    total_loss = alpha * distill_loss + (1 - alpha) * hard_loss

    return total_loss

# ====================== 4. 训练与验证流程 ======================
def train_and_validate():
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)

    # 2. 构建数据集（训练集:800条，验证集:200条）
    full_dataset = SentimentDataset(tokenizer, sample_num=1000)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 3. 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 实例化学生模型
    vocab_size = tokenizer.vocab_size
    student_model = LightweightStudentModel(
        vocab_size=vocab_size,
        hidden_size=STUDENT_HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)

    # 5. 优化器
    optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

    # 6. 训练循环
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        # ---------- 训练阶段 ----------
        student_model.train()
        train_total_loss = 0.0
        train_correct = 0
        train_total = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]") as pbar:
            for batch in pbar:
                # 加载批次数据到设备
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                soft_label = batch["soft_label"].to(DEVICE)
                hard_label = batch["hard_label"].to(DEVICE)

                # 梯度清零
                optimizer.zero_grad()

                # 学生模型前向传播
                student_logits = student_model(input_ids, attention_mask)

                # 计算蒸馏损失
                loss = distillation_loss(
                    student_logits=student_logits,
                    teacher_soft_labels=soft_label,
                    hard_labels=hard_label,
                    temperature=TEMPERATURE,
                    alpha=ALPHA
                )

                # 反向传播 + 参数更新
                loss.backward()
                optimizer.step()

                # 统计训练损失和准确率
                train_total_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(student_logits, 1)
                train_total += hard_label.size(0)
                train_correct += (predicted == hard_label).sum().item()

                # 更新进度条
                pbar.set_postfix({
                    "loss": train_total_loss / train_total,
                    "acc": train_correct / train_total
                })

        # 计算训练集平均损失和准确率
        avg_train_loss = train_total_loss / train_size
        train_acc = train_correct / train_total

        # ---------- 验证阶段 ----------
        student_model.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]") as pbar:
                for batch in pbar:
                    # 加载批次数据到设备
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    soft_label = batch["soft_label"].to(DEVICE)
                    hard_label = batch["hard_label"].to(DEVICE)

                    # 学生模型前向传播
                    student_logits = student_model(input_ids, attention_mask)

                    # 计算蒸馏损失
                    loss = distillation_loss(
                        student_logits=student_logits,
                        teacher_soft_labels=soft_label,
                        hard_labels=hard_label,
                        temperature=TEMPERATURE,
                        alpha=ALPHA
                    )

                    # 统计验证损失和准确率
                    val_total_loss += loss.item() * input_ids.size(0)
                    _, predicted = torch.max(student_logits, 1)
                    val_total += hard_label.size(0)
                    val_correct += (predicted == hard_label).sum().item()

                    # 更新进度条
                    pbar.set_postfix({
                        "loss": val_total_loss / val_total,
                        "acc": val_correct / val_total
                    })

        # 计算验证集平均损失和准确率
        avg_val_loss = val_total_loss / val_size
        val_acc = val_correct / val_total

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), "best_student_model.pth")
            print(f"保存最佳模型，验证准确率：{best_val_acc:.4f}")

        # 打印epoch统计信息
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("-" * 50)

    # 训练结束
    print(f"训练完成！最佳验证准确率：{best_val_acc:.4f}，模型保存为 best_student_model.pth")

# ====================== 5. 运行训练与验证 ======================
if __name__ == "__main__":
    train_and_validate()