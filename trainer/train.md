训练参数图
===


| 参数               | pretrain          | ful-sft           | lora                |
|--------------------|-------------------|-------------------|:--------------------|
| save_weight        | pretrain          | full_sft          |                     |
| batch_size         | 32                | 16                | 32                  |
| learning_rate      | 5e-4              | 1e-6              | 1e-4                |
| accumulation_steps | 8                 | 1                 | 1                   |
| hidden_size        | 512               | 512               | 512                 |
| num_hidden_layers  | 8                 | 8                 | 8                   |
| max_seq_len        | 340               | 340               | 340                 |
| data_path          | pretrain_hq.jsonl | ft_mini_512.jsonl | lora_identity.jsonl |
| from_weight        | none              | pretrain          | full_sft            |
| lora_name          |                   |                   | lora_identity       |

