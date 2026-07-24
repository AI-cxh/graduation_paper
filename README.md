# 图文语义冲突下的模态干预效用与选择性融合

本仓库用于硕士毕业论文实验、数据审计与论文材料管理。

## 当前阶段

执行一周 Go/No-Go 验证。第一项正式实验为 `EXP-001`：在 MMMC 分层子集上运行图文冲突、仅冲突文本和匹配干净图文三种条件，确认图文冲突下的模态跟随差异是否真实且可重复。

当前已完成 Qwen2.5-VL-3B-Instruct 的 50 配对推理冒烟测试，并建立本地可复现的词级指标、显式前提拒绝规则、配对转移统计和人工语义复核表。规则代理指标不等同于语义准确率。

## 目录

- `src/`：可复用的推理、打分、诊断和评测代码；
- `scripts/`：实验入口和一次性管理脚本；
- `configs/`：模型、数据、prompt 与实验配置；
- `tests/`：单元测试和小规模回归测试；
- `data/`：原始数据、处理结果和固定样本清单；
- `annotations/`：人工审核表与标注说明；
- `external/`：锁定版本的外部官方代码；
- `models/`：本地模型权重或缓存；
- `outputs/`：运行日志、逐样本预测和汇总指标；
- `environment/`：环境定义与依赖锁定文件。

## 记录要求

所有操作、失败实验和研究决策均追加到 [`实验与操作日志.md`](./实验与操作日志.md)。正式实验必须记录 Git commit、数据版本、模型 revision、prompt、解码参数、随机种子、资源消耗和结果文件路径。

## 当前复现命令

运行固定 50 配对、三种条件的 batch=1 冒烟推理：

```bash
CUDA_VISIBLE_DEVICES=0 .conda/bin/python scripts/run_exp001_smoke.py
```

生成确定性评测、配对统计和人工复核表：

```bash
.conda/bin/python scripts/evaluate_exp001.py
```

对官方参考答案执行四条件教师强制打分：

```bash
CUDA_VISIBLE_DEVICES=0 .conda/bin/python \
  scripts/score_exp001_reference_candidates.py
```

校验并汇总人工语义复核进度（标签未填完时不会伪造指标）：

```bash
.conda/bin/python scripts/summarize_exp001_semantic_review.py
```

运行 EXP-002 轻度图像扰动稳定性诊断：

```bash
CUDA_VISIBLE_DEVICES=0 .conda/bin/python \
  scripts/run_exp002_perturbation_reliability.py
```

人工复核规范见 [`annotations/EXP-001_答案语义复核说明.md`](./annotations/EXP-001_答案语义复核说明.md)。
