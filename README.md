# 图文语义冲突下的模态干预效用与选择性融合

本仓库用于硕士毕业论文实验、数据审计与论文材料管理。

## 当前阶段

已完成首轮 Go/No-Go、公开数据复核、EXP-007 无参考特征诊断、EXP-008 分组选择器验证和 EXP-009 普通 VQAv2 独立对照。当前结论是：课题问题可做，原生回答的反事实视觉贡献不仅在两类 HaloQuest 样本中具有排序信号，从 HaloQuest 冻结训练的仅贡献选择器也在300条 VQAv2 正常问题上取得初步正向迁移；但现有轻度扰动可靠性仍没有形成稳定联合增量，不能作为既成贡献。

当前已完成 Qwen2.5-VL-3B-Instruct 的 MMMC 50配对冒烟测试，以及 HaloQuest 287条 false-premise 样本的原生/前提核验两动作比较。规则代理指标不等同于语义准确率，Codex AI审核也不称作人工标注或官方 HaloQuest Auto-Eval。

HaloQuest 非false-premise控制实验也已完成：299条控制问题、598次生成。重点AI审核同时发现固定核验的帮助与伤害，支持继续研究样本级干预效用，但仍需普通VQA能力保持对照。

EXP-007 在两类审核样本各91条上完成了728次无参考候选打分。特征只使用原生回答、问题和图像，不读取官方参考答案；因此它对应“先生成一次，再决定是否执行核验式二次回答”的选择性干预，而不是零次生成前路由。

EXP-009 从 VQAv2 val2014 确定性抽取300条高共识、互异图像问题（yes/no、number、other各100条），完成600次双动作生成和1200次无参考候选评分。固定前提核验的归一化 VQA 软准确率为0.8570，原生回答为0.8557，差值区间跨0；冻结的仅贡献选择器只干预25条，策略分数为0.8650，相对原生提高0.0093，但帮助/伤害各仅7条，仍需扩大验证。

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

将用户明确确认的 AI 预复核行同步到正式复核表：

```bash
.conda/bin/python scripts/confirm_exp001_ai_prereview.py
.conda/bin/python scripts/merge_exp001_confirmed_review.py
.conda/bin/python scripts/summarize_exp001_semantic_review.py
```

按已确认的数据有效性问题执行排除敏感性分析：

```bash
.conda/bin/python scripts/analyze_confirmed_validity_sensitivity.py
```

运行 EXP-002 轻度图像扰动稳定性诊断：

```bash
CUDA_VISIBLE_DEVICES=0 .conda/bin/python \
  scripts/run_exp002_perturbation_reliability.py
```

运行并评估 EXP-003 前提核验干预：

```bash
CUDA_VISIBLE_DEVICES=0 .conda/bin/python \
  scripts/run_exp003_intervention_smoke.py
.conda/bin/python scripts/evaluate_exp003_intervention.py
```

准备、运行并评估 EXP-005 HaloQuest 公开数据基线：

```bash
.conda/bin/python scripts/prepare_haloquest_eval.py
CUDA_VISIBLE_DEVICES=3 .conda/bin/python \
  scripts/run_exp005_haloquest_baseline.py --device cuda:0
.conda/bin/python scripts/evaluate_exp005_haloquest.py
.conda/bin/python scripts/summarize_exp005_ai_audit.py
```

准备、运行并评估 EXP-006 HaloQuest 控制实验：

```bash
.conda/bin/python scripts/prepare_haloquest_control_eval.py
CUDA_VISIBLE_DEVICES=3 .conda/bin/python \
  scripts/run_exp005_haloquest_baseline.py \
  --config configs/exp006_haloquest_control.yaml \
  --manifest data/manifests/haloquest_control_eval.jsonl \
  --output outputs/predictions/exp006/qwen2_5_vl_3b_native_vs_verification.jsonl \
  --device cuda:0
.conda/bin/python scripts/evaluate_exp006_haloquest_control.py
.conda/bin/python scripts/summarize_exp006_control_ai_audit.py
```

提取 EXP-007 原生回答贡献与扰动可靠性特征（支持断点续跑）：

```bash
CUDA_VISIBLE_DEVICES=3 .conda/bin/python \
  scripts/run_exp007_native_answer_features.py --device cuda:0
```

运行 EXP-008 按图像分组的轻量选择器与双向跨子集验证：

```bash
.conda/bin/python scripts/run_exp008_grouped_selector.py
```

准备、运行并评估 EXP-009 普通 VQAv2 能力保持对照：

```bash
.conda/bin/python scripts/prepare_exp009_vqav2_control.py
CUDA_VISIBLE_DEVICES=3 .conda/bin/python \
  scripts/run_exp009_vqav2_baseline.py --device cuda:0
.conda/bin/python scripts/evaluate_exp009_vqav2.py
CUDA_VISIBLE_DEVICES=3 .conda/bin/python \
  scripts/run_exp009_native_answer_features.py --device cuda:0
.conda/bin/python scripts/evaluate_exp009_frozen_selector.py
```

人工复核规范见 [`annotations/EXP-001_答案语义复核说明.md`](./annotations/EXP-001_答案语义复核说明.md)。
