# Baseline environment

当前隔离环境位于项目根目录的 `.conda`。

```bash
conda activate /data/disk0/chenxuhao/graduation_paper/.conda
```

已验证的核心环境：

- Python 3.11.13
- PyTorch 2.8.0+cu128
- torchvision 0.23.0+cu128
- Transformers 4.55.0
- datasets 3.6.0
- qwen-vl-utils 0.0.14

环境建立过程：

1. 默认 Conda 源因持续 TLS/超时未能建立新环境；
2. 使用 `--offline` 将服务器已有的 `llama-factory` 环境克隆到项目 `.conda`；
3. 安装 `qwen-vl-utils==0.0.14`、`pytest==8.3.5` 和 `jsonlines==4.0.0`；
4. 完成 A800 BF16 运算、Qwen2.5-VL 类导入和 `pip check`。

`requirements-baseline.txt` 记录直接 Python 依赖。PyTorch/cu128 应从 PyTorch 对应 CUDA wheel 源单独安装，不能由默认 PyPI 隐式解析。

