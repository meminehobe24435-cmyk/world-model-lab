# world-model-lab —— 常用命令
#
# Windows 上如果没有 make，直接照抄下面的 python 命令即可（每条都独立可跑）。

PY ?= python
DATA ?= data
RUNS ?= runs

.PHONY: help test smoke check data train bench ablation all clean

help:
	@echo "make test      - 跑全部单元测试"
	@echo "make smoke     - 渲染 + 物理自检（几十秒，不需要数据集）"
	@echo "make data      - 采集完整数据集（约 5 分钟，CPU）"
	@echo "make train     - 训练主模型"
	@echo "make bench     - 在主模型上跑完整评测"
	@echo "make ablation  - 跑三个消融（无动作 / GRU / VQ tokenizer）"
	@echo "make all       - data -> train -> bench -> ablation 全流程"
	@echo "make clean     - 清掉数据集与实验产物"

test:
	$(PY) -m pytest tests -q

smoke:
	$(PY) scripts/selfcheck_render.py
	$(PY) scripts/selfcheck_physics.py

data:
	$(PY) scripts/collect_all.py --out $(DATA) --scale 1.0

train:
	$(PY) -m wm.train --data $(DATA) --out $(RUNS) --name base --epochs 12

bench:
	$(PY) -m wm.bench.run_benchmark --ckpt $(RUNS)/base/ckpt.pt --data $(DATA) \
		--out $(RUNS)/base --tag val
	$(PY) -m wm.bench.run_benchmark --ckpt $(RUNS)/base/ckpt.pt --data $(DATA) \
		--out $(RUNS)/base --tag hard --skip-mpc

ablation:
	$(PY) -m wm.train --data $(DATA) --out $(RUNS) --name no_action --no-action --epochs 12
	$(PY) -m wm.train --data $(DATA) --out $(RUNS) --name gru --dyn gru --epochs 12
	$(PY) -m wm.train --data $(DATA) --out $(RUNS) --name vq --vq --epochs 12

all: data train bench ablation

clean:
	rm -rf $(DATA) $(RUNS) .pytest_cache
