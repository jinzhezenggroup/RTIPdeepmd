# DeepMD 版 RTIP 使用说明

目标：让当前 Python/JAX 版本以 DeepMD 为真实势能面，稳定地运行 RTIP 偏置 NVT MD，并输出可诊断日志。

## 1. 激活环境

按下面命令激活（与现有作业脚本一致）：

```bash
source /group/software/deepmd-kit-3.1.1/bin/activate /group/software/deepmd-kit-3.1.1
```

## 2. 运行入口

主入口：`python -m src.lib`

默认行为：

- 按 `lib.rs` 对应的配方读取一组 xyz（`LIBRS_RECIPE`）；
- 自动构造 supercell；
- 以 DeepMD 作为 real PES 运行 `evolution_nvt_md`；
- 输出 `rtip.pdb`、`rtip.out`、`rtip_decreasing_steps`。

## 3. 最小可运行命令

### 3.1 本地单进程测试（推荐先 dry-run）

```bash
python -m src.lib \
  --no-mpi \
  --xyz-dir . \
  --dry-run
```

### 3.2 实际运行（指定模型）

```bash
python -m src.lib \
  --no-mpi \
  --xyz-dir . \
  --model ./deepmd_pretrained/DPA-3.2-5M.pt \
  --output-index 0 \
  --supercell-out supercell-0.xyz
```

也可用环境变量：

```bash
export MODEL_PATH=/abs/path/to/model.pt
python -m src.lib --no-mpi --xyz-dir . --model "$MODEL_PATH"
```

## 4. Slurm 运行

项目已提供 `run.slurm`，直接提交即可：

```bash
sbatch run.slurm
```

脚本会：

- 激活 DeepMD 环境；
- 自动解析 `MODEL_PATH` / `DEEPMD_MODEL`；
- 默认模型回退到 `deepmd_pretrained/DPA-3.2-5M.pt`；
- 导出 `JAX_ENABLE_X64=1`；
- 在未显式设置时使用保守 MD 默认值：
  - `MD_PRINT_STEP=10`
  - `MD_DT=0.05`, `MD_TAU=200`
  - `MD_TEMP_BATH=800`, `MD_TEMP_START_SCALE=0.02`, `MD_TEMP_RAMP_STEPS=1000`
  - `MD_THERMOSTAT_LAM_MIN=0.995`, `MD_THERMOSTAT_LAM_MAX=1.005`

可选环境变量（推荐用于 GPU 调试与稳态运行）：

- `MD_MAX_STEP`：覆盖 `MdPara.max_step`。
- `MD_PRINT_STEP`：覆盖 `MdPara.print_step`。
- `MD_DT` / `MD_TAU`：覆盖 `dt` 与 `tau`。
- `MD_TEMP_BATH` / `MD_TEMP_START_SCALE` / `MD_TEMP_RAMP_STEPS`：覆盖升温策略。
- `MD_THERMOSTAT_LAM_MIN` / `MD_THERMOSTAT_LAM_MAX`：覆盖 thermostat 缩放范围。
- `OUTPUT_INDEX`：指定输出目录（默认 `SLURM_JOB_ID`）。
- `SUPERCELL_OUT`：指定 supercell 输出文件名。
- `RTIP_DRY_RUN=1`：只做初始化与 supercell 生成，不进入 MD。

示例：先做一个小步数 GPU 冒烟作业

```bash
MODEL_PATH=/home/lhshen/JAXversion/deepmd_pretrained/DPA-3.2-5M.pt \
MD_MAX_STEP=200 MD_PRINT_STEP=10 \
sbatch run.slurm
```

## 5. 关键稳定性策略（当前代码已接入）

1. 强制 JAX x64（启动时在代码中设置）。
2. thermostat 采用升温斜坡（`temp_start_scale` + `temp_ramp_steps`）。
3. thermostat 速度缩放有上下限（`thermostat_lam_min/max`）。
4. `evolution_nvt_md` 使用三态驱动：`Increasing -> Decreasing -> Falling`。
5. 事件判据接入主循环：
   - `judge_adj_of_mol`
   - `judge_variation_of_bonding`
6. 增加 runaway 早停：
   - 温度超阈值；
   - bias_dist 超阈值；
   - `f_real / f_bias` 连续过大。

## 6. 输出文件说明

假设 `--output-index 163`，主要输出在 `163/`：

- `163/rtip.pdb`：轨迹（逐步追加）。
- `163/rtip.out`：逐步标量日志。
- `163/rtip_decreasing_steps`：`Decreasing` 阶段区间记录。

此外：

- `slurm-<jobid>.out`：标准输出；
- `slurm-<jobid>.err`：错误/警告（含 DeepMD/JAX 提示）。

## 7. rtip.out 列含义

`rtip.out` 当前列为：

- `step`：步数
- `time`：时间（fs）
- `bias_dist`：RTI 距离
- `temp`：瞬时温度
- `kin`：动能
- `pot_real`：真实势能（DeepMD）
- `pot_bias`：RTIP 偏置势
- `f_real`：真实力范数
- `f_bias`：偏置力范数
- `target_temp`：当前 thermostat 目标温度（斜坡后）
- `a_min`：当前 RTIP 强度参数
- `state`：RTIP 状态（Increasing/Decreasing/Falling 或其他模式）
- `event`：事件与状态跃迁信息

若触发早停，日志会追加：

```text
# STOP step=...: ...
```

## 8. 常用参数

在 `src/io/input.py` 的 `MdPara` 中可调：

- `dt`, `tau`, `temp_bath`
- `temp_start_scale`, `temp_ramp_steps`
- `thermostat_lam_min`, `thermostat_lam_max`
- `decreasing_multiple`, `decreasing_bound`
- `bonded_multiple`, `unbonded_multiple`
- `runaway_temp`, `runaway_bias_dist`
- `runaway_force_ratio`, `runaway_ratio_patience`, `runaway_ratio_min_step`

同时可通过入口参数临时覆盖（不改源码默认值）：

- `--md-max-step`, `--md-print-step`
- `--md-dt`, `--md-tau`
- `--md-temp-bath`, `--md-temp-start-scale`, `--md-temp-ramp-steps`
- `--md-thermostat-lam-min`, `--md-thermostat-lam-max`

## 9. 常见问题

1. 模型路径错误

- 现象：启动时报 `model file not found`。
- 处理：确认 `--model` 指向现有 `.pb/.pth/.pt` 文件，或使用可用 pretrained 名称。

2. type_map 不匹配

- 现象：报元素不在 `type_map`。
- 处理：优先使用模型自带 `type_map`；必要时通过 `--type-map` 或 `--type-map-json` 显式传入。

3. 很快温度失控

- 现象：`temp`、`bias_dist` 急速上升。
- 处理：
  - 减小 `dt`；
  - 提高 `tau`；
  - 增大 `temp_ramp_steps`；
  - 收紧 `thermostat_lam_max`；
  - 调整 `runaway_*` 阈值做保护。




我先定位输入解析函数里的关键分支和报错条件，拿到精确行号后给你一份可直接照着写的输入模板。接下来会提炼成“语法规则 + 示例 + 常见踩坑”三部分。

Searched for regex `def read_input|def read_a_chemical_reaction|def split_multiple_species` (`**/microkinetics.py`), 3 results

Searched for regex `len\(words\) > 4|len\(words\) == 2|len\(words\) == 3|words\[0\] != "//"` (`**/microkinetics.py`), 4 results

Searched for regex `T\"\) and \(words\[1\] == \"=\"\)|dt\"\) and \(words\[1\] == \"=\"\)|max_step|print_step` (`**/microkinetics.py`), 22 results

Searched for regex `\"->\"|\"\+\"|\(words\[index\]\[0:1\] == "\("\)|reverse_barrier|forward_barrier` (`**/microkinetics.py`), 11 results

Searched for regex `temperature < 0\.000001|dt < 0\.0|max_step == 0|print_step == 0` (`**/microkinetics.py`), 4 results

定位完成了主要入口，我再补上“物种写法”的细则行号（比如系数默认值、纯数字非法）后就给你完整格式规范。

Searched for regex `if not ch\.isdigit|if index == 0|multiple = 1|species = multiple_species|raise Exception\(f\"\\n\\n\\n ERROR: There is some problem with the species` (`**/microkinetics.py`), 7 results

Searched for regex `Set the concentration of the uninitialized species to be zero|concentration = 0\.0` (`**/microkinetics.py`), 4 results

Searched for regex `if len\(words\) > 4|else:\n\s+raise Exception\(f\"\\n\\n\\n ERROR: There is some problem with the following line` (`**/microkinetics.py`), 9 results

Searched for regex `KCAL_TO_EV|reactions\[i\]\.barrier\[0\]|reactions\[i\]\.barrier\[1\]` (`**/microkinetics.py`), 5 results

Searched for regex `read_input\("input"\)` (`**/microkinetics.py`), 1 result

这个输入文件的格式要求可以按下面 6 条来写，基本就是解析器在 microkinetics.py 里硬编码的规则。

1. 文件里只识别三类有效行  
参考 microkinetics.py, microkinetics.py, microkinetics.py
- 反应行：分词后单词数 > 4
- 初始浓度行：分词后单词数 == 2
- 参数行：分词后单词数 == 3
- 其他长度会报错（见 microkinetics.py）

2. 注释和空行规则  
参考 microkinetics.py
- 空行会忽略
- 只有“首个单词是 //”的整行注释会忽略
- 行尾注释不安全，容易被当成额外 token 导致报错

3. 反应行语法（最关键）  
反应解析在 microkinetics.py
- 结构必须是：反应物 + ... -> 产物 + ... (正向势垒, 反向势垒)
- 箭头必须是 ->（见 microkinetics.py）
- 物种之间用 + 分隔（见 microkinetics.py, microkinetics.py）
- 势垒必须写成两个 token：第一个以 ( 开头且以 , 结尾，第二个以 ) 结尾（见 microkinetics.py, microkinetics.py）
- 也就是要写成例如 (12.5, 10.2)，不要写成 (12.5,10.2)

4. 物种写法规则  
物种拆分在 microkinetics.py
- 支持前置整数化学计量数：如 2H2O
- 不写系数时默认 1（见 microkinetics.py）
- 纯数字非法（见 microkinetics.py）
- 本质上格式是：前导整数系数 + 物种名

5. 浓度行规则  
参考 microkinetics.py
- 格式：物种名 浓度
- 浓度必须能转成 float
- 如果反应里出现但未在浓度行定义，会自动补成 0.0（见 microkinetics.py）

6. 参数行规则（必须给全）  
参考 microkinetics.py, microkinetics.py, microkinetics.py, microkinetics.py
- 只接受这四个键：
- T = 数值
- dt = 数值
- max_step = 整数
- print_step = 整数

并且最终校验：
- T >= 1e-6（见 microkinetics.py）
- dt >= 0（见 microkinetics.py）
- max_step > 0（见 microkinetics.py）
- print_step > 0（见 microkinetics.py）

补充两点：
- 程序固定读取当前目录下名为 input 的文件（见 microkinetics.py）。
- 势垒数值按 kcal/mol 解释后再换算参与速率计算（见 microkinetics.py, microkinetics.py）。

如果你愿意，我可以按这个解析器的规则给你直接生成一个可运行的 input 模板。