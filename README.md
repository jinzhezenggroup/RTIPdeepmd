# RTIPdeepmd

用 DeepMD 势能面驱动的 RTIP 分子演化 MD（RTIP evolution MD）。这是上游 Rust 程序
[MillenniumDream/RTIP-MD](https://github.com/MillenniumDream/RTIP-MD) 中 `EvolutionPot` 工作流的 Python 移植：
物理公式与 Rust 逐行对应，只把 CP2K（B97-3c）换成了 DeepMD。

默认体系与 Rust `lib.rs` 相同：1 Ca + 2 OH + 1 线性戊糖 + 3 CH₂O + 6 H₂O（54 原子，甲醛成糖反应），
输入结构在 `molecules/`。

## 运行（n5）

```bash
git clone git@github.com:jinzhezenggroup/RTIPdeepmd.git
cd RTIPdeepmd
mkdir -p deepmd_pretrained && cp /path/to/DPA-3.2-5M.pt deepmd_pretrained/   # 模型不进 git
sbatch run.slurm --max-step 20000 --stop-temp 30000
```

- `run.slurm` 会激活 `/group/software/deepmd-kit-3.1.1`（可用 `DEEPMD_ENV` 覆盖），
  模型路径默认 `deepmd_pretrained/DPA-3.2-5M.pt`（可用 `MODEL_PATH` 覆盖）。
- `sbatch run.slurm` 后面的参数原样传给 `python -m rtipdeepmd`。
- 输出在 `runs/<jobid>/`。
- 只想检查初始盒子：`python -m rtipdeepmd --dry-run --seed 1 --output-dir runs/test`。

## 参数

默认值等于 Rust `Para::new()`（`src/io/input.rs`），都可以用命令行覆盖：

| 命令行 | Rust 字段 | 默认 | 含义 |
|---|---|---|---|
| `--a0` | `rtip_para.a0` | 0.0005 Ha | RTIP 高度每步的变化量 |
| `--dt` | `md_para.dt` | 0.5 fs | 步长 |
| `--tau` | `md_para.tau` | 10 fs | Berendsen 耦合时间 |
| `--temp-bath` | `md_para.temp_bath` | 1500 K | 热浴温度 |
| `--decreasing-multiple` | `md_para.decreasing_multiple` | 2.0 | Decreasing 阶段每步回升 `2·a0` |
| `--decreasing-bound` | `md_para.decreasing_bound` | 0.5 | `a_min` 回升到成键变化时的 0.5 倍后重新搜索 |
| `--split-step` | `md_para.split_step` | 100 | 每 100 步重新切分分子（≤0 关闭） |
| `--max-step` | `md_para.max_step` | 1e8 | 最大步数 |
| `--print-step` | `md_para.print_step` | 1 | 轨迹帧间隔 |

DeepMD 选项：

- `--head`：多任务模型的 head，默认 `OMol25`（分子 DFT，ωB97M-V）。
  DPA-3.2-5M 有 23 个 head；不指定时 deepmd 会用 "Default"，它对应 OMat24（无机材料，PBE），不适合这个分子体系。
  对单 head 模型，这个参数会被忽略。
- `--charge`、`--spin`：DPA-3.x 的 frame parameter，即总电荷和自旋多重度 2S+1。
  默认电荷为 0，多重度取与电子数一致的最低值。默认体系中性时有 225 个电子，所以是双重态；
  模型自带的默认值 `[0, 1]` 对它不成立。如果要把戊糖当作醇盐负离子，用 `--charge -1`（226 电子，单重态）。
  Rust/CP2K 的 `B97-3C.inp` 写的是 Charge 0、Multiplicity 1，对这个 225 电子的体系同样不成立。

其他选项：

- `--seed`：建盒的随机种子，不给则随机生成，并记录在 `run.json` 里。
- `--min-dist`：建盒时分子间的最小距离，默认 3.5 Å。
- `--init file.xyz`：从给定结构开始，不再建盒。
- `--pbc`：把盒子当作周期性晶胞传给 DeepMD。默认关闭，因为 Rust/CP2K 用的是 `PERIODIC NONE`。
- `--stop-temp T`：温度超过 T 时提前停止。这个判据 Rust 里没有，只是为了避免跑飞的任务继续空耗 GPU。

## RTIP 系数与演化算法（对照 Rust `md.rs`）

每一步：

1. **构造目标结构。** 把每个分子沿"分子几何中心 → 全体几何中心"方向平移 1%，得到目标结构。
2. **偏置势。** 以目标结构为中心放一个高斯：
   `E = Σ_k (w_k/W)·a_min·exp(-d_k²/2σ²)`，其中 `w_k = d_k⁻⁷`；
   宽度 σ 取当前结构到目标结构的 RTI 距离（每步重新计算）。
3. **系数 `a_min` 为负，即吸引势。** 它由一个三态机控制：

| 状态 | 进入条件 | 每步 `a_min` 的变化 |
|---|---|---|
| Increasing | 默认 | `a_min -= a0`（势阱加深） |
| Falling | 两个分子已经接触（原本不成键的原子对距离 < 1.2(rᵢ+rⱼ)） | `a_min += a0` |
| Decreasing | 键发生变化：成键对 > 1.6(rᵢ+rⱼ)，或非键对 < 1.0(rᵢ+rⱼ) | `a_min += 2·a0`，直到 `a_min > 0.5·a_min(变化时)`，然后重新识别键连和分子，回到 Increasing |

**注意：** Falling 阶段没有下限。如果分子长时间保持接触，`a_min` 会越过 0 变成排斥势，这是 Rust 原版的行为。

原子半径用 Cordero 2008 共价半径，质量用原子单位（amu × 1822.888），两者都与 Rust 的
`constants.rs` 一致。被忽略的元素对（H–O、C–O、H–Ca、C–Ca、O–Ca、Ca–Ca）也与 Rust 相同。

## 输出

`runs/<jobid>/` 下：

- `supercell.xyz`：初始结构。
- `run.json`：本次运行的全部参数、随机种子和模型路径。
- `rtip.pdb`：轨迹（Å）。
- `rtip.out`：逐步日志，列与 Rust 相同，并额外加了最后一列 `a_min`：
  `step time(fs) rti_dist(Bohr) temp(K) kin(Ha) pot_real(Ha) pot_rtip(Ha) f_real f_rtip(Ha/Bohr) rtip_status a_min(Ha)`。
- `rtip_decreasing_steps`：每一段 Decreasing 的起止步。

## 与旧版 Python 移植的差异

旧版（`src/`，已删除）的轨迹在几步内温度就到了 10⁶ K 以上，原因是以下几处与 Rust 不一致：

| 问题 | 旧版 | 现在（= Rust） |
|---|---|---|
| 原子质量单位 | amu，相当于步长放大了 √1823 ≈ 43 倍 | 电子质量 |
| RTIP 系数 | `a_min += a0`，为正、无界，变成了排斥势，把分子推开 | 负值，由三态机控制 |
| 共价半径 | 另一套表（如 O 0.48 Å），C–O 单键会被判成断键 | Cordero 2008（O 0.66 Å） |
| `get_adj_mat` | 没有 −1（非键）标记 | 1 / −1 / 0 |
| `judge_variation_of_bonding` | 阈值 1.0 与 1.6 用反了 | 断键 > 1.6，成键 < 1.0 |
| `judge_adj_of_mol` | 任意近距离即判接触 | 只看原本不成键的分子间原子对；单个分子返回 True |
| 周期性 | 盒子作为周期晶胞传给 DeepMD，但 RTI 距离用的是未回折坐标 | 非周期，与 CP2K 设置一致 |
| DeepMD head | 默认 head，即 OMat24（材料），fparam [0, 1] | OMol25（分子），fparam 为 [电荷, 自洽的多重度] |
| 浮点精度 | JAX 未开 x64 时会被截断成 float32 | NumPy float64 |

另外删去了这条 DeepMD 工作流用不到的模块：自研 NN、CP2K 接口、pathway sampling / IDWM 等。
完整的 pathway sampling 移植见 [jinzhezenggroup/RTIP](https://github.com/jinzhezenggroup/RTIP)。

## 测试

```bash
python -m pytest
```

测试覆盖以下几点（不需要 DeepMD）：

- RTI 距离的平移和转动不变性，以及与 Kabsch RMSD 的等价；
- RTIP 力等于能量的负梯度（有限差分检验）；
- 键连判据的边界情况；
- 三态机的系数变化；
- 吸引势确实能把分子拉近。
