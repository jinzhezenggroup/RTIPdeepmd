# 亚甲基硼烷–N₂ 配合物与 CO₂ 的 N₂ 偶联（Inorg. Chem. 2025, 64, 17778；doi:10.1021/acs.inorgchem.5c02384）

论文用 ωB97X-D/cc-pVTZ//ωB97X-D/cc-pVDZ（Gaussian 16）研究 1-R + CO₂ → 1-R-CO₂.P 的 [3+2] 偶联
（N_β–C 与 B–O 成键），以及 1-tBu 与 CS₂、H₂CO、MeCN、MeCH=NMe 的反应。SI 给出了全部结构和
ωB97X-D/cc-pVDZ 单点能（`structures/`、`manifests/manifest.csv`）。

这里所有几何、过渡态、频率和反应路径都用 DeepMD（DPA-3.2-5M，OMol25 head，即 ωB97M-V/def2-TZVPD）计算；
DFT 只用来验证 DeepMD 的精度（`examples/accuracy/`）。

## 文件

| 文件 | 作用 |
|---|---|
| `validate_si.py` | DeepMD 与 SI 能量对比（反应能、竖直能垒、不同 head），`--relax` 时比较优化后的几何 |
| `ts_barriers.py` | 从 SI 过渡态出发，用 dimer 方法在 DeepMD 势能面上精修鞍点并检查虚频；`--from-path` 改为精修 RTIP 路径最高点 |
| `thermo.py` | DeepMD 数值 Hessian（去掉平动转动）+ 理想气体/刚性转子/谐振子，298.15 K、1 atm 的 ΔG 与 ΔG‡ |
| `map_atoms.py` | 用键连图子图匹配把 SI 产物/过渡态重排成"反应物 1 + 反应物 2"的原子顺序（`*_ordered.xyz`） |
| `reactant_complex.py` | 把两个反应物分别对齐到产物中对应的原子上，再沿质心连线拉开到最短原子间距 GAP，得到反应物复合物 |
| `analyze_rtip.py` | 判断 RTIP 轨迹是否生成 B–N=N–C–X–B 五元环，DeepMD 弛豫终点并给出表 1 的键长和反应能 |
| `reactive_site.py` | 打印 1-R 的 B–N–N 反应位点，供 `--fragment` 使用 |
| `manifests/paper_values.csv` | 论文表 1 的 ΔG、正文的 ΔG‡、表 3/4 的 EDA ΔE（cc-pVTZ） |
| `slurm/` | n5 上用过的作业脚本（在仓库根目录 `sbatch examples/ic5c02384/slurm/...`，先 `mkdir -p runs validation`） |

## 流程

```bash
sbatch examples/ic5c02384/slurm/run_ts.slurm        # SI 过渡态 → DeepMD 鞍点，validation/ts_barriers.json
sbatch examples/ic5c02384/slurm/thermo.slurm        # ΔG、ΔG‡，validation/thermo.json（需要上一步的 ts_geometries/）
R="1-tBu__CO2 1-Me__CO2 1-Ph__CO2 1-PMe2__CO2 1-SiMe3__CO2 1-H__CO2 1-CN__CO2 1-tBu__CS2 1-tBu__H2CO 1-tBu__MeCN 1-tBu__MeCH=NMe"
sbatch examples/ic5c02384/slurm/run_aligned.slurm "$R" "2.5 3.5" --max-step 4000   # RTIP AttractivePot + 路径最高点精修
```

## 结果

### 1. DeepMD 精度

- 相对 SI（ωB97X-D/cc-pVDZ，SI 几何）：42 个 CO₂ 偶联反应能 MAE 2.5 kcal/mol（最大 5.8），Spearman 0.87；
  9 个竖直能垒 MAE 4.0（最大 8.6，系统性偏高）。Default head（OMat24）更差（MAE 3.6，最大 13.6），所以默认用 OMol25。
  DeepMD 优化后的几何与 SI 几何的 RMSD 约 0.17 Å。
- 相对模型自己的训练泛函（PySCF ωB97M-V/def2-TZVPD，SI 几何，`validation/dft_ic5c.json`）：
  反应能误差 −0.9（tBu+CO₂）、−1.2（Me+CO₂）、−2.7（tBu+H₂CO）kcal/mol；竖直能垒误差 +2.6、+1.3、+4.1；
  力 RMSE 0.06–0.11 eV/Å。也就是说，与论文的差别有相当一部分来自泛函（ωB97M-V vs ωB97X-D），不是拟合误差。

### 2. 过渡态与能垒（`ts_barriers.py` + `thermo.py`，kcal/mol）

DeepMD 鞍点都只有一个虚频（−169 到 −493 cm⁻¹），离 SI 过渡态的 RMSD 0.11–0.48 Å。

| 反应 | ΔE‡ DeepMD | ΔE‡ 论文（cc-pVTZ，表 3/4） | ΔE‡ SI（cc-pVDZ） | ΔG‡ DeepMD | ΔG‡ 论文 |
|---|---|---|---|---|---|
| 1-tBu + CO₂ | 12.0 | 13.1 | 10.1 | 25.2 | 25.1 |
| 1-Me + CO₂ | 15.5 | 16.2 | 13.9 | 27.6 | |
| 1-PMe₂ + CO₂ | 12.8 | 17.6 | 14.2 | 26.4 | |
| 1-Ph + CO₂ | 16.7 | 18.6 | 16.2 | 28.9 | |
| 1-SiMe₃ + CO₂ | 23.7 | 21.8 | 18.5 | 34.4 | |
| 1-tBu + H₂CO | 0.6 | −1.8 | −4.7 | 15.2 | 11.9 |
| 1-tBu + CS₂ | 11.0 | 8.1 | 5.1 | 22.7 | 19.2* |
| 1-tBu + MeCN | 13.4 | 13.3 | 10.2 | 27.7 | 25.7 |
| 1-tBu + MeCH=NMe | 15.4 | 12.1 | 8.3 | 31.0 | 27.5 |

\* 论文正文把 19.2 写成 sulfur dioxide，但表 4 和 SI 都是 CS₂。

- 与论文 cc-pVTZ 能垒相比 MAE 2.1 kcal/mol（最大 4.8，PMe₂ 偏低）；ΔG‡ 相差 0.1–3.5。
- 底物顺序 H₂CO < CS₂ < CO₂ < MeCN < 亚胺 与论文完全一致；1-R 中 tBu 能垒最低、SiMe₃ 最高也一致。
  唯一的次序差别是 PMe₂（DeepMD 低估约 5 kcal/mol，排到第二）。

### 3. 产物（表 1）

| R | ΔG DeepMD | ΔG 论文 | B–N | N–N | B–O | C–O | C–N₂（DeepMD / 论文，Å） |
|---|---|---|---|---|---|---|---|
| tBu | −14.4 | −10.1 | 1.567 / 1.553 | 1.230 / 1.242 | 1.479 / 1.489 | 1.327 / 1.331 | 1.504 / 1.491 |
| Me | −12.1 | −6.5 | 1.574 / 1.557 | 1.228 / 1.241 | 1.476 / 1.486 | 1.325 / 1.332 | 1.507 / 1.492 |
| PMe₂ | −13.7 | −6.3 | 1.557 / 1.540 | 1.233 / 1.245 | 1.465 / 1.475 | 1.333 / 1.338 | 1.494 / 1.485 |
| Ph | −7.8 | −3.1 | 1.566 / 1.554 | 1.229 / 1.240 | 1.472 / 1.482 | 1.327 / 1.333 | 1.502 / 1.493 |
| SiMe₃ | −2.7 | −1.3 | 1.541 / 1.527 | 1.237 / 1.249 | 1.478 / 1.482 | 1.331 / 1.339 | 1.489 / 1.478 |
| H | −7.3 | 0.3 | 1.562 / 1.552 | 1.228 / 1.242 | 1.463 / 1.473 | 1.329 / 1.335 | 1.504 / 1.491 |
| CN | −5.0 | 2.0 | 1.566 / 1.564 | 1.228 / 1.238 | 1.455 / 1.460 | 1.334 / 1.341 | 1.501 / 1.494 |

键长取 RTIP 终点经 DeepMD 弛豫后的结构，与论文相差都在 0.02 Å 以内（B–N、C–N₂ 系统性长约 0.01，N–N 短约 0.012）。
ΔG 整体比论文负约 5 kcal/mol（标准差 2.2），tBu 仍然最有利（Spearman 0.86）。这个偏移主要来自泛函：
在 SI 几何上，ωB97M-V/def2-TZVPD 本身给出的 tBu、Me 反应能就比论文的 cc-pVTZ 值（由论文 ΔG 减去热校正估计）负 4–5 kcal/mol，
DeepMD 相对 ωB97M-V 只再多负约 1 kcal/mol。

### 4. RTIP 反应路径

- **SynthesisPot（不给产物的探测）**：整分子或只取 B–N–N 位点作 fragment，随机放置 CO₂，全部以 `climb` 结束。
  原因：随机朝向时 CO₂ 往往从 tBu/NHC 的氢一侧被拉向 1-R 的质心，位阻使能量在成键前就超过 `pot_climb`（0.185 Ha）；
  只取三个原子作 fragment 时 RTI 距离很小，σ 随之变小，偏置力 ∝ |a|/σ 极大。
- **AttractivePot，随机放置**（4 个种子 × 11 个反应，2000 步）：13/44 在 2000 步内成环，
  弛豫后的反应能与弛豫 SI 产物的结果相差不超过 1.4 kcal/mol；tBu+CO₂、H₂CO、MeCN 没有一次成功，CO₂ 常需要绕过整个分子。
- **AttractivePot，从对齐的反应物复合物出发**（`reactant_complex.py`，GAP 2.5 和 3.5 Å）：22/22 生成产物。
  路径最高点（带偏置）比反应物高 12–48 kcal/mol，只是上界；用 `ts_barriers.py --from-path` 精修后：

| 反应 | 路径最高点 | 精修后的 ΔE‡ | 从 SI 过渡态精修的 ΔE‡ | 正在形成的 N_β–C / B–X（Å）：RTIP 鞍点 vs SI TS |
|---|---|---|---|---|
| 1-tBu + CO₂ | 30.4 | 12.8 | 12.0 | 1.76 / 2.57 vs 1.70 / 2.71 |
| 1-Me + CO₂ | 36.2 | 15.1（GAP 2.5；GAP 3.5 收敛到 62.7 的另一个鞍点） | 15.5 | 1.73 / 2.57 vs 1.75 / 2.59 |
| 1-PMe₂ + CO₂ | 32.2 | 13.7 | 12.8 | 1.74 / 2.55 vs 1.72 / 2.53 |
| 1-SiMe₃ + CO₂ | 38.0 | 24.0 | 23.7 | 1.80 / 2.39 vs 1.67 / 2.54 |
| 1-Ph + CO₂ | 47.6 | dimer 退回反应物，无鞍点 | 16.7 | |
| 1-H + CO₂ | 39.8 | 20.9 | （SI 无 TS） | 1.77 / 2.48 |
| 1-CN + CO₂ | 41.2 | 26.3 | （SI 无 TS） | 1.82 / 2.34 |
| 1-tBu + CS₂ | 20.7 | 13.3 | 11.0 | 2.05 / 3.19 vs 2.09 / 3.01 |
| 1-tBu + H₂CO | 12.0 | dimer 退回反应物，无鞍点 | 0.6 | |
| 1-tBu + MeCN | 23.6 | 13.5 | 13.4 | 2.30 / 2.20 vs 2.16 / 2.35 |
| 1-tBu + MeCH=NMe | 29.1 | 15.1 | 15.4 | 2.22 / 2.39 vs 2.13 / 2.37 |

除注明外，两个 GAP 给出同一个鞍点（能垒相差 < 0.01 kcal/mol），都只有一个虚频。

不依赖 SI 过渡态，RTIP + DeepMD 在 11 个反应中的 9 个找到了同一反应通道的鞍点（N_β–C 先成键、B–X 较长的非同步 [3+2]），
能垒与从 SI 过渡态精修的结果相差 0.1–2.3 kcal/mol；整体 RMSD（0.4–1.1 Å）主要来自取代基构象和 CO₂ 两个 O 的置换。
失败的两个：Ph 从路径最高点（47.6，偏置使路径远离最低能量路径）出发的 dimer 退回了反应物；H₂CO 在 DeepMD 上几乎无能垒，最高点就在反应物复合物附近。
SI 没有给出的 1-H、1-CN 过渡态也由此得到：能垒 20.9 和 26.3 kcal/mol，
高于 tBu、Me、Ph、PMe₂，与它们在论文中 ΔG 为正（最不利）一致。
