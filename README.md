
```
source /group/software/deepmd-kit-3.1.1/bin/activate /group/software/deepmd-kit-3.1.1
```


主程序入口在 lib.rs:53。
按“输入/输出/做了什么工作”来总结如下。
输入
1. 分子结构文件（XYZ）
• 在 lib.rs:95 到 lib.rs:116 连续读取了 22 个 xyz 文件（如 CH2O.xyz、Ca.xyz、20-linear-pentose.xyz 等）。
• 读取逻辑在 system.rs:182。
2. CP2K 输入文件
• 使用 B97-3C.inp 构建 CP2K 势能面对象，见 lib.rs:147。
• CP2K 对象构造在 cp2k.rs:519。
3. 程序参数（不是外部配置文件）
• 参数由 Para::new() 直接给默认值，见 lib.rs:77 和 input.rs:107。
• 包含 MD 步长、温控参数、最大步数、打印频率等，字段在 input.rs:52。
4. MPI/并行环境
• 初始化 MPI，并用 split_by_color 划分通信器，见 lib.rs:56 和 lib.rs:78。
输出
1. 标准输出
• 打印 Begin Initialization 和 Finalization done，见 lib.rs:55 与 lib.rs:244。
2. 输出目录
• 通过 create_output_path 创建输出目录，见 lib.rs:79 和 output.rs:19。
• 这里传入 world.rank() % 1，恒为 0，所以目录固定是 0/。
3. 初始超级晶胞文件
• ROOT rank 写出 supercell.xyz，见 lib.rs:135。
4. MD 轨迹与日志文件
• EvolutionPot 的 rtip_nvt_md 会写：
• rtip.pdb（结构轨迹）md.rs:345
• rtip.out（每步能量/温度/力/状态）md.rs:346
• rtip_decreasing_steps（RTIP 降低区间）md.rs:347
5. CP2K 输出
• cp2k.out 路径在 lib.rs:80，由 CP2K 计算过程追加写入。
程序主要做的工作
1. 初始化计算环境
• 先初始化 MPI，再调用 CP2K 初始化（不重复初始化 MPI），见 lib.rs:56 与 lib.rs:58，对应接口在 cp2k.rs:70。
2. 组装初始体系
• 读取多个小分子/中间体结构。
• 按给定倍数 mul 通过 auto_super_cell 自动放入一个足够大的盒子，见 lib.rs:117 到 lib.rs:119，算法在 system.rs:459。
3. 并行同步初始坐标
• ROOT 把坐标和晶胞广播给其他 rank，见 lib.rs:130 到 lib.rs:132。
4. 运行 RTIP 偏置 NVT 分子动力学
• 创建 EvolutionPot 和 Cp2kPES 后进入主循环，见 lib.rs:138 到 lib.rs:149。
• 循环内主要是：
• leapfrog 积分 + Berendsen 恒温；
• 调用 CP2K 求真实势能与力（get_energy_force）md.rs:553与 cp2k.rs:559；
• 计算 RTIP 偏置势并根据成键变化/分子接近程度调整 RTIP 状态（Increasing/Decreasing/Falling），见 md.rs:492、md.rs:569、md.rs:630；
• 按 print_step 写轨迹与日志，见 md.rs:710 到 md.rs:715。
5. 收尾
• 结束前调用 CP2K finalize，见 lib.rs:243，对应实现 cp2k.rs:90。








