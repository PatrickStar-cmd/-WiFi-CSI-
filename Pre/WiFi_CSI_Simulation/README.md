# Wi-Fi CSI 人体运动检测 Python 仿真

依据 `../仿真方案/Wi-Fi CSI人体运动检测仿真方案.v2.docx` 实现。核心任务为人体运动／静止检测，以及单链路、双接收和四链路融合比较。2026-11-04已完成基线和优化后的核心实验。最新优化结果位于 `outputs/core_optimized_20261104_v1`，分析入口为其中的 `核心仿真实验结果分析.v3.html`；原基线及完整记录保存在 `outputs/core_20261104_v2`。

90个母场景按54/18/18划分；18个测试场景各有3个噪声重复。B2最佳固定单链路、四链路100 Hz、四链路50 Hz的平均F1分别为29.95%、35.85%、36.69%；四链路相对单链路的配对区间跨越0，平均收益存在场景差异。TinyCNN四链路100 Hz跨3个初始化种子的平均F1为45.27%。当前主要优化方向为弱运动召回和动作转换阈值。原始执行记录 `outputs/core_20261104_v1` 保留用于数值诊断；v2复用同一原始CSI和训练权重，完成浮点精度修正后的验证集阈值标定与测试评价。

## 1. 项目入口

| 文件 | 用途 |
|---|---|
| `run_simulation.py` | `plan` 查看计划；`run` 执行实验；`plot` 读取已有结果绘图 |
| `configs/default.json` | 场景、信道、动作、检测、训练、实验与图表参数 |
| `check_code.py` | 语法、导入、配置与手工输入的软件逻辑检查 |
| `analyze_results.py`、`audit_results.py` | 已保存结果的中文分析报告、数据与概率核验 |
| `recompute_core.py` | 复用已有数据和检查点，执行数值精度修正后的阈值标定与评价 |
| `diagnose_thresholds.py` | 仅使用验证集分析阈值与误报、召回的关系 |
| `optimize_core.py`、`configs/optimization.json` | 多尺度特征、训练与阈值选择；冻结后执行原测试和新增场景 |
| `csi_sim/optimized.py` | 40维观测特征、当前时刻标签训练、独立开启／关闭判定 |
| `report_optimization.py` | 基线复现核验、优化对比表、时间轴和动画 |
| `csi_sim/scene.py` | 躯干、左右手的连续运动轨迹与独立真值 |
| `csi_sim/channel.py` | 墙壁反射、静态散射、人体回波与观测误差 |
| `csi_sim/ofdm.py` | 已知训练符号、分数时延信道、LS估计和E0检查 |
| `csi_sim/dataset.py` | 母场景划分、HDF5存储与因果时窗 |
| `csi_sim/features.py` | RSSI等效统计、幅度方差、自相关与16维特征 |
| `csi_sim/models.py`、`cnn.py` | 逻辑回归和7,121参数TinyCNN |
| `csi_sim/fusion.py`、`evaluate.py` | 融合、阈值、状态确认、分类及事件评价 |
| `csi_sim/runner.py` | E0–E5、演示、空间覆盖和区域识别的实验编排 |
| `csi_sim/plots.py` | 场景动画、检测时间轴、链路对比与补充图表 |

## 2. Python环境

推荐 Python 3.12。当前项目已经建立 `.venv`，安装了基础依赖和 CPU 版 PyTorch。进入本项目目录后使用：

```powershell
.\.venv\Scripts\python.exe run_simulation.py plan
.\.venv\Scripts\python.exe check_code.py
```

以上两条命令用于查看计划和检查软件接口。实验入口由 `run` 子命令显式选择。

在其他电脑建立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
```

`requirements-lock.txt`记录交付环境的完整版本；`validation/code_checks.v3.json`记录软件检查和源文件哈希。中文字体通过`plots.font`配置，当前使用Windows的Microsoft YaHei。CPU是默认计算设备；GPU执行可通过`--device cuda`选择，需安装对应的PyTorch环境。

## 3. 执行与复查命令

新实验指定新的结果目录。已完成结果可直接绘图、分析和核验。

```powershell
# 完整核心比较：E0、E1、E2和25秒动作演示；包含B3的三个训练种子
.\.venv\Scripts\python.exe run_simulation.py run --stages core --output outputs/course_core

# 以B2完成场景、时间轴和链路对比，随后可扩展其他方法
.\.venv\Scripts\python.exe run_simulation.py run --stages E0 E2 demo --models B2 --output outputs/course_b2

# 完整主实验：E0–E5、动作演示和空间覆盖图
.\.venv\Scripts\python.exe run_simulation.py run --stages all --output outputs/course_all

# 任选阶段：参数扫描，或选做的左中右区域识别
.\.venv\Scripts\python.exe run_simulation.py run --stages E3 E4 E5 --models B2 --output outputs/robustness
.\.venv\Scripts\python.exe run_simulation.py run --stages region --models B2 --output outputs/region

# 从已有结果重新生成图表与GIF
.\.venv\Scripts\python.exe run_simulation.py plot --run-dir outputs/course_core

# 读取本次完成的核心结果
.\.venv\Scripts\python.exe analyze_results.py --run-dir outputs/core_20261104_v2 --version 2
.\.venv\Scripts\python.exe audit_results.py --run-dir outputs/core_20261104_v2
```

`--no-plots`让实验先保存数据与预测，图表可随后通过`plot`生成。`plot --no-animation`仅生成静态图。`demo`单独执行时会先完成B2训练及验证，再生成固定动作序列的检测结果。`coverage`为E2的空间覆盖子任务，可单独选择；`all`包含它。`region`为独立选做任务。

## 4. 场景与信道实现

- 房间6 m × 5 m；A/B发送，C/D接收，四链路为AC、AD、BC、BD。A/B发包相位错开半个包周期，默认各100包/s。
- 背景由直达路径、四面墙的一次镜像反射和8个静态散射点构成。每个母场景固定背景，同一母场景的噪声与参数变体共享背景随机种子。
- 人体用躯干和双手表示。走动采用余弦启停速度与平滑停止的折返路径；手部摆动在每段首尾平滑回到固定位置。挥手采用平滑包络调制的正弦位移。静止人体保留固定回波。
- 人体反射路径长度为发送端至散射点、散射点至接收端两段距离之和。使用实际频率`fc + k·Δf`计算传播相位，并在复数域叠加各路径。
- 人体回波使用四链路共享的强度系数，在房间中心参考姿态标定至默认−15 dB。每条链路的观测噪声参考该链路固定背景功率；节点移位实验保持原始功率标定。
- 主实验增益为1。扰动项包括丢包、时间戳抖动、接收端残余时钟偏差、共享接收增益、元信息补偿误差、链路掉线、风扇散射和C节点移位。C/D时钟偏差分别取配置值的正负方向。
- OFDM使用64点正交归一化FFT、16点CP、56个频点和两次BPSK训练估计平均。17抽头Kaiser窗sinc滤波器实现分数时延，并补偿8采样群时延。训练块尾部的周期保护样本用于数值插值；E0分别记录几何真值误差、FIR近似误差和观测噪声误差。
- 两次训练平均后，CSI噪声方差为时域复噪声方差的一半。E0和长序列共用这一噪声标定关系。频率、反射强度和场景参数均保存在结果配置中。

## 5. 数据、训练和比较规则

90个母场景在切窗前分为54训练、18验证、18测试。测试母场景各有3个独立观测噪声重复。训练／验证随机覆盖速度0.1–1.5 m/s、SNR 10–30 dB和回波比−30至−10 dB。

每场先用10秒静态校准，之后按1秒窗口、0.2秒步长输出。窗口仅访问当前时刻已经到达且时间戳位于窗口内的观测。原始有效率至少80%，最大缺口0.05秒。统计量使用原始有效包，频谱与CNN使用窗口内插值后的序列。导数特征定义为相邻有效原始包的幅度差除以时间差后的均方值。自相关使用包序号相邻的原始观测对。

模型输入来自观测组。动作表、事件、轨迹和标签保存在独立的`truth`组，用于监督训练、评价和动画。训练采用运动占比≥80%或≤20%的窗口；测试按窗口右端真值评价全部时段，同时记录稳态指标。

| 方法 | 实现 |
|---|---|
| B0 / B0q | 等效RSSI波动；B0q使用1 dB量化 |
| B1 | 跨子载波幅度方差中位数 |
| B1a | CSI功率一阶自相关的跨频点均值 |
| B2 | 16维特征＋训练集标准化＋L2逻辑回归；C由验证集log-loss选择 |
| B3 | 两层Conv1D、GAP、FC；三个初始化种子，验证损失早停 |

阈值在验证集上选择：执行连续两次状态确认后，静态FPR≤1%的候选中取运动F1最高者，并列时取更高阈值。257个分位数候选加两端全静止／全运动候选覆盖整个分数范围。训练、模型选择和阈值标定完成后，测试阶段冻结这些参数。

四链路共享每链路模型。融合权重来自有效率与静态校准幅度噪声方差；正权重最大最小比限制为10。有效链路集合为空时输出“数据不足”，恢复后重新累计两次状态确认。最佳固定单链路由验证集选定。

B2/B3推断保留稳定得分 `e = -ln(1-p) = softplus(logit)`，加权融合通过概率补集的logsumexp计算，保持原加权概率平均的数学定义。使用float64权重求和，保留接近概率1时的排序精度。预测文件同时保存得分域、得分、阈值与概率；阈值判定使用稳定得分。图表纵轴相应标注为“运动检测得分”。

E2同时比较两种资源条件：①每链路100 Hz，单／双接收100包/s、四链路200包/s；②总计100包/s，单／双接收100 Hz、四链路各50 Hz。采样率或窗口变化时，使用相同母场景划分重新训练；SNR、速度、丢包和环境扰动使用冻结模型。E3–E5默认扫描B2，`experiments.sweep_models`可加入B3。

静态专项测试覆盖空房、静止人体各10分钟；各段另带10秒校准。空间覆盖在躯干活动区内留出0.2 m短运动余量，以0.5 m步长选择位置，每点8个方向、3个噪声重复。区域识别采用独立三分类逻辑回归，边界附近0.2 m样本单列。

## 6. 结果与可视化文件

| 路径（相对于结果目录） | 内容 |
|---|---|
| `figures/scene_animation.gif` | 房间、三个身体散射点、四链路幅度、检测得分与状态同步回放 |
| `figures/scene_snapshot.png/.svg` | 动画中的静态场景截图 |
| `figures/detection_timeline.png/.svg` | CSI变化、运动检测得分、阈值、预设状态与预测状态 |
| `figures/single_vs_four_links.png/.svg` | 两种发包预算下的F1、召回率和误报率 |
| `figures/csi_heatmap.png/.svg` | 四条链路的子载波幅度热力图 |
| `figures/method_comparison.*`、`roc.*`、`confusion_matrices.*` | 方法比较、ROC与带“数据不足”列的混淆矩阵 |
| `figures/sweep_*.png/.svg` | 信号与采集参数扫描曲线 |
| `figures/disturbance_comparison.*` | 增益、时钟、掉线、风扇和节点移位的扰动对比 |
| `figures/spatial_coverage.*` | 短运动事件召回率空间分布 |
| `figures/region_confusion.*` | 选做的左中右区域识别混淆矩阵 |
| `E0/physical_checks.json`、`ofdm_packets.csv` | 几何连续性、多普勒、零噪声整时延和OFDM误差检查 |
| `conditions/*/data/*.h5` | 窗口特征、质量标记、真值；主实验另存原始CSI |
| `conditions/*/predictions/*.csv` | 每次输出的分数、状态、阈值、有效链路数与真值 |
| `conditions/*/events/*.json` | 逐事件命中、漏检、起始与停止延迟 |
| `metrics.csv`、`summary.csv` | 单场指标及母场景均值、95%置信区间 |
| `paired_differences.csv`、`fixed_budget_paired_differences.csv` | 两类预算条件下四链路相对最佳固定单链路的成对差值 |
| `cnn_seed_summary.csv` | B3不同训练种子的均值和标准差 |
| `models/*/calibration.json` | 冻结阈值、验证误报率及最佳单链路 |
| `config.json`、`split.json`、`environment.json`、`resources.json` | 完整配置、数据划分、依赖版本、时间和峰值内存 |
| `核心仿真实验结果分析.v2.html`、`ppt_comparison.v2.csv` | 中文分析、动画图表和PPT数据表 |
| `numerical_correction.v1.json`、`audit.v1.json` | 数值修正记录、数据复用证据和结果核验 |

图表统一从保存的预测、观测和真值读取。GIF使用PillowWriter生成。动画中的轨迹标记为预设运动轨迹；检测图表标记为仿真结果。

## 7. 指标口径与检查状态

分类指标包含Precision、Recall、F1、宏平均F1、FPR和AUC。“数据不足”单列并计入输出覆盖率。事件召回覆盖完整运动时段；运动开始前已经持续的误报单独记录；事件起始匹配采用运动开始后的新稳定上升沿。停止延迟在下一个运动事件开始前统计。误报间隔不足1秒时合并，按静态分钟数归一化。

先平均同一母场景的噪声重复，再按母场景进行1,000次bootstrap。`summary.csv`中的延迟为场内统计量的母场景平均；逐事件原始延迟位于`events`目录，便于另行计算总体分位数。B3训练种子另报均值与标准差。模型预测计时为共享四链路推理批次耗时，融合计时单列；整体内存和时间包含数据生成、训练、评价与绘图阶段。

当前18项软件检查覆盖分组隔离、阈值、缺测融合、权重限制、概率精度、事件匹配、发包预算、CNN结构、多尺度噪声归一化、短窗口时间范围和状态判定，并完成全部Python文件语法与模块导入检查。E0数值检查、核心性能评价与图表渲染已完成；基线结果核验覆盖255个HDF5缓存、54组跨协议配对场景及1,057,000条学习模型预测概率。优化核验另检查原基线复现、新旧场景隔离与模型冻结哈希。E3–E5、空间覆盖和区域识别保持后续可执行状态。

## 8. 模型和实现参考

- [CARM，静态与动态信道叠加](https://www.sigmobile.org/mobicom/2015/papers/p65-wangA.pdf)
- [Widar，传播几何与多普勒](https://cswu.me/papers/MobiHoc17_widar_paper.pdf)
- [WiDetect，CSI功率自相关](https://cswu.me/papers/ubicomp19_widetect_paper.pdf)
- [Espressif ESP-CSI](https://github.com/espressif/esp-csi)
- [SciPy periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [PyTorch Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
- [Matplotlib PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html)

## 9. 召回率优化版本

原16维特征增加0.2、0.4、1秒三个尺度，每尺度8个噪声归一化统计量，总计40维。新增量包括子载波方差、相邻采样协方差、相关性、活动子载波占比和分组平均幅度的变化。静态校准噪声来自最初10秒观测。全部特征只访问决策时刻以前的数据。

优化模型使用全部有效窗口的当前时刻标签，静止／走动／挥手的样本权重总量为0.5／0.25／0.25。在原54个训练场景拟合多尺度逻辑回归及7叶／15叶梯度提升树，在原18个验证场景选择模型、平均／最大值融合、开启／关闭阈值和确认次数。100 Hz选择15叶树＋质量加权平均；50 Hz选择7叶树＋最大得分融合。各自包含160轮提升迭代。

四组对照：O0原方案；T1保持原模型、特征和确认逻辑，仅重新标定阈值；A1使用优化算法并保持验证整体FPR≤1%；A2使用优化算法、验证整体FPR≤5%且稳态FPR≤1%。整体FPR的统计包含全部动作转换窗口。稳态定义沿用完整1秒历史均静止，单独报告。

模型和阈值冻结后，评价原18个测试场景及18个新背景／新轨迹场景，各3个噪声重复，两个发包协议合计216场运动回放。另保留20分钟静态和25秒演示。E0直接引用相同物理实现的既有验证记录。新场景保持原房间布局和标称信噪比，场景ID与全部已有划分互斥。`freeze.v1.json`保留评价前冻结哈希，`audit.v1.json`核对评价后状态及原基线复现。

原测试集四链路100 Hz：O0的F1为35.85%，T1为83.75%，A1为88.16%，A2为97.13%；A2挥手帧召回率96.00%，整体静止FPR为3.08%。新增场景A2的F1为97.53%，挥手帧召回率98.02%。本轮结果为纯仿真，完整指标、延迟和静态误报见分析报告。

```powershell
# 使用新的目录重新训练、冻结，再评价；两步顺序执行
.\.venv\Scripts\python.exe optimize_core.py fit --source outputs/core_20261104_v2 --output outputs/optimization_new_run
.\.venv\Scripts\python.exe optimize_core.py evaluate --output outputs/optimization_new_run
.\.venv\Scripts\python.exe report_optimization.py --run-dir outputs/optimization_new_run
```

入口中的原始CSI和基线权重来自本地基线结果目录；代码压缩包保存源程序与配置，原始数据和模型位于本机`outputs`目录。汇报素材包保存最新图表、报告、数据表与冻结记录。`presentation/demo_A2.csv`仅将已保存A2预测适配到通用绘图接口，预测值和状态保持一致。
