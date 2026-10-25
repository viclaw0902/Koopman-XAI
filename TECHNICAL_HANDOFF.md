# TEP Koopman-XAI 实验技术文档与跨电脑交接记录

文档日期：2026-10-25，Asia/Shanghai。

用途：使另一台电脑上的Codex或研究人员在没有完整聊天记录时，理解本项目最终约定、代码实现、实验进度和风险，并能核对状态后继续工作。

适用版本：本仓库 `Original_structure/` 的Final_exp实验及2026-10-25迁移快照。代码核对基点为提交 `00c09cb18decc62eae9c6f46273d327db2775c19`；该提交之后新增本交接文档。GitHub：<https://github.com/viclaw0902/Koopman-XAI>。

重要说明：文字文件能够转移项目知识和决策背景，不能转移模型内部记忆、隐藏状态或不存在于本仓库中的历史结果。以下“已完成”均指旧电脑停止时已核验的状态；新电脑实时进度必须重新查看文件和进程。

---

## 1. 接手者首先需要知道的结论

1. 项目研究对象是时序故障分类器的原始特征归因，不是把动力学预测误差当作故障解释目标。
2. 分类器固定为CNN-GRU，所有解释方法解释同一个冻结分类器。同一种子下不应出现“路线1分类准确率”和“路线3分类准确率”不同。
3. 路线1、路线2、Koopman IG的区别主要是解释路径：原空间直线、流形解码曲线、Koopman动力学约束曲线。
4. 当前分类数据有Normal和Fault1–20，共21类；四项核心归因指标只在Fault1–12、14、15，共14类上计算。Fault21不存在。
5. 33维输入包含XMEAS1–22和XMV1–11。当前Koopman把二者共同编码为状态，不再把XMV分离为控制量。
6. 同一种子内，CNN-GRU与时序卷积AE是并行输入支路；AE不是读取GRU的64维输出。历史命名称其为结构B。
7. 路线2与Koopman使用同一套冻结AE。Koopman额外训练升维、独立逆映射和每类一个K。
8. 固定真实正常/故障两端点；K通过故障端点的预测类别选择，解释中固定不变。
9. ADD/DEL统一采用原始33通道逐通道替换。潜空间替换曾讨论过，但不是当前主指标。
10. 旧电脑训练已停止；seed42分类器和AE已完成，z16动力学保存到第42轮，其他维度、种子和归因尚未完成。
11. 新电脑显卡为RTX5060，默认环境已更新为PyTorch2.7.1+cu128。未在旧电脑上实际验证5060，必须在新电脑运行自检。
12. `Changed_structure/` 只是预留目录，不做训练或实现。

## 2. 研究目标与实验分组

### 2.1 研究问题

给定正常到故障的同一运行轨迹，解释CNN-GRU为何把某个故障窗口判断为某个Fault类别，比较不同归因方法对已知根因通道的定位和对分类器输出的忠实程度。

核心假设依次为：

- 相比原空间直线，受数据流形几何约束的路径可能减少脱离数据分布的插值；
- 在共享流形基础上，Koopman动力学约束可能提供更符合时序演化的路径；
- 这些假设必须由实验检验，不预先保证Koopman IG一定更优。

归因是对分类器的解释。即使根因命中，也不等于完成了因果识别；不应把分类器路径梯度直接称为物理因果效应。

### 2.2 消融实验

| 方法标识 | 起点 | 终点 | 路径 |
|---|---|---|---|
| `route1_normal_ig` | 同轨迹正常medoid | 同轨迹故障候选medoid | 原始标准化窗口直线 |
| `route2_geodesic_ig` | 相同真实正常窗口 | 相同真实故障窗口 | h空间解码能量优化路径，解码回原空间 |
| `koopman_ig` | 相同真实正常窗口 | 相同真实故障窗口 | z空间固定端点动力学约束路径，z→h→原空间 |

同一种子、同一轨迹共享真实端点和目标类别。不同种子独立训练分类器，故各自的故障medoid和目标可能变化。

### 2.3 对比实验

当前比较Saliency Map、IDGI、Vanilla IG、Guided IG、Expected IG、DeepLIFT、ABIGX与Koopman IG。

早期用户提出过SHAP、CP、LLBBC；后续明确CP/LLBBC先不做，并选定现有列表。当前Final_exp代码不含SHAP，不应声称已运行SHAP。路线1正常medoid IG与对比组零基线Vanilla IG是两个不同的实验条目。

四项核心指标：correctness AUC、correctness SUM、consistency ADD、consistency DEL。保留分类指标、top-3和诊断量作辅助输出。

## 3. 路径、版本与文件边界

旧电脑项目根目录：`D:\Work\Koopman_for_TEP`。

旧电脑当前原始实验目录：`D:\Work\Koopman_for_TEP\Final_exp\Original_structure`。

用于提交GitHub的独立迁移副本：`D:\Work\Koopman_for_TEP\Final_exp\Migration_GitHub`。

新电脑用户准备使用：`D:\luo_WORK`，可以克隆为 `D:\luo_WORK\Koopman-XAI`。新位置不必与旧电脑盘符一致。

本文件中的相对路径均相对于Git克隆仓库根目录，不相对于旧项目根目录。

```text
Koopman-XAI/
├─ AGENTS.md                      Codex项目阅读入口
├─ TECHNICAL_HANDOFF.md            本文档
├─ README.md                      简短迁移说明
├─ MIGRATION_STATUS.json          停止时的静态进度快照
├─ environment.yml                RTX5060默认Conda环境
├─ requirements.txt               numpy/matplotlib固定版本
├─ requirements-cu128.txt          新GPU的PyTorch安装入口
├─ requirements-cu124.txt          原电脑旧环境追溯
├─ environment/source_environment_snapshot.json
├─ artifacts/                     数据与seed42训练状态ZIP分卷
├─ transfer_manifest.json          分卷/解压文件SHA-256清单
├─ build_transfer.py               制作迁移包；续训不需要重跑
├─ restore_transfer.py             校验并恢复快照；不启动训练
├─ check_environment.py            GPU及模型/优化器迁移自检
├─ start_resume.py                 先自检，再启动实验队列
├─ rieth_prepared/                 恢复后出现，Git忽略
├─ Changed_structure/              恢复后创建，保持为空
└─ Original_structure/
   ├─ config.json                 训练与解释配置
   ├─ core.py                     模型、窗口、标准化、真值映射
   ├─ train.py                    数据审计、分类、AE、Koopman训练
   ├─ explain.py                  端点、路径、归因、四项指标
   ├─ deeplift_gru.py              显式GRU的DeepLIFT适配
   ├─ summarize.py                配对统计、图像、CSV汇总
   ├─ save_endpoints.py            可独立执行的端点推理保存
   ├─ run_experiments.py           多种子串行训练/解释/汇总队列
   ├─ README.md                    实现说明
   └─ Results/                    恢复/续训结果，Git忽略
```

旧目录 `15faults_output`、`15faults_Ks`、`EXPERIMENTS`、无流形版本等不属于当前迁移运行入口，不要混用checkpoint或报告其结果为本次Final_exp结果。

`Original_structure` 是用户保留的目录名，不代表把历史所有实现原封不动复用。经代码核对，当前保留的是时序卷积AE并行支路结构B。

## 4. 数据来源、类别与防泄漏

### 4.1 数据来源与输入

使用用户已准备的Rieth扩展版TEP数组。旧原始RData目录为 `D:\Work\Datasets\dataverse_files`；迁移包只含本实验所需prepared数组，不含原始RData。

特征顺序严格固定：XMEAS1,…,XMEAS22，随后XMV1,…,XMV11，共33通道。不要按字符串字典顺序排序，否则XMEAS10会排在XMEAS2之前。

XMEAS是测量变量，XMV是操纵变量。本次二者联合构成自主状态。能否从33通道推断自主演化是建模假设，不意味着XMV在物理系统中从此不再具有控制属性。

### 4.2 划分和窗口数量

先按运行轨迹划分，再滑窗；不是把高度重叠窗口随机拆成训练/验证/测试。

| 划分 | 每类轨迹 | 总轨迹 | 每条采样点 | 每条窗口 | 全部分类窗口 |
|---|---:|---:|---:|---:|---:|
| train | 400 | 8,400 | 500 | 481 | 4,040,400 |
| validation | 50 | 1,050 | 500 | 481 | 505,050 |
| test | 50 | 1,050 | 960 | 941 | 988,050 |

窗口长度20、步长1。窗口数为T−20+1。训练、验证、分类测试使用全部合法窗口，不抽稀、不回到早期每轨迹20窗口设置。

所有种子复用相同轨迹划分，改变的是训练随机性，不重新分割数据。审计检查跨划分run ID无重叠，同一划分中故障轨迹注入前片段与同run正常轨迹一致。

### 4.3 分类标签与注入位置

类别编码：0=Normal，1–20=Fault1–20。

故障轨迹train/validation的前20点、test的前160点是真实正常前缀。代码使用零基窗口起点s、窗口长度L=20：

- s+L≤normal_prefix：窗口标签Normal；
- 否则采用轨迹故障标签；
- 分类保留跨注入窗口，按窗口末端状态标注；
- 故障解释候选必须s≥normal_prefix，即整个窗口在注入后；
- 动力学训练排除跨注入状态切换的样本链。

以test为例，正常候选s=0,…,140，共141个；完整注入后窗口s=160,…,940，再按预测是否为Fault筛选。不要把这些边界改成“连续三个正确分类窗口”，那是被后续medoid规则替代的旧方案。

### 4.4 标准化

“仅正常训练数据拟合标准化”指：只利用train中Normal轨迹计算每通道均值μ和标准差σ，再对所有类别、所有划分统一应用(x−μ)/σ。它不意味着分类器或AE只学习正常样本。

历史prepared数组已经标准化，但旧float32均值累积对低方差通道产生较大误差，XMEAS9曾出现标准化均值约0.9966、标准差约0.08017。当前 `corrected_scaler()` 在现有prepared正常训练数组上用float64重新计算shift/scale，再在内存进行等价仿射修正，不改原数组。

输出 `Original_structure/Results/normal_scaler_corrected.npz` 包含prepared_shift、prepared_scale及换算后的raw_mean/raw_std。该修正受已有float32存储精度限制，不是重新读取原始RData。

实际分类器和解释输入是修正后的标准化坐标；“原空间”在本项目指原33通道窗口空间，不是潜空间，也不保证以物理单位绘图。零基线意味着这些坐标的零点，即正常训练均值，而非所有物理传感器值为零。

## 5. 整体架构：分类与解释如何连接

```text
同一个标准化窗口 W_t ∈ R^(33×20)
  ├─ CNN-GRU → 64维分类表示 → 21类logit → 预测类别
  │       ↑
  │       └─ 所有解释路径解码到原空间后，逐点送入这个冻结分类器求输入梯度
  └─ 共享时序卷积AE编码器 E → h_t ∈ R^16
          ├─ 路线2：h空间路径 → AE解码器 D → 原窗口路径
          └─ Koopman升维 φ → 标准化z_t ∈ R^d
                 ├─ z_(t+1) ≈ K_c z_t
                 └─ 逆映射 ψ → h → D → 原窗口路径
```

没有“把分类后概率再做IG”。分类器提供待解释的可微标量函数F_c(W)和选K的预测标签；AE/Koopman提供输入空间曲线。对曲线上每个原空间窗口，始终计算同一F_c的原始输入梯度。

训练后分类器和AE冻结，冻结参数不等于关闭对输入的梯度。推理归因时分类器必须eval，不能让BatchNorm统计或Dropout随路径变化。

## 6. CNN-GRU的逐层结构

批量输入形状[B,33,20]。

```text
Conv1d(33→48, kernel=5, padding=2, stride=1)
→ BatchNorm1d(48)
→ GELU                         [B,48,20]
→ Conv1d(48→64, kernel=3, padding=1, stride=1)
→ BatchNorm1d(64)
→ GELU                         [B,64,20]
→ transpose(1,2)               [B,20,64]
→ GRU(input_size=64, hidden_size=64, batch_first=True)
→ 取最后时间步 recurrent[:,-1]  [B,64]
→ Linear(64→64) → GELU → Dropout(0.2) → Linear(64→21)
→ logit                        [B,21]
```

当前无MaxPool/AvgPool、无双向GRU、无额外多层GRU；两个卷积保持时间长度20。BatchNorm位于每个卷积之后、GELU之前；AE和Koopman MLP中没有这些BatchNorm层。

分类训练目标是21类交叉熵。各解释方法共享同一种子的classifier.pt，不独立训练“路线2分类器”“路线3分类器”。

## 7. 共享时序卷积自编码器

编码器：

```text
[B,33,20]
→ Conv1d(33→48,k5,p2) → GELU
→ Conv1d(48→64,k3,p1) → GELU      [B,64,20]
→ Flatten                       [B,1280]
→ Linear(1280→128) → GELU
→ Linear(128→16)                 h:[B,16]
```

解码器：

```text
h:[B,16]
→ Linear(16→128) → GELU
→ Linear(128→1280) → GELU
→ reshape                       [B,64,20]
→ Conv1d(64→48,k3,p1) → GELU
→ Conv1d(48→33,k5,p2)            [B,33,20]
```

AE在全部训练模式上优化平均重构MSE，验证集MSE选最优模型。它不是严格数学意义上保证可逆的坐标图，也不保证对每个故障完全重构。解码器重构误差及端点连接段必须保留并报告。

## 8. Koopman状态、逆映射与多个K

### 8.1 坐标定义

升维网络 φ：Linear(16→128)→GELU→Linear(128→d)。候选d=16、32、64、128。d=16不是严格升维，只是同维非线性重参数化候选。

预训练后，在所有训练窗口上计算φ(h)的逐维均值m与标准差s并固定：

z = (φ(h)−m) ⊘ s。

所有K作用在这个标准化z坐标。逆映射ψ：Linear(d→128)→GELU→Linear(128→16)，实际解码为：

h_hat = ψ(z⊙s+m)，W_hat = D(h_hat)。

独立逆映射读入全部d维；严禁使用z[:16]替代。K训练、路径约束、解码必须遵守同一坐标约定。

### 8.2 动力学定义

当前自主多K模型：z_(t+1)≈K_c z_t，K_c∈R^(d×d)，c=0,…,20。

- 训练同模式样本链时，c来自训练标签；
- 解释时，c来自故障端点分类器预测，固定整条路径；
- 不使用z_(t+1)=Kz_t+Bu_t，没有单独控制支路，也没有额外仿射偏置；
- h、升维与逆映射共享，只有K按类别独立；并非每条轨迹训练一个K。

数学上Koopman算子作用于可观测函数，若z=φ(E(W))近似张成有限维不变子空间，可得到线性近似。有限d、噪声、窗口状态不完备和模式切换都会导致残差，不能仅靠“升维”保证精确线性化。

一个全局K并非理论上绝不可能表示多模式，但要求共享有限维表示具有足够闭合性；多K提供分模式近似，未必自动改善分类或归因。当前冻结分类器，所以换K本身不会改变分类ACC。

### 8.3 岭回归初始化

对每个模式收集列向量状态对X=[z_t]、Y=[z_(t+1)]，求：

K_c = argmin_K (1/N_c)||Y−KX||_F² + λ||K||_F²

闭式形式为K_c=(YXᵀ/N_c)(XXᵀ/N_c+λI)^(-1)，实际采用线性方程求解，避免显式求逆。

λ候选：1e−5、1e−4、1e−3、1e−2、1e−1。每类按验证集1到10步纯递推平均z误差选λ，不按测试归因指标选择。

### 8.4 纯递推和联合损失

预测链：z_hat_(t+k)=K_c^k z_t，k=1,…,10。中间不喂未来真实h/z。目标z通过当前升维网络计算后detach，未来h作为监督。

记MSE为对应张量元素平均，当前代码每batch主要损失为：

L = 0.5·MSE(z_hat_(t+1),z_(t+1))
    +0.5·mean_(k=1..10) MSE(z_hat_(t+k),z_(t+k))
    +mean_(k=1..10) MSE(decode_h(z_hat_(t+k)),h_(t+k))
    +1.0·MSE(decode_h(z_t),h_t)
    +0.1·MSE(encode(decode_h(z_t)),stopgrad(z_t))
    +0.1·mean_j ReLU(0.5−std_batch(z_j))²
    +0.1·MSE(D(decode_h(z_t)),W_t)。

这不是只用10步终点误差；一步项之外还平均监督1–10步。解码h多步项权重隐式为1。

每轮结束额外优化谱半径软惩罚0.01·mean_c ReLU(ρ(K_c)−1.05)²，再将超限K整体缩放到ρ≤1.05。它不是每batch加入谱损失，也不保证任意长预测稳定。谱半径约束无法排除非正规矩阵的瞬态放大。

### 8.5 合法动力学链

H=10时要求t窗口和未来10个窗口不跨轨迹边界、不跨注入边界。故障轨迹注入前的有效链要求s+20+10≤normal_prefix；注入后要求s≥normal_prefix。由于训练故障轨迹前缀仅20点，正常K的10步训练链主要来自独立Normal轨迹。

### 8.6 训练阶段与超参数

| 阶段 | 可训练部分 | 设置 |
|---|---|---|
| 分类 | CNN-GRU及分类头 | AdamW，lr1e−3，wd1e−4 |
| 流形AE | 时序卷积编码器/解码器 | AdamW，lr1e−3，wd1e−4 |
| 升维预训练 | φ、ψ | 5轮，lr1e−3，h重构+方差约束 |
| z标准化 | 固定统计buffer | 所有训练窗口，预训练后计算 |
| K初始化 | 每类岭回归 | 验证递推误差选λ |
| 动力学微调 | φ、ψ、全部21个K | φ/ψ lr1e−4，K lr1e−5，wd1e−4 |

分类和AE：batch128、梯度累积2，典型有效更新批量256，尾组不足时按实际组长平均；ReduceLROnPlateau(patience3,factor0.5)；至少15、至多50轮，早停patience7，梯度范数裁剪1。

重要实现细节：`gradient_accumulation=2`用于分类与AE；动力学微调每batch直接更新，不做两步梯度累积，且没有同样的lr scheduler。不能在报告里笼统声称所有阶段有效batch256。

动力学同样至少15、至多50轮、patience7，按验证选择分数选best。训练种子42–46串行；每个d使用seed×1000+d初始化。

## 9. z维度选择：规则与塌缩风险

每个d输出：z一步/10步MSE、h10步MSE、解码原窗口10步MSE、原窗口持久性基线、原窗口重构MSE、h逆映射MSE、有效秩、最小方差、谱半径和逐类诊断。

选择分数S_d = 宏平均原空间10步预测MSE + 宏平均原空间当前重构MSE。

当前精确规则：先计算所有候选的最小S，再筛选S≤1.02·min(S)且effective_rank≥8的候选，选最小d。若无候选，报错停止，不生成解释。

注意：当前min(S)包含低秩候选，报错文字虽写“All latent candidates collapsed”，也可能是非塌缩候选不在2%带内，并非所有候选都低秩。遇到此情况应检查全部候选，不可误诊或擅改选择规则。

有效秩定义：对验证z协方差特征值λ_j归一化p_j=λ_j/Σλ，r_eff=exp(−Σp_j log p_j)。逐维方差大也可能强相关导致低有效秩。

旧电脑z16第42轮有效秩约2.66，低于8，已经构成风险。低latent预测误差不能掩盖表示塌缩；不要为了跑通而自动降阈值、去掉防塌缩条件或选择测试指标最好的维度。

## 10. 端点、目标与样本单位

每个评价故障使用全部50条test轨迹，共14×50=700条/种子。不是只随机抽一整条轨迹，也不是将test标签用于模型推理。

真实标签只用于事后按故障分组和查根因，分类器前向只输入窗口。

同轨迹正常候选：完全处于注入前的窗口，无需分类器同时预测正常。

故障候选：完全处于注入后且分类器预测为任意Fault的窗口，无需预测等于真实故障，也不要求连续三个正确窗口。

候选medoid是候选样本中与其他候选的欧氏距离之和最小者：

i* = argmin_i Σ_j ||vec(W_i)−vec(W_j)||_2。

在共同修正标准化33×20空间精确计算，使用float64分块距离；不是均值、不是离均值最近样本，也不是平方距离和。

得到真实正常窗口W_b与真实故障窗口W_f；目标c=argmax classifier(W_f)，解释目标F_c(W)为固定类别logit；K=K_c。不沿路径重新选目标或切换K。

如果没有故障候选，报告解释覆盖失败，不虚构端点。误分类轨迹保留，输出classification_correct；不能只保留预测正确的样本提高归因成绩。

seed42已保存700对端点：endpoint_selection.json/csv、endpoint_windows.npz、endpoint_selection_complete.json。NPZ windows形状[700,2,33,20]。解释脚本校验classifier.pt SHA后复用JSON端点索引；不盲用其他种子端点。

## 11. 三条路径的数学与实现

### 11.1 路线1：原空间直线

γ(α)=W_b+α(W_f−W_b)，α∈[0,1]。

所有原始坐标按同一α变化。它保证真实端点，但中间点未必属于数据分布。

### 11.2 路线2：解码器拉回几何

h_b=E(W_b)，h_f=E(W_f)，γ(α)=D(h(α))。

连续拉回度量形式为G(h)=J_D(h)ᵀJ_D(h)，几何能量为∫||J_D(h)h'(α)||²dα。

代码不显式计算完整雅可比或Christoffel符号，而固定h端点，优化离散解码能量N²·mean||D(h_(i+1))−D(h_i)||²。自动微分经过解码器回传到内部节点。

这是离散化、局部数值优化的测地线近似。不能声称证明了全局最短测地线；解码器非单射或秩不足时，拉回矩阵也未必正定。

### 11.3 Koopman IG：固定端点动力学桥

先对同轨迹W_b至W_f之间按时间等距取最多32个区间、33个观测窗口，编码到z作为初值，固定首尾，只优化内部节点。

设相邻初始窗口起点的真实时间差为Δt_i，固定故障预测K_c，当前路径目标为：

L_path = 1.0·mean_i ||z_(i+1)−K_c^(Δt_i) z_i||²
         +0.1·N²·mean_i ||D_K(z_(i+1))−D_K(z_i)||²
         +0.1·MSE(encode(decode_h(z_nodes)),z_nodes)，

其中D_K(z)=D(ψ(z⊙s+m))，encode是标准化Koopman编码。

Adam优化内部节点，lr0.01，最多400次；梯度范数小于1e−5，或至少迭代到约100次后连续40次无足够改善，会标记收敛。保存最优节点和完整历史。

K的幂次来自真实窗口时间差，不是IG积分点编号，也不是一律10步。训练10步是拟合约束，解释两端相距可能远大于10，因此可能出现长时外推误差。

当前路径是带几何和有效性项的动力学约束桥，不是简单z_i=K^i z_b的自由滚动。任意给定故障端点一般不等于K^T z_b，固定两端点与严格零残差滚动不能同时保证。

更严格地说，此目标不是某个已证明黎曼度量的纯测地线能量。代码标识沿用Koopman路径概念，科研描述应写“固定端点动力学约束路径/桥”；若要严格度量测地线需要另行定义和验证，不能无声替换当前实验。

解释只用同轨迹观测作初始化，当前没有额外逐节点贴合真实轨迹的锚定损失。优化后中间点可偏离观测。

### 11.4 真实端点与重构端点的连接

AE/逆映射不是精确可逆，解码的首尾一般不等于真实W_b/W_f。当前完整路径为：

真实W_b → 重构W_b → 解码潜空间路径 → 重构W_f → 真实W_f。

两端连接段在原空间线性插值并参与积分，不可丢弃。输出两端重构MSE及endpoint_connector_absolute_attribution_ratio；该比例是连接段合并贡献绝对量与总归因绝对量之比，可能因抵消而大于1，不应当作严格概率。

保存的`*_path.npz`是优化节点及其粗解码路径，不是自适应积分使用的全部稠密点；复画积分采样时要按同一解码和细化规则重建。

## 12. 路径IG到底计算什么

对原始通道j、窗口内时间τ：

A_(j,τ) = ∫_0^1 [∂F_c(γ(α))/∂γ_(j,τ)] · dγ_(j,τ)/dα dα。

离散积分：

A ≈ Σ_i 0.5·[∇F_c(W_i)+∇F_c(W_(i+1))] ⊙ (W_(i+1)−W_i)。

曲线路径不能用总端点差乘平均梯度代替，因为每个分段的方向和幅度不同。当前是在解码原空间点上求分类器输入梯度，不是计算“z维度贡献后把标签改成XMEAS/XMV”。

若路径连续且F足够光滑，Σ_(j,τ) A_(j,τ)=F_c(W_f)−F_c(W_b)。这叫完备性，不意味着每个通道贡献都是真实因果。

初始积分目标128步，必要时细化；完备性容差为0.01·max(|Δlogit|,1)，上限预算约1024步。实现按区间数向上取整并包含连接段，不保证点数恰好128或1024。达到预算仍不收敛时保留integration_converged=false，不伪装成功。

归因梯度batch16。CuDNN GRU输入梯度计算在代码中使用禁用CuDNN的上下文，以支持eval下反向。分类器参数不更新。

带符号通道贡献S_j=Σ_τ A_(j,τ)；绝对通道贡献B_j=Σ_τ|A_(j,τ)|。注意B_j≠|S_j|。指标排序与TOP-3使用B_j，signed图保留正负。

XMV和XMEAS共享标准化窗口坐标，积分贡献以同一logit变化单位表达，可联合排序。但可比单位不意味着没有相关性或控制反馈混杂。

## 13. 对比归因方法的当前实现

| 方法 | 当前行为 | 必须说明的差异 |
|---|---|---|
| Vanilla IG | 从全零标准化窗口到故障窗口，直线自适应积分 | 不同于消融路线1的同轨迹正常medoid |
| Saliency Map | 故障窗口固定logit的输入梯度 | 非差分贡献，不具有同样的完备性保证 |
| Guided IG | 零基线，优先更新低绝对梯度坐标，fraction0.25、max_distance0.02 | bounded路径、自写实现、左Riemann积分，有完备性诊断 |
| IDGI | 零基线直线，相邻输出增量按前点梯度平方归一化分配 | 128步；零梯度段单独记录，不是普通梯形IG |
| Expected IG | 16个正常训练背景分别做完整直线IG后平均 | 是多正常背景IG平均；没有额外添加随机噪声 |
| DeepLIFT | 同轨迹正常medoid作reference | 显式GRU规则适配，不是宣称融合GRU被任意现成库原生支持 |
| ABIGX | 为待解释窗口求自适应AFR正常基线，再做直线IG | 适配窗口CNN-GRU，不能声称精确复现论文模型及数值 |

Expected IG背景来自正常训练轨迹，不从测试正常片段采样；种子确定抽取16窗口。AFR中心由全部正常训练窗口的64维GRU表示平均得到。

AFR配置：1000步、归一化梯度、步长0.02、以原故障窗口为中心L2半径2000、clip[−1000,1000]，最小化64维表示到正常中心的MSE。记录baseline预测是否Normal及距离/损失，失败不静默换基线。

DeepLIFT对单变量非线性使用Rescale，GRU门乘法采用参考感知的对称RevealCancel/Shapley乘积分配。必须标识实现变体；前向一致和完备性自检不是与所有软件包输出完全一致的证明。

不同基线属于方法设定差异，对比并非只隔离“路径几何”；消融组才共享同一正常medoid。所有方法ADD/DEL则统一使用该medoid和真实故障窗口，避免扰动评价基线也随方法变化。

## 14. 四项核心指标的精确定义

### 14.1 根因标签及通道分数

依据ABIGX补充材料Table B.2的curated ground-truth root variables，当前代码中`TRUTH`为：

| 故障 | 根因通道 |
|---|---|
| Fault1、Fault2、Fault7 | XMV4，XMEAS4 |
| Fault3、Fault9 | XMV1，XMEAS2 |
| Fault4 | XMV10，XMEAS21 |
| Fault5、Fault12 | XMEAS22 |
| Fault6 | XMEAS1，XMV3 |
| Fault8 | XMEAS18，XMV8 |
| Fault10 | XMEAS18 |
| Fault11、Fault14 | XMV10，XMEAS9，XMEAS21 |
| Fault15 | XMV11 |

故障根因来自真实轨迹标签，即使目标logit预测错类，仍按真实故障根因评价并保留误分类标记。该表是curated根因集合，不是所有可能受影响下游变量的完整因果图。

旧PDF路径 `D:\Work\研究生毕业设计方面\ABIGX_supplemental items.pdf` 不在迁移包中。若需要逐字核验论文原表，应在新电脑提供原论文材料，不能谎称新电脑已打开PDF。

令B_j=Σ_τ|A_(j,τ)|，q_j=B_j/max(Σ_jB_j,1e−12)。不将正负贡献直接相加后取绝对值排序。

### 14.2 Correctness AUC：越高越好

33个通道作为33个二分类样本，根因集合G为正例，其他通道为负例，分数为q_j。当前计算等价于：

AUC = [Σ_(g∈G,n∉G) (1[q_g>q_n]+0.5·1[q_g=q_n])] / (|G|·|G^c|)。

它是根因通道排序ROC-AUC，不是分类器故障检测ROC-AUC，也不是ADD/DEL曲线面积。完全并列时AUC为0.5。

### 14.3 Correctness SUM：越高越好

SUM=Σ_(j∈G)q_j，表示绝对贡献质量分配到真根因通道的比例，范围[0,1]。它不是所有signed归因之和。全零归因在当前实现下SUM=0，应与诊断一起解释。

### 14.4 Consistency ADD：越高越好

按q降序得到通道π_1,…,π_33。第k步从正常W_b开始，把前k个通道的整段20点替换为故障W_f对应通道：

W_ADD^(k)=W_b+M_k⊙(W_f−W_b)，k=0,…,33。

取固定目标概率p_c(W)=softmax(F(W))_c，对34个点以横坐标k/33做梯形积分：ADD=∫_0^1 p_c(W_ADD(r))dr。

重要通道先加入若能更快提高故障目标概率，面积较大。这里原始概率曲线不做额外端点归一化或单调修正。

### 14.5 Consistency DEL：越低越好

从故障W_f开始，把前k个重要通道换回正常W_b：

W_DEL^(k)=W_f−M_k⊙(W_f−W_b)。

同样对p_c的34点曲线积分。重要通道先移除若能更快降低故障概率，DEL面积较小。当前DEL不是“下降量”，所以不要写成越高越好。

### 14.6 评价边界与权威性

ADD/DEL所有方法统一整通道原空间替换；路线2/Koopman的潜空间路径只用于生成排序归因，不用于替换评分窗口。这样隔离归因排序，避免不同方法的重构器改变评价函数。

原始通道替换可能破坏通道相关性并产生分布外混合窗口，这是共同评价局限，不能据此宣称干预因果效应。四项指标按论文概念适配窗口分类器，不自动等于ABIGX原始实验的逐项精确复现。

潜空间逐坐标替换不是逐原始特征替换，维度语义、解码误差和评价对象都会改变。除非用户另立补充实验，不要擅自恢复之前潜空间ADD/DEL方案。

### 14.7 辅助指标

- top3_hit：前3通道至少命中一个真根因，单轨迹为0或1；
- top3_coverage：前3命中的真根因数/该故障真根因总数；
- top3_precision：前3命中的真根因数/3；
- 分类ACC：全部test窗口中预测正确比例；Balanced ACC：有支持类召回率平均；Macro-F1：21类F1平均。

核心四项按14故障评价；分类ACC覆盖21类，不要误写成只在14类计算。辅助top-3保存在轨迹记录中，当前四指标主汇总不自动给出top-3五种子汇总图，若用户需要需明确追加。

## 15. 汇总、配对和图像

每轨迹每方法计算一次指标；先在每故障内平均50轨迹，再宏平均14故障，最后在5个独立训练种子上计算均值和样本标准差(ddof=1)。不是把所有时间窗口当作独立实验重复。

`summarize.paired_rows()`只保留同seed、fault、run同时具有全部10个方法记录的轨迹。主比较和消融因此共用完整方法配对集合；缺方法的轨迹不会独立留给某组。端点失败和方法中途失败需检查覆盖日志。

种子层箱线图是独立训练重复的展示；轨迹层箱线图只是描述性分布，轨迹嵌套于种子，不能把3500条轨迹当成3500个独立模型做显著性结论。当前没有自动配对显著性检验。

汇总路径 `Original_structure/Results/Summary/`：

- `per_trajectory_paired.csv`：配对轨迹数据；
- `per_fault.csv`、`per_seed.csv`、`mean_std.csv`；
- `Ablation_*boxplot.png/svg`：三条消融路线；
- `Method_comparison_*boxplot.png/svg`：8种方法对比；
- `classification_by_seed.csv`、`classification_summary.json`；
- `confusion_matrix_seed_*.png/svg`：Normal、Fault1–20，图上带数值；
- `feature_attributions/<method>/Fault_*.csv/png/svg`：signed/absolute均值；
- `coverage.json`与`status.json`：覆盖和是否最终完成。

归因平均顺序：signed先保留符号再平均；absolute先对逐时间贡献取绝对值再汇总，不能先平均signed再取绝对值。CSV使用UTF-8 BOM，可供Origin导入。逐轨迹图默认关闭以减少I/O，但逐轨迹原始归因和CSV仍保存。

## 16. 每条轨迹的结果在哪里

`Original_structure/Results/seed_<s>/explanations/Fault_<ff>/run_<rrr>/`：

| 文件 | 内容 |
|---|---|
| `endpoints.json` | run、窗口起点、真实故障、预测目标及候选数 |
| `endpoints.npz` | 真实起终点、观察初始化的window_starts |
| `<method>.npz` | attribution[33,20]、归一化feature_scores、ADD/DEL曲线 |
| `<method>_attribution.csv` | 33通道signed、absolute、normalized_absolute |
| `<method>.json` | 四指标、top3、target、classification_correct |
| `<method>_diagnostics.json` | 完备性、AFR/路径等诊断 |
| `route2_geodesic_ig_path.npz` / `koopman_ig_path.npz` | 原空间粗路径、潜路径、物理步差 |
| `*_optimization.csv` | 路径优化历史 |
| `afr_baseline.npy` | ABIGX自适应基线 |
| `complete.json` | 整轨迹已完成或端点失败记录 |

seed级`explanations/per_trajectory.csv`记录汇总轨迹行。方法JSON存在时可跳过该方法；轨迹complete存在时可跳过整条。若中途结束，重跑会重做未完整保存的当前方法，不保证任意积分点断点恢复。

## 17. 停止训练时的真实状态

以下是2026-10-25旧电脑静态快照，不是新电脑实时状态：

| 项目 | 状态 |
|---|---|
| seed42分类器 | 已完成16轮，最优checkpoint保存 |
| seed42共享AE | 已完成50轮 |
| seed42 train/validation h缓存 | 已完成，迁移包已含 |
| seed42 z16动力学 | 第42轮完整保存；第43轮中途停止 |
| z32、64、128 | 未开始 |
| seed43–46 | 未开始 |
| seed42端点 | 停训后仅推理生成700对，0失败 |
| 全部归因与最终统计 | 未开始，不存在可报告的最终四项指标 |

seed42分类test全部988,050窗口：ACC=0.9129477253，Balanced ACC=0.9344771429，Macro-F1=0.9353927739。它不是五种子均值。

z16第42轮记录：z_one_step≈0.00557562、z_ten_step≈0.04235560、raw_ten_step≈1.15246596、raw_reconstruction≈0.65653114、effective_rank≈2.66187。

第41轮为当时更好的验证score，`best_validation.json`对应effective_rank≈2.64544、selection_score≈1.80334839；不要把last第42轮与best第41轮混为一谈。

seed42 classifier SHA-256：`73eddfcb27a0eb8b070bbbadcd5a052a97e946659e6771972cdaf9a48042ec1b`。

seed42配置指纹：`2f501211f3c7d9cb68d6af984734964992478878421bc8d516e9e83aef14b350`。不要为改路径随意编辑config.json，否则旧checkpoint不再满足配置一致性。

最初完整上传提交：`e997fee073165ee93a97f3f63ec45f823a782de8`。RTX5060环境更新提交：`00c09cb18decc62eae9c6f46273d327db2775c19`。后续文档提交以`git log`为准。

## 18. 检查点、恢复和不保证的事项

分类/AE：`classifier.pt`、`manifold.pt`为最优state_dict，`*_last.pt`包含完整轮末模型、optimizer、scheduler、history、早停计数、NumPy/Torch/单设备CUDA RNG。

Koopman各维目录：`z_<d>/best.pt`为最优权重；`last.pt`含轮末model、optimizer、history、best、stale、epoch、NumPy和CPU Torch RNG；另含lift_pretrained、ridge_initialized、ridge_selection.csv、best_validation*、history.csv等。

Koopmanlast当前不保存CUDA RNG或Python random状态。不能宣称完整恢复所有随机源或任意batch级断点；跨PyTorch和显卡不保证逐位一致。

第43轮未完成的更新没有保存，续训从第42轮last恢复并重新执行第43轮。已完成分类/AE通过complete标记跳过；h缓存通过shape/complete标记复用；已完成维度通过complete跳过。

当前h缓存校验主要依赖标记与shape，不校验AE哈希；更换AE会使缓存无效。正常续训不要换AE或改预处理。若科研上要改，需另开版本并重建相应缓存/端点。

路径/归因保存JSON、NPZ后再写方法完成行；崩溃可能留下未标记文件，应核对完整性，不把孤立文件当完成证据。模型torch.save不是全套事务式归档，正在写入时强杀可能损坏该文件，应尽量轮末安全停止并校验。

## 19. RTX5060环境与首次运行

### 19.1 环境

新环境：Python3.11.15、torch2.7.1+cu128、numpy1.26.4、matplotlib3.7.0。RTX5060属于Blackwell架构，不沿用旧PyTorch2.5.1+cu124。

官方支持依据：<https://pytorch.org/blog/pytorch-2-7/>；Windows/Python3.11的cu128 wheel索引：<https://download.pytorch.org/whl/cu128/torch/>。

安装适配5060的较新NVIDIA驱动。PyTorch wheel自带所需运行时，一般不用单独安装完整CUDA Toolkit，也不需要torchvision、torchaudio、Captum、rdata。不能把`nvidia-smi`显示的CUDA能力上限直接当作PyTorch实际运行时版本。

旧电脑环境：Python3.11.15、torch2.5.1+cu124、MX350 2GB。保留旧环境清单作追溯，不要求把旧环境所有无关包全部安装到新电脑。

### 19.2 在哪里执行

Windows开始菜单打开Miniconda Prompt或Anaconda Prompt，不是在Python的`>>>`里。CMD切盘符需要`cd /d`：

```bat
cd /d D:\luo_WORK
git clone https://github.com/viclaw0902/Koopman-XAI.git
cd Koopman-XAI
conda env create -f environment.yml
conda activate koopman_xai
python restore_transfer.py
python check_environment.py
```

仅当用户明确要求续训、且自检通过后再运行：

```bat
python start_resume.py
```

`start_resume.py`本身也先做一次自检，然后启动完整队列。前置单独自检方便定位问题。

已克隆时不要重复clone，可在仓库目录`git pull --ff-only`。已用旧环境创建时激活环境后`python -m pip install -r requirements-cu128.txt`。已有新训练结果时不要重新restore覆盖。

### 19.3 自检检查什么

`check_environment.py`默认要求CUDA；加载真实seed42分类器、实际验证窗口、显式GRU DeepLIFT；加载z16/last、恢复optimizer，在内存前向/反向更新并执行谱计算；输出GPU名称、计算能力、Torch/CUDA版本、saved_epoch、next_epoch和归因完备性。

它不保存测试更新后的模型，但WindowStore会写入标准化校正文件，因此不是严格零写入检查。`--allow-cpu`只是允许CPU核验，不能当作5060可用证明。

已在旧电脑MX350/torch2.5.1运行通过：checkpoint和optimizer可读；DeepLIFT完备性绝对误差约3.8e−6。RTX5060及新torch版本必须由新电脑实测，不可写“已通过5060测试”。

### 19.4 数据路径与快照恢复

程序优先查仓库根目录`rieth_prepared/`，其次旧config里的prepared_dir。若设置`KOOPMAN_DATA_DIR`，以该显式路径为准。不要修改config的旧绝对路径只为迁移。

分卷共20个（prepared_data14份、resume_seed42 6份），每份≤45MiB，总压缩约923MB。恢复工具依次验证分卷、拼接ZIP和解压文件SHA-256，并拒绝覆盖不同内容的现存文件。`--verify-only`只校验分卷，不解压。

运行目录里不再需要原电脑数据盘。Git仓库不自动跟踪解压后的Results或prepared数据；GitHub保存的是停止时固定快照，新电脑续训结果不会自动上传。

## 20. 队列顺序、监控与故障处理

### 20.1 队列

每个seed顺序执行：train.py → explain.py → summarize.py，再进入下一seed。不是先训练完5个seed才统一解释。每seed训练内顺序：分类→AE→h缓存→d16/32/64/128→选维。

`start_resume.py`为前台等待进程，子训练输出主要重定向到Results/logs，终端看起来安静不代表没有训练。关闭窗口、休眠或重启会中断，运行时保持电脑供电、不休眠。

实时查看PowerShell命令（在仓库根目录）：

```powershell
Get-Content .\Original_structure\Results\pipeline_status.json
Get-Content .\Original_structure\Results\seed_42\status.json
Get-Content .\Original_structure\Results\logs\train_42.stdout.log -Tail 15
Get-Content .\Original_structure\Results\logs\train_42.stderr.log -Tail 30
```

需要持续观察可以`Get-Content <日志路径> -Tail 10 -Wait`，Ctrl+C结束的是日志观察，不必然终止训练。

给用户汇报进度应包含seed、阶段、d、epoch、已处理/总窗口及最近日志时间，不能仅用GPU占用或文件存在推测完成百分比。

### 20.2 锁

`Results/pipeline.lock`用于避免双队列。存在锁时先核对当前PID/命令行，不要直接删除。只有确认对应训练和runner均已退出，且用户要求恢复时，才将陈旧锁改名备份后重启。

旧电脑停止时已处理旧锁；新克隆不应复用旧进程PID。不要在新电脑使用旧`--adopt-pid`。

### 20.3 常见错误

| 现象 | 优先检查 |
|---|---|
| git/conda不是命令 | Git/Miniconda是否安装，是否打开正确Prompt |
| cd后仍在C盘 | CMD改用`cd /d D:\luo_WORK` |
| CUDA unavailable / kernel不支持 | 驱动、当前环境、torch是否为cu128；不要静默转CPU |
| Prepared data not found | 是否完成restore，KOOPMAN_DATA_DIR是否指错 |
| SHA不匹配 | 下载是否完整；不要绕过校验 |
| Refusing to overwrite | 已有文件与旧快照不同；保留新结果，不强行覆盖 |
| Config changed | 比较原配置和检查点指纹，不直接删除保护 |
| all candidates collapsed | 查看所有d有效秩及2%选择带，先审查再讨论 |
| 文件有但Summary不完整 | seed/归因是否全部完成，检查覆盖与失败日志 |
| GPU更快但仍耗时 | 全窗口量大、CPU内存搬运/验证/AFR/路径/IG也耗时 |

不要未经确认增batch、改步长、减少轨迹或减种子。硬件升级不自动授权改变评价协议；若要加速，应先区分不改变数学计算的工程优化和会改变训练结果的实验调整。

## 21. 已知局限与下一位Codex必须避免的误读

1. **低秩风险已出现**：先报告，不能声称维度扫描已证明线性化成功。
2. **路径非严格物理轨迹**：固定端点桥一般有动力学残差；Koopman路径不是已证明的纯黎曼测地线。
3. **长时域问题**：K仅受1–10步训练约束，解释段可能跨数百点，用K^Δt会外推；谱半径1.05仍允许长时增长。
4. **模式切换问题**：训练排除了正常→故障切换链，解释却对整段固定故障K，物理上不保证能精确表示注入过程。这是当前已选方案的局限，不可隐瞒。
5. **重构误差**：连接段可能贡献显著，需检查其比例，不能把连接段贡献全部归功于动力学。
6. **数值收敛有限**：path optimization_converged是代码启发式标记，不是全局最优证明；integration_converged与路径收敛是两个不同诊断。
7. **不同方法基线不同**：主对比混合了基线与算法差异，三路消融才共享基线。
8. **实现适配**：ABIGX、Guided IG、DeepLIFT等为当前代码具体实现；没有验证便不能声称与官方所有实现逐位一致。
9. **标签不进入分类器前向**：测试真实故障用于评价分组，不用于Oracle选K。不要把Predicted-K改为Oracle-K。
10. **共享分类器**：归因方法之间ACC相同是设计要求，不是复制结果的错误；不同种子ACC可不同。
11. **归因根因不等于因果证明**：TEP表中只列curated根因，传播变量贡献可能合理却降低根因指标。
12. **迁移不等于完成**：当前没有最终四项指标、最终箱线图或所有种子完成证据。

这些风险需在结果讨论中呈现。若用户要求改进，先说明改变的假设、对比条件和目录，确认后另开实验；不要在续训途中静默换方法。

## 22. 历史决策压缩记录

项目经历过XMEAS-only、33维、受控/自主、单K/多K、无流形、线性投影/时序卷积AE、不同故障子集和不同ADD/DEL定义。旧版本结果不能混为一个实验序列直接比较。

当前Final_exp有效决策链：

1. 最终输入回到33维，XMEAS/XMV共同参与；自主Koopman10步纯递推。
2. 分类器共享，避免联合动力学梯度改变分类器后，把分类差异误当路径优势。
3. 保留历史多K时序卷积AE并行结构；路线2与Koopman共享同一AE。
4. 从z返回h必须独立网络，禁止截取前16维。
5. 分类用全部可用21类；只在指定14个有根因表的故障上做四项归因评价。
6. 端点改为同轨迹注入前/后候选的真实medoid，不再随机单条轨迹或连续三个正确窗口。
7. 固定故障端点预测类别logit与K；不同路径解码后统一输入相同CNN-GRU。
8. ADD/DEL经过讨论后回到统一原通道替换，不用潜坐标替换作主指标。
9. d在验证集扫描16/32/64/128，五种子42–46，输出均值/标准差和箱线图。
10. 用户要求换电脑，旧训练停止、保存状态和700端点，提交迁移仓库；新GPU5060改cu128环境。

此前C-MAPSS、HAI、SMD、CausRCA等是其他项目探索，不属于当前TEP Final_exp执行范围。本仓库不包含它们全部脚本和结果，不要自称继承了可运行的所有跨数据集实验。

## 23. 接手核对清单与完成标准

第一次接手：

- 完整阅读本文件、根README、MIGRATION_STATUS、config和Original_structure/README；核对`git status`及最新提交；
- 查新电脑当前环境、GPU、文件、日志、进程，不依据旧快照推断正在训练；
- 检查是否恢复数据及seed42检查点，是否有新电脑更晚的结果；
- 按用户当前请求决定只解释、诊断、续训还是修改，不把历史聊天请求当自动执行命令；
- 需要续训时先自检通过，保持配置，保证一个活跃队列；
- 运行失败时报告真实错误和阶段，保护现有检查点。

全部实验完成必须同时确认：五个种子训练和选维完成；每seed全部计划轨迹归因处理完并报告失败覆盖；四项指标与分类统计汇总；两类箱线图与特征图生成；pipeline状态complete；Summary状态不再标记未完成。仅classifier_complete或某个z/complete不等于全部完成。

## 24. 在新电脑发给Codex的建议开场语

> 请先完整阅读当前仓库的AGENTS.md和TECHNICAL_HANDOFF.md，再核对README.md、MIGRATION_STATUS.json、Original_structure/config.json与实际代码、日志、进程。这里是TEP Koopman-XAI最终实验的迁移项目，新电脑显卡RTX5060。请区分旧电脑2026-10-25的静态进度与本机实时状态，先告诉我你理解的实验设计、已完成阶段、风险和下一步。不要先改配置、覆盖结果或启动重复训练；如我要求继续训练，通过环境自检后从有效检查点恢复。

该开场语配合文件提供项目连续性，不代表新Codex获得了完整原始对话。遇到本文件未记录的重要决策，先查当前代码及日志；仍无法确定时明确询问用户，不凭空补全。
