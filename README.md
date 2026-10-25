# Koopman-XAI: TEP experiment migration snapshot

这是2026-10-25停止训练后保存的完整迁移快照。原电脑训练及队列均已停止。**不是已完成实验，不应把阶段结果当作最终结果。**

## 当前保存进度

| 阶段 | 状态 |
|---|---|
| seed42 CNN-GRU | 已完成；测试ACC0.9129477、Balanced ACC0.9344771、Macro-F1 0.9353928 |
| seed42共享时序卷积自编码器 | 已完成50轮 |
| 训练/验证h缓存 | 已保存并打包 |
| seed42、z16 Koopman | 完整保存至第42轮；继续时从第43轮开始 |
| seed42解释端点 | 14类×50轨迹＝700对；0条失败 |
| z32/z64/z128 | 未开始 |
| seed43–46 | 未开始 |
| 归因与最终对比 | 未开始 |

第43轮中途停止，未保存的部分更新不保留；从第42轮完整检查点恢复。已保存模型参数、优化器状态、训练历史和原程序记录的随机数状态。不同显卡/库版本可能产生数值差异，不承诺逐位一致。

## 文件结构

```text
Original_structure/       本次原结构B实验；脚本可独立运行，不依赖旧TEP项目
Changed_structure/        预留空目录，由restore_transfer.py创建，不运行任何实验
artifacts/                数据和检查点的分卷，每个不超过45MiB，无需Git LFS
transfer_manifest.json    所有分卷和原始文件的SHA-256与大小
environment.yml           推荐Conda环境，含PyTorch2.5.1 CUDA12.4
requirements-cu124.txt    等价pip安装入口
environment/              原电脑环境版本清单，仅供追溯
restore_transfer.py       拼接、校验、恢复数据和训练状态
check_environment.py      CUDA、实际分类器、GRU归因、优化器迁移自检
start_resume.py           自检成功后才开始完整续训队列
MIGRATION_STATUS.json     迁移阶段说明
```

Git不能保存空文件夹，所以Changed_structure在克隆后由恢复工具创建，保持为空。大文件以ZIP分卷提交，避免GitHub单文件100MB限制。不要手工只解压一个part；恢复工具会按清单拼接。

## 新电脑操作（Windows NVIDIA GPU）

安装Miniconda和兼容NVIDIA驱动。使用Anaconda/Miniconda Prompt：

```powershell
git clone https://github.com/viclaw0902/Koopman-XAI.git
cd Koopman-XAI
conda env create -f environment.yml
conda activate koopman_xai
python restore_transfer.py
python check_environment.py
python start_resume.py
```

仓库可放在任意磁盘路径。恢复后数据位于仓库根目录rieth_prepared，程序优先读取该位置；不要修改config.json中的旧绝对路径，因为训练检查点会校验配置。若想把数据放到别处，可设置环境变量KOOPMAN_DATA_DIR指向prepared数据目录。它不会改变训练配置指纹。

`restore_transfer.py`只恢复不存在或与快照完全相同的文件；发现新训练产生的不同文件会拒绝覆盖。不要在续训后再次强行恢复旧快照。所有分卷都提供SHA-256校验。新克隆不带旧pipeline.lock或旧进程PID。

如不使用Conda YAML，也可：

```powershell
conda create -n koopman_xai python=3.11.15 pip -y
conda activate koopman_xai
python -m pip install -r requirements-cu124.txt
```

原环境关键版本：Python3.11.15、torch2.5.1+cu124、numpy1.26.4、matplotlib3.7.0。不必复制整套虚拟环境目录，也不需要安装旧工程所有包、Captum、rdata或完整CUDA Toolkit。本实验只读取已准备好的数组。

### 新显卡兼容性

RTX50/Blackwell等新GPU可能无法使用PyTorch2.5.1+cu124。此时按https://pytorch.org/get-started/locally/安装支持新显卡的PyTorch CUDA构建，仍使用Python3.11和本仓库numpy/matplotlib版本，再执行check_environment.py。检查失败时不要开始训练。环境变化需保留版本记录，跨版本不承诺完全相同数值。

check_environment会加载真实checkpoint、执行一次GRU前向/归因、恢复优化器并在内存做一次测试更新，不保存测试后的权重。默认要求GPU；仅检查文件时可使用`python check_environment.py --allow-cpu`，但这不代表已验证新GPU可用。

## 恢复后实际会做什么

1. 跳过seed42已经完成的分类器和流形训练。
2. 复用train_h.npy和validation_h.npy，避免重复编码。
3. 读取seed42/z_16/last.pt，从第43轮继续；最优模型best.pt仍保留。
4. 继续32、64、128维验证扫描，按原规则选维。
5. 解释时复用保存的700对端点，使用分类器SHA-256确认没有混用模型。
6. 执行归因和指标，继续seed43–46，最后汇总均值/标准差与箱线图。

更换硬件后不要直接增大batch或修改配置，否则会影响可比性并触发配置校验。先保持batch128、累积2和原实验配置。保持电脑不休眠。当前队列是前台进程，关闭终端会中断；必要时使用自己的终端会话管理。

## 端点文件

恢复后在Original_structure/Results/seed_42中：

- endpoint_selection.json：700条轨迹的起终点索引、真实故障、预测目标、候选数量和分类器哈希；
- endpoint_selection.csv：同样的端点元数据；
- endpoint_windows.npz：700×2×33×20的标准化后原始特征窗口；
- endpoint_selection_complete.json：数量与完整性标记。

每条轨迹取完全注入前正常窗口的medoid作为起点；终点是完全注入后且预测为任意Fault窗口的medoid。距离在共同标准化原始空间计算，目标logit/K根据故障medoid预测选择并固定。错误分类不以真实标签替换。

这些端点是停止训练后用已完成分类器执行推理选出的，不是额外训练，也不是已经计算出的IG。新种子的分类器不同，因此各自选择端点，不盲目复用seed42端点。

## 模型和评价协议

原结构B：窗口分别进入CNN-GRU与共享时序卷积AE。h16经独立升维映射得到z；**全部z坐标输入独立训练的逆映射z->h**，再经共享解码器回原始窗口，禁止截取前16维。自主模型z(t+1)=K_c z(t)，21类各一个K，无单独XMV控制项。分类器和AE在动力学训练时冻结。

输入33维，窗口20/步长1，分类使用Normal+Fault1–20；归因只评价Fault1–12、14、15。每类50测试轨迹，5个种子42–46。所有方法解释同一分类器目标logit，ADD/DEL使用目标Softmax概率和统一原始通道替换。路线2/Koopman曲线路径解码后逐点进入相同CNN-GRU计算原始输入梯度。

方法对比：Saliency、IDGI、零基线Vanilla IG、Guided IG、Expected IG、DeepLIFT、ABIGX、Koopman IG。消融：正常medoid直线IG、共享流形测地线IG、Koopman IG。详细实现与局限见Original_structure/README.md。

仍须关注z16的有效秩诊断（第42轮约2.66）；本快照不修改原选维条件、不掩盖问题。若所有候选均不满足防塌缩条件，原程序会停止，需要审查后再决定，不会自动放宽以生成结果。

## 数据来源与公开范围

数据是本项目已有Rieth扩展版TEP的33通道prepared数组，原始RData不包含在仓库。原数据划分和标准化来源记录在data_audit.json与数据summary.json；本次仅打包用户指定实验所需文件，不含用户目录、凭据、整个旧工程或无关数据。

迁移脚本只改路径解析与保存端点的复用方式，不改变训练损失、网络层或已保存模型。恢复工具不运行训练，只有显式执行start_resume.py才开始。
