<!-- Keep the project architecture image here; do not remove the image file. -->
<p align="center">
  <img width="1200" alt="ORPHEUS-architecture" src="https://github.com/user-attachments/assets/fa1683ea-cda9-490c-9d46-db464e010913" />
</p>

# ORPHEUS

ORPHEUS — Optimized Representation of Protein-ligand Hybrid Ensembles via Unified flow-matching and Structure-distillation

项目概述
----------------
ORPHEUS 是一个面向酶-底物识别与特异性预测的研究与工程平台，目标是将“物理直觉 + 高维结构信息 + 蒸馏学习”结合，训练出既有高准确性又计算高效的在线推理模型。

项目初步架构
----------------
Part 1: 数据引擎铸造
- 核心任务：从一维序列生成可用于训练教师模型的高精度“真值”数据集（ORPHEUS-DB），并包含动力学信息。
- 两大生成模型：
  - 酶结构生成：AlphaFold 3 / ESMFold，将 UniProt 序列批量转为高置信度 3D 静态结构。
  - 复合物构象生成：Mtchaa / FlowDock 风格的流匹配模型，生成酶-底物结合构象系综，提取推理过程中的速度场（velocity flow fields）。

Part 2: 输入层
- 节点：Input Data（Enzyme Sequence + Substrate 2D Graph）。
- 设计哲学：极低门槛，用户只需提供序列（FASTA）与底物的 SMILES（或 2D 图）即可。

Part 3: 教师体系（Offline，重算力）
- 3.1 几何教师（Geometry Teacher）
  - 模型：Interaction-GCP-VQVAE + SaProt（结构感知的蛋白语言模型）。
  - 功能：用 SE(3)-equivariant GNN 提取稳健的局部/全局几何特征，并将局部构象压缩为离散的几何 codebook tokens。

- 3.2 物理教师（Physics Teacher）
  - 模型：Riemannian Flow Matching（基于 FlowDock/Mtchaa 架构）。
  - 功能：在黎曼流形上模拟底物如何“流”入活性口袋，提取速度场 vt(x) 作为动力学特征。

- 3.3 能量教师（Energy Teacher）
  - 模型：AutoDock Vina + Uni-KP。
  - 功能：提供基于物理和统计的结合能、动力学参数（kcat、Km）作为监督信号。

Part 4: 学生体系（Online，轻量推理）
- 核心模型：ORPHEUS Student（Cross-Modal Transformer），结合轻量的 ESM-2（序列）与 GNN（分子图）。
- 知识蒸馏：学生通过多任务学习预测几何 token、速度流场和亲和力分数，从而在无 3D 输入下恢复教师级别的信息表现。

Part 5: 输出层
- 节点：Final Prediction（Specificity Score + Interpretability）。
- 输出：酶-底物特异性概率（0-1）和可解释性信息（attention 高亮的突变位点/关键残基）。

使用场景
----------------
- 酶定向进化筛选
- 代谢通路挖掘
- 生物催化剂筛选与设计

如何贡献
----------------
- 本仓库包含三个主要子项目目录：`Student`、`Physics_Teacher`、`Geometric_Teacher`。
- 请在提交前确保不添加大型模型权重或数据（将大文件保存在外部存储或通过下载脚本提供）。

版权与许可
----------------
请参见仓库根目录中的 `LICENSE` 文件。

联系方式
----------------
如需讨论技术细节或合作，请联系作者：spongelovesorange
