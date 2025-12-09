# Experiment Plan: Pocket-Aware Geometry Teacher Generation

**Project:** ORPHEUS (Geometry Distillation Module)
**Dataset:** PDBBind (Refined/General Set)
**Model:** GCP-VQVAE (Pre-trained on monomer backbones)

## 1. 实验目标 (Objective)

构建“几何老师”数据流水线。利用 PDBBind 数据集中的**全复合物 (Holo-Complex)** 结构，提取底物结合口袋区域的**离散几何 Token (Discrete Geometric Tokens)**。

这些 Token 将作为“真值 (Ground Truth)”，用于训练轻量级学生模型（Student Model）根据 Sequence + SMILES 预测结合口袋的三维几何形状。

## 2. 核心原理 (Rationale: Why do it this way?)

  * **诱导契合 (Induced Fit):** 我们使用的是复合物（Holo）结构而非空载（Apo）结构。这意味着 VQVAE 看到的蛋白质骨架已经被底物“撑开”或“诱导”成了特定的形状。学生模型通过学习这些 Token，实际上是在学习“当遇到这个 SMILES 时，蛋白质会变成什么样子”。
  * **注意力聚焦 (Attention Focusing/Masking):** 全长蛋白质中，远离口袋的区域通常对特异性结合贡献很小（那是噪音）。我们只保留底物周围（如 $6\mathring{A}$）的 Token，强迫学生模型专注于预测**活性位点**的几何构象。
  * **离散化 (Quantization):** 使用 GCP-VQVAE 的 Codebook [cite: 20, 121]，将难以预测的连续坐标转化为 4096 分类的 Token 预测任务，大幅降低学生模型的训练难度。

## 3. 数据准备 (Prerequisites)

  * **Input Data:** PDBBind Dataset (包含 `*_protein.pdb` 和 `*_ligand.sdf/mol2`)。
  * **Tool:** GCP-VQVAE 官方仓库（需加载预训练权重）。
  * **Libraries:** `biopython` (解析 PDB), `rdkit` (解析配体), `scipy` (计算距离), `torch`.

## 4. 详细步骤 (Implementation Pipeline)

请按照以下四个步骤编写脚本 `generate_geometry_labels.py`。

### Step 1: Data Parsing (数据解析)

读取 PDBBind 的目录结构。每一对样本需要加载：

1.  **Protein Structure:** 解析 PDB，提取 Backbone Atom ($N, C_\alpha, C$) 的坐标。
2.  **Ligand Structure:** 解析 SDF/MOL2，提取所有重原子的坐标。
3.  **Sequence:** 从 PDB 中提取氨基酸序列（用于后续学生输入）。
4.  **SMILES:** 从配体文件中提取 SMILES（用于后续学生输入）。

### Step 2: Pocket Masking (空间掩码生成)

这是核心逻辑。我们需要找出哪些残基属于“口袋”。

  * **Logic:** 计算 Protein $C_\alpha$ 原子集合与 Ligand 原子集合之间的距离矩阵。
  * **Criteria:** 对于第 $i$ 个残基，如果其 $C_\alpha$ 到任意配体原子的最小距离 $< 6.0 \mathring{A}$，则标记 `Mask[i] = True`。
  * **Output:** 一个布尔列表 `is_pocket_residue`，长度等于蛋白序列长度。

### Step 3: VQVAE Inference (几何编码)

将**全长蛋白质骨架**输入 GCP-VQVAE 模型。

  * **Input:** Protein Backbone Coordinates (Shape: `[1, Length, 3, 3]`).
  * **Process:**
    1.  运行 `model.encode()` 得到 continuous embeddings。
    2.  运行 `model.quantizer()` (或查看代码中获取 indices 的方法)。
  * **Output:** 全长 Token 序列 (Shape: `[Length]`)，包含 0-4095 的整数 [cite: 20]。
      * *注意：这里输入的是全长蛋白，因为 VQVAE 需要上下文来正确编码局部。我们是在获得输出后才进行过滤。*

### Step 4: Label Filtering & Saving (标签过滤与保存)

结合 Step 2 的 Mask 和 Step 3 的 Token。

  * **Logic:**
    ```python
    final_labels = {}
    for i in range(length):
        if is_pocket_residue[i]:
            final_labels[i] = all_tokens[i] # 只保存口袋区域
        else:
            final_labels[i] = -100 # PyTorch Ignore Index (可选，或直接不存)
    ```
  * **Save:** 保存为 `.pt` (Torch) 或 `.jsonl` 文件。
      * Format: `{"pdb_id": "1a2b", "seq": "...", "smiles": "...", "pocket_labels": {45: 1024, 46: 5...}}`
