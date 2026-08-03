# Efficient Video Diffusion：算法原理、负载机理与系统资源影响

**副标题：以 Efficient Video Diffusion Models: Advancements and Challenges 为主线的论文详述版**  
**研究范围：本地“paper/负载”三篇论文，并补充截至 2026 年 8 月可检索的公开研究**  
**版本：2026-08-03**

## 执行摘要

Video diffusion 的部署瓶颈不是一个孤立的“FLOPs 太高”问题，而是三个乘数共同造成的系统压力：时空 token 数量 N、去噪网络调用次数 NFE、以及每次调用中的参数与中间状态搬运。对 DiT 而言，线性层成本近似随 N 增长，而全注意力的计算和未优化注意力矩阵存储随 N² 增长；扩展分辨率、帧率、时长中的任意一项都会放大 N，扩展时长还会增加缓存和跨卡通信。因此，视频相较图像并非“多几帧”，而是同时引入长序列、迭代采样和时间一致性约束。

本报告的主要结论如下。

1. **负载趋势由算力主导转向算力、HBM 容量和带宽共同主导。** 短序列、低分辨率时，GEMM 更偏计算受限；长视频时，attention、归一化、残差、量化/稀疏路由、跨卡 all-to-all 及 CPU-GPU offload 更容易成为带宽或通信受限。只报告 FLOPs 会系统性高估真实加速。
2. **token 压缩是最强的“每步成本”杠杆。** FSVideo 将 VAE 压缩提高到 64×64×4，并使用 128 个 latent channels；按输入视频与 latent 张量的元素数量计算，总体压缩比为 384:1。对 5 秒、720×1280、24 fps（121 帧）的示例，若先按 VAE latent 网格单元计算，8×8×4 VAE 约产生 446,400 个网格单元，而 64×64×4 VAE 在向上取整后约产生 7,440 个网格单元，约减少 60 倍；忽略边界取整时，理论空间缩减为 64 倍。latent 网格单元经过 patchify 后才形成 DiT token，因此实际 token 数还取决于 patch 大小。FSVideo 在 2×H100 上报告从 Wan2.1 的 822.1 s 降到 19.4 s，即 42.3×，说明减少每 NFE 成本可以远大于仅减少采样步数。
3. **减少 NFE 是最强的“采样深度”杠杆。** 一致性、分布和对抗蒸馏可把几十步压到 4–8 步甚至一步，理论上近似按 NFE 成比例降低主干计算和参数读取次数。但一步化容易出现轨迹估计误差、运动退化和训练不稳定；工业上 4–8 步通常更稳妥。
4. **稀疏注意力的理论降幅与端到端收益之间存在显著折损。** RainFusion2.0 在 80% sparsity 下实现 1.5–1.8×端到端加速；Sparse-vDiT 的理论 FLOP 降幅为 1.67–2.38×，真实推理加速为 1.58–1.85×。原因包括路由开销、非 attention 模块、kernel 启动、负载不均和不规则访存。2026 年 HASTE 通过 mask reuse 和 head-wise budget 将 720p 加速推至最高 1.93×，表明研究重点已从“找稀疏性”转向“低开销、硬件可执行的稀疏性”。
5. **量化首先解决容量和带宽，再决定能否兑现算力收益。** 14B 参数在 BF16 下仅权重约 28 GB，FP8/INT8 约 14 GB，4 bit 约 7 GB。权重常驻 HBM 时，量化释放显存容量并降低权重读取带宽；存在 offload 时，收益更大，因为它减少 PCIe/NVLink 传输量。6Bit-Diffusion 报告内存占用降至基线的约 1/3.32，并实现 1.92×端到端加速，说明动态混合精度与 cache 联用比静态低比特更符合 diffusion 的 timestep 非平稳性。
6. **缓存以存储换算力，必须区分 feature cache、KV cache 和轨迹复用。** cache 减少重复 block/NFE，但增加显存占用、读写带宽和长时漂移风险。短视频可能受益显著；流式长视频若 KV 随历史线性增长，会把算力瓶颈迁移为容量与带宽瓶颈，需要窗口化、压缩、刷新和驱逐策略。
7. **组合加速应按“乘法潜力、误差加法、瓶颈迁移”设计。** NFE 压缩、token 压缩、低比特和结构化稀疏在理想情况下近似相乘；但误差会叠加，且主干加速后 VAE decoder、文本编码器、通信和调度占比上升。Flash-VAED 的约 6× decoder 加速最终只带来最高 36% 端到端提升，正是 Amdahl 定律下瓶颈迁移的直接证据。

综合判断：面向可部署系统，优先级通常应为 **latent/token 压缩 → 4–8 步蒸馏 → BF16 到 FP8/INT8 的硬件原生量化 → 规则块稀疏/低开销动态稀疏 → cache 与分布式通信优化 → VAE decoder 优化**。具体顺序需由 profile 决定；若模型无法常驻显存，量化和模型布局应提前到第一优先级。

## 1. 研究材料、口径与证据等级

### 1.1 本地材料

本报告完整阅读并交叉使用了以下三篇本地论文：

- RainFusion2.0: Temporal-Spatial Awareness and Hardware-Efficient Block-wise Sparse Attention，arXiv:2512.24086v2，6 页。
- FSVideo: Fast Speed Video Diffusion Model in a Highly-Compressed Latent Space，arXiv:2602.02092v1，22 页。
- Efficient Video Diffusion Models: Advancements and Challenges，arXiv:2604.15911v1，66 页。

第三篇是综述性材料，用于建立方法分类；前两篇用于提供硬件实测和具体设计。联网补充仅采用论文主页或项目的一手信息，重点覆盖 2026 年的新进展：HASTE、6Bit-Diffusion、StreamFusion、Causal-RoPE SP 和 Flash-VAED。

### 1.2 指标口径

报告区分四类指标，避免不等价比较：

- **硬件无关成本**：参数量、token 数、NFE、FLOPs/MACs、理论 attention complexity。
- **设备内成本**：峰值 VRAM、HBM bytes、kernel latency、利用率、算术强度。
- **跨设备成本**：all-to-all/all-gather/reduce-scatter 数据量、互联带宽、同步等待。
- **端到端指标**：首帧时延、单视频时延、吞吐、实时因子、能耗及质量指标。

论文中的 speedup 仅在相同模型、分辨率、帧数、精度、NFE、并行度和 kernel 环境下可直接比较。凡条件不完整的数字，本报告只作为趋势证据，不作绝对排序。

## 2. Video Diffusion 的统一负载模型

### 2.1 token 数是第一状态变量

对输入视频 F×H×W，设 VAE 时间下采样率 r_t、空间下采样率 r_h 和 r_w，DiT patch 尺寸为 p_t×p_h×p_w，则视觉 token 数近似为：

N ≈ ceil(F/r_t) × ceil(H/(r_h p_h)) × ceil(W/(r_w p_w))。

> **公式 1：视觉 token 数**　N = ⌈F/r_t⌉ · ⌈H/(r_h p_h)⌉ · ⌈W/(r_w p_w)⌉

其中，F、H、W 分别表示输出视频的帧数、像素高度和像素宽度；r_t、r_h、r_w 是 VAE 在时间、高度和宽度方向的下采样率；p_t、p_h、p_w 是 DiT patch 尺寸。若时间方向也进行 patchify，应将第一项进一步除以 p_t。VAE 输出的是 latent 网格，网格经过 patchify 后才得到送入 DiT 的视觉 token。这个公式最重要的含义是：**Video Diffusion 的主干工作量首先由视觉 token 数决定，而不是直接由输出像素总数决定。**

文本 token 和条件 token 通常远小于视频 token，但 MMDiT 将多模态 token 拼接后做联合 self-attention，仍会提高序列长度和投影成本。实际模型还会因 causal VAE 的首帧边界、padding 和 patchify 规则产生差异。

由上式可见：分辨率各边放大 2×，N 约放大 4×；时长或帧率放大 2×，N 约放大 2×。在 full attention 区域，前者可使 attention score 计算放大约 16×，后者约 4×。这解释了为什么从 480p 到 720p、从 5 秒到长视频会出现非线性时延与显存增长。

### 2.2 单层 DiT 的算力构成

忽略常数和分组细节，隐藏维度为 d、FFN 扩张比为 m，则一个 transformer block 的主要计算可写为：

- QKV 与输出投影：约 4Nd²；
- 全注意力 score 与 value aggregation：约 2N²d；
- FFN：约 2mNd²；
- 归一化、激活和残差：O(Nd)，FLOPs 不高但可能带宽受限。

> **公式 2：单个 Transformer block 的近似 FLOPs**　C_block ≈ (4 + 2m)Nd² + 2N²d

其中，d 为隐藏维度，m 为 FFN 扩张比，N 为 token 数。第一项包含 Q/K/V/输出投影与 FFN，随 N 线性增长；第二项是 attention score 和 value aggregation，随 N² 增长。

为了判断究竟是线性层还是 attention 主导，可比较两项大小。令二者相等：

> **公式 3：attention 转为主导的大致临界点**　N* ≈ ((4 + 2m)/2)d = (2 + m)d

例如 m=4 时，N*≈6d。若 d=4,096，则临界 N 约为 24,576。N 明显低于该值时，大模型的投影与 FFN 可能仍是主要算力；N 明显高于该值时，N² attention 会迅速占据主导。这个近似忽略了 FlashAttention、GQA、cross-attention 和实际 kernel 效率，但比笼统地说“attention 一定是瓶颈”更准确。

总推理成本再乘 block 数 L 和 NFE S。Classifier-free guidance 若以两次前向实现，会使每个 diffusion step 对应约 2 NFE；CFG distillation 可以消除这一倍数。

> **公式 4：DiT 主干总算力**　C_DiT ≈ S · L · [(4 + 2m)Nd² + 2N²d]

若使用双前向 CFG，可令 S≈2S_step；若已经做 CFG distillation，则通常 S≈S_step。该公式说明三种最强杠杆分别是：减少 S（少步蒸馏）、减少 N（VAE/token 压缩）和减少 attention 的有效连接数（稀疏 attention）。

一个重要判断是：并非所有 Video DiT 都始终由 N² attention 主导。当 d 很大、N 经 VAE/patchify 显著压缩时，Nd² 的线性层和 FFN 仍可能占主导；当 N 随分辨率和时长增大时，attention 迅速接管。这也是同一种 sparse attention 在不同模型和分辨率上 speedup 不同的原因。

### 2.3 显存与持久化存储

推理峰值显存可分为：模型权重、临时 activation、attention workspace、条件编码、VAE 中间特征、通信 buffer、cache 和 runtime fragmentation。权重下界为 P×b_w/8。以 14B 为例：

> **公式 5：推理权重容量下界**　M_weight = P · b_w / 8

P 为参数量，b_w 为每个权重的位数，结果单位为 byte。以 14B 参数为例，BF16 为 14×10⁹×16/8≈28 GB；FP8/INT8 约 14 GB；4 bit 约 7 GB。这只是权重下界，不包含量化 scale、对齐填充、activation 和运行时 workspace。

| 权重格式 | 理论权重体积 | 相对 BF16 | 主要影响 |
|---|---:|---:|---|
| FP32 | 56 GB | 2.0× | 通常用于 master weights 或部分训练状态 |
| BF16/FP16 | 28 GB | 1.0× | 常见推理基线 |
| FP8/INT8 | 14 GB | 0.5× | 容量与权重带宽减半，需硬件 kernel 支持 |
| 6 bit | 10.5 GB | 0.375× | 需 packing、scale 和混合精度处理 |
| 4 bit | 7 GB | 0.25× | 压缩最强，但 activation/outlier 和解包开销更敏感 |

这些是未计 scale、zero-point、embedding、norm 和对齐填充的理论值。FSVideo 同时包含 14B base DiT 与 14B refiner，磁盘/主存中 BF16 权重合计理论约 56 GB；若顺序执行，可以只把一个主干常驻 GPU，但切换模型会引入加载和 offload 带宽成本。论文明确指出两个 14B DiT 难以同时放入单张 80 GB H100，单卡测试依赖参数 offloading。

训练阶段的内存/显存需求远高于推理。采用 mixed precision Adam 时，参数、梯度、master weight 和一二阶优化器状态的朴素总量常达到约 16–20 bytes/parameter；14B 模型仅这些训练状态即约 224–280 GB，尚未计 activation。因而训练必须依赖 ZeRO/FSDP、tensor/sequence/context parallel、activation checkpointing 和数据管线；视频 activation 又随 N 增长，常成为实际峰值。

> **公式 6：推理峰值显存预算**　M_peak ≈ M_weight,resident + M_activation + M_attention + M_VAE + M_cache + M_comm + M_runtime

这个加法式比“参数量 × 2 bytes”更适合部署估算。特别是长视频中，M_activation 和 M_cache 会随 token/历史增长；多卡时 M_comm 还包括 all-to-all、all-gather 或 reduce-scatter buffer。工程上建议在测得的峰值之外预留 10%–20% fragmentation 和并发裕量。

对未采用 activation checkpointing 的训练，activation 可用下面的粗略形式理解：

> **公式 7：activation 的一阶估计**　M_act ≈ κ · B · L_saved · N · d · b_a/8

B 为 batch size，L_saved 为反向传播需保存的层数，b_a 为 activation bitwidth，κ 汇总 Q/K/V、FFN 中间量、残差及框架常数。FlashAttention 会消除显式 N² attention matrix，但不会消除所有随 N 增长的 activation。

### 2.4 带宽与通信

带宽需求分为四层：HBM、片上 SRAM/shared memory、GPU 互联、CPU/SSD 到 GPU。FlashAttention 的核心价值不是减少数学上的 exact attention FLOPs，而是分块在片上完成 softmax 更新、避免显式写出 N×N attention matrix，从而减少 HBM 读写。

若 14B BF16 权重无法常驻，每次 NFE 都从主存搬运完整 28 GB 权重，则 60 NFE 的理论传输量下界达到 1.68 TB/clip；实际框架会通过分层驻留、预取和重用降低或隐藏部分传输，但这个数量级说明 offload 很容易压倒计算。量化在这一场景中不仅减少容量，也按位宽近似减少传输 bytes。

> **公式 8：参数 offload 的传输量下界**　D_offload ≥ S · (1-f_resident) · M_weight

f_resident 表示能够常驻 GPU 的权重比例。若 14B BF16 模型完全不能常驻、S=60，则 D_offload≥60×28 GB=1.68 TB。若希望在 T 秒内完成，仅参数搬运所需的平均链路带宽下界为：

> **公式 9：offload 带宽下界**　BW_required ≥ D_offload / T

例如 T=60 s 时，1.68 TB/60 s≈28 GB/s，已经接近高端 PCIe 链路的有效上限，且尚未计算 activation、VAE 和协议开销。因此，存在大规模 offload 时，模型量化和常驻布局往往比继续减少少量 FLOPs 更优先。

判断 kernel 是算力受限还是带宽受限，可使用 roofline 关系：

> **公式 10：Roofline 性能上限**　Performance ≤ min(P_peak, BW_HBM · AI)

其中 P_peak 是设备峰值算力，BW_HBM 是 HBM 带宽，AI=FLOPs/bytes 是算术强度。量化减少 bytes，并可能提高低精度 P_peak；FlashAttention 主要减少 bytes、提高 AI；不规则稀疏虽减少 FLOPs，却可能同时降低有效带宽和 tensor-core 利用率，因此实际性能未必按 FLOPs 成比例提高。

多卡 sequence/context parallel 把 N 分片，降低单卡 activation，但 attention 需要交换 Q/K/V 或局部结果。分辨率和时长增长时，通信占比提高；跨节点链路远慢于节点内 NVLink，拓扑和 all-to-all 重叠成为关键。StreamFusion 针对这一问题采用 topology-aware sequence parallel、Torus Attention 和 one-sided communication，报告平均 1.35×、最高 1.77×优于既有方案。

> **公式 11：单层分布式执行时间**　T_layer ≈ max(T_compute, T_comm) + T_unhidden

当算通能够完全重叠时，层时间接近二者最大值；不能重叠的同步、尾部通信和负载不均计入 T_unhidden。增加 GPU 数只能降低 T_compute 的一部分，却可能提高 T_comm 和 T_unhidden，所以扩展效率应写为：

> **公式 12：并行效率**　E_G = T_1 / (G · T_G)

G 为 GPU 数。E_G=1 表示理想线性扩展；Video DiT 在长序列 all-to-all、跨节点和不规则稀疏下通常显著低于 1。

### 2.5 端到端部署瓶颈：一个可计算的时延分解

为了避免把局部 kernel speedup 等同于产品时延，可将一次视频生成写成：

> **公式 13：端到端时延**　T_E2E = T_text + S(T_DiT + T_CFG + T_comm + T_cache + T_offload) + T_VAE-enc + T_VAE-dec + T_post + T_queue

T_text 是文本编码；括号内是每个 NFE 的主干、CFG、通信、cache 和 offload；其余项为 VAE 编解码、后处理和服务排队。若离线 benchmark 不含 T_queue，应明确标注。

若优化模块原来占总时延比例为 q，局部加速 a 倍，则全链路最大加速遵循 Amdahl 定律：

> **公式 14：端到端加速上限**　Speedup_E2E = 1 / [(1-q) + q/a]

例如 attention 占总时延 60%，即使 attention kernel 无限快，端到端上限也只有 1/(1-0.6)=2.5×；若 attention 加速 2×，全链路仅为 1/[0.4+0.6/2]=1.43×。这正是 sparse attention 的理论 FLOP 降幅通常大于真实端到端收益的原因。

进一步地，部署瓶颈可写成最大项而非简单求和的近似：

> **公式 15：主导瓶颈判定**　T_service ≳ max(C_total/P_eff, D_HBM/BW_eff, D_link/BW_link, M_required/M_available 引发的惩罚)

前两项分别对应有效算力和 HBM 带宽，第三项对应 PCIe/NVLink/网络通信；若 M_required>M_available，系统会触发分片、重算或 offload，产生非连续的“容量惩罚”。因此显存 OOM 不是平滑变慢，而是可能让系统跨入完全不同的性能区间。

## 3. Video Diffusion 负载演进趋势

### 3.1 从短片生成走向长上下文与流式生成

传统离线生成一次处理完整 clip，峰值显存随总 token 增长。流式/causal 方案将视频切成 chunk，可降低首帧时延和单次工作集，却引入历史 KV/cache、误差滚动和 chunk 边界一致性问题。若不做窗口化或压缩，cache 随历史长度近似线性增长；因此实时并不等于低资源，它常把一次性 activation 压力转化为长期驻留 cache 和持续带宽压力。

2026 年 Causal-RoPE SP 在 8×A800 上通过 sequence-parallel causal RoPE、算通重叠、operator fusion 和预计算，对 5 秒 480p 报告 1.58×，并达到亚秒首帧与接近实时。这表明下一阶段的指标会从“单 clip 总时延”扩展到首帧、稳定帧率、cache 上界和长时漂移。

### 3.2 从单卡容量问题走向多卡通信问题

更大模型、更高分辨率和更长视频首先使单卡 OOM，继而迫使 sequence/context/tensor parallel。并行度增加后，计算量被分摊，但每层通信、同步和负载不均的相对占比上升。稀疏 attention 又可能造成不同 rank 的有效 block 数不均，出现“算法 FLOPs 降了、最慢 rank 决定时延”的现象。

因此未来系统的有效性能指标应至少包括：每卡 peak VRAM、节点内/跨节点 bytes、通信占比、overlap 比例、最慢 rank 负载和端到端扩展效率，而不能只给总 GPU 数和 speedup。

### 3.3 从主干受限走向 VAE 与后处理受限

当 NFE、attention 和精度被优化后，VAE decoder 的时延占比会上升。视频 VAE 在接近 RGB 输出端时需要处理高分辨率特征图，特征图占用很大；3D convolution 的时间窗口又使 activation 难以切分。FSVideo 的 FSAE-Lite 通过减少末端通道、采用 group-causal convolution 以及 slicing/tiling，将内存占用降低到原来的约 1/1.75–1/2。Flash-VAED 进一步报告约 6× decoder 加速，但端到端最高只改善 36%，说明 decoder 已成为显著但非全部的剩余瓶颈。

### 3.4 从 FLOPs 优化走向 bytes 和执行规则性优化

动态 top-k/top-p 稀疏的数学成本看似低，但若先计算接近完整 score 才知道 mask，收益会被预测开销抵消；若 mask 不规则，GPU tensor core 和内存合并访问利用率下降。RainFusion2.0 用 block mean 代表 token、3D window permutation 和 first-frame sink，让 mask 预测和 block 执行更规则，并验证在 NPU/ASIC 类硬件上的可迁移性。HASTE 则通过跨 timestep mask reuse 与按 head 分配稀疏预算继续降低路由成本。趋势不是更高 sparsity 数字本身，而是更低元数据、可复用 mask、tile 对齐、静态形状和跨硬件 kernel 支持。

## 4. Efficient Diffusion 方法总览：四类主范式与两条优化轴

综述将方法分为 step distillation、efficient attention、model compression、cache/trajectory optimization 四类。系统上可进一步压缩为两条正交轴：

- **减少网络调用次数 S（NFE 轴）**：蒸馏、solver、CFG distillation、trajectory redesign、并行采样。
- **降低每次调用成本 C_step（每步轴）**：token/VAE 压缩、稀疏或线性 attention、量化、剪枝、feature reuse、kernel 与并行优化。

端到端时间可粗略写成 T ≈ S×C_step + T_VAE + T_text + T_comm + T_IO + T_runtime。这个表达式揭示了为什么两轴技术有乘法潜力，也解释了为什么端到端收益总低于局部 kernel 的乘积。

### 4.1 Step distillation：减少 NFE

一致性蒸馏让不同噪声时间点映射到一致输出；分布蒸馏让 few-step student 匹配 teacher 的输出分布；对抗蒸馏用判别/偏好信号补偿少步生成的锐度和运动感。其直接效果是 S 从 40–100 NFE 降至 4–8，甚至 1。

| 资源维度 | 变化 | 解释 |
|---|---|---|
| 算力 | 近似随 NFE 成比例下降 | 主干前向次数减少；训练蒸馏本身成本高 |
| 峰值显存 | 通常小幅下降 | 单次前向工作集未必改变；teacher-student 训练会显著增加 |
| HBM 带宽 | 总 bytes/clip 近似随 NFE 下降 | 权重、activation 和 kernel 中间数据被读取更少次 |
| 持久化存储 | 可能增加 | 保留 teacher/student、EMA 或多阶段 checkpoint |
| 风险 | 少步误差和时间一致性退化 | 一步最敏感，4–8 步通常更稳健 |

FSVideo 的 refiner 先做 CFG distillation，再 progressive distillation 至 32 step，最后 SiDA 至 8 NFE，报告 refiner inference time 降低 87%。这也说明 CFG 的双前向可单独蒸馏掉，NFE 比“diffusion steps”更适合作为统一口径。

### 4.2 Efficient attention：减少 N² 计算与 HBM traffic

FlashAttention 保持 exact attention，但避免显式 attention matrix；static sparsity 用固定局部、条纹、径向或 block-diagonal mask；dynamic sparsity 根据当前内容路由；linear/hybrid attention 将复杂度近似降为线性，但更容易损伤长程关系。

对 KV-only 稀疏，keep ratio ρ 时 attention 主项由 O(NqNk d) 变为 O(ρNqNk d)。80% sparsity 意味着 ρ=0.2，attention 理论主项约为 dense 的 20%，即局部上限约 5×；但端到端不可能直接达到 5×，因为 FFN、投影、VAE、路由和运行时不随该项减少。

RainFusion2.0 的 80% sparsity 对不同视频/图像模型报告 1.5–1.8×端到端加速，HunyuanVideo1.5 480p/720p 的收益又分别只有约 1.16×/1.28×，显示 attention 在不同 workload 中占比差异显著。该结果是评估稀疏技术时最应保留的现实校正。

### 4.3 Model compression：量化、剪枝与 VAE 压缩

量化将 weight/activation 降至 FP8、INT8、6 bit 或 4 bit；剪枝移除 token、channel、block 或参数；VAE 压缩则直接减少 DiT 序列长度。三者虽同属 compression，但系统效应不同：

- weight-only quantization 主要减少模型体积和权重 bandwidth；activation quantization 还能降低中间数据与通信，但对异常值和 timestep 变化更敏感。
- channel/block pruning 直接减少 GEMM，但要形成结构化形状才能兑现 tensor-core 吞吐；非结构化零值可能只减少理论参数。
- token pruning 同时减少后续线性层和 attention；若 query token 也被裁剪，收益更高，但错误会直接影响相应 token 的特征更新，并可能映射为局部画面缺失或时序不一致。
- VAE compression 在主干前就降低 N，对所有 L 层和所有 S 次前向生效，是乘法层级最高的优化；代价是重建细节和 latent 可生成性变差，往往需要更大 latent channels、专门正则、refiner 或 decoder 增强。

6Bit-Diffusion 使用 NVFP4/INT8 动态混合精度，并依据 block 在不同 timestep 的稳定性分配精度；再用 Temporal Delta Cache 跳过稳定 block，报告 1.92×端到端加速，并将内存占用降至基线的约 1/3.32。它代表“量化 + cache + timestep-aware control”的组合趋势，而不是固定 W4A4 一刀切。

### 4.4 Cache 与 trajectory optimization：以容量和读写换算力

feature cache 复用相邻 timestep 的 block 输出或残差；KV cache 复用 causal history；trajectory 方法修改 noise/state 或并行推进部分路径。缓存的关键公式不是只看命中率，而是：

收益 ≈ 被跳过计算时间 − cache 查找/融合/刷新时间 − 新增内存访问时间。

若缓存内容位于 HBM 且算术强度低，cache 可能把 compute-bound block 变成 bandwidth-bound copy；若 cache 太大导致模型权重无法常驻，甚至可能触发 offload，反而恶化端到端性能。长视频还必须控制 drift，所以需要按 timestep、layer、motion 或 error 自适应刷新，而非固定复用间隔。

## 5. Step Distillation：从“重复执行几十次”转向少步生成

Step distillation 的研究对象不是单次 DiT 前向有多贵，而是同一大模型为何必须沿噪声轨迹重复执行几十次。设一次主干前向成本为 C_step、网络调用次数为 S，则主干成本为 S·C_step。只要单次工作集不变，把 50 NFE 降至 4 NFE，理论主干计算、权重读取和多数 activation 流量都可降到 8%；这是蒸馏成为最高杠杆的根本原因。

但视频不能简单照搬图像少步化。图像蒸馏的单帧误差主要表现为局部纹理差异；视频中的误差会沿时间和自回归 chunk 累积，表现为运动冻结、速度不连续、身份漂移、镜头边界跳变。因此，视频蒸馏的核心问题是：**如何在大幅减少 NFE 的同时保持完整去噪分布、运动轨迹和长时 rollout 的稳定性。**

### 5.1 Consistency Distillation：稳定优先的少步路线

**动机。** 多步扩散轨迹上的不同噪声状态最终应映射到同一干净样本。与其逐步复现 teacher 的所有中间状态，不如训练 student 对相邻或跨步状态保持输出一致，从而跳过中间求解过程。

**具体做法。** 设在线 student 为 f_θ，EMA teacher 为 f_θ̄；从较晚噪声状态 z_t 通过 teacher 得到较早状态 z_s，再约束两者经一致性映射后的结果接近：

> **公式 17：一致性蒸馏目标**　L_cons = E[d(f_θ(z_t,t), f_θ̄(z_s,s))]，s<t

视频方法通常再加入三类补偿：第一，使用帧差、光流或可学习 motion extractor 对齐运动；第二，用 reward/preference model 约束文本一致性和视觉质量；第三，对高运动片段加权，避免 student 通过生成近静态视频取得较小平均误差。VideoLCM、Motion Consistency Model 和 DCM 的差别主要在 motion representation 的设计。

**计算影响。** 推理阶段从几十次主干前向降至 4–8 次，主干 FLOPs 近似按 NFE 比例下降；单次前向的 attention 和 FFN 结构不变。训练阶段则需要 student、EMA teacher，有时还需要 reward model，因此训练计算明显上升。

**带宽影响。** 推理时参数和中间状态的总读取次数随 NFE 下降；若权重常驻 HBM，总 HBM bytes/clip 近似下降。若存在 CPU-GPU offload，减少 NFE 还会同步降低跨 PCIe/NVLink 的重复权重搬运。训练时 teacher/student 双模型读取使 HBM 和跨卡通信压力增大。

**存储与显存影响。** 推理峰值显存通常不会随 NFE 线性下降，因为一次前向仍需相同权重和 activation；但总能耗和临时 buffer 生命周期缩短。训练需保存 student、EMA teacher、optimizer states 和可能的 reward model，持久化 checkpoint 和训练显存明显增加。

**意义与边界。** 一致性蒸馏具有稳定、易迁移、适合 controllable generation 的优势，但严格的一致性目标相对保守，在一步或两步极限下通常不如 distribution matching。它更适合质量优先的 4–8 步服务，而不是无条件追求一步生成。

### 5.2 Distribution Distillation：匹配输出分布而非逐步轨迹

**动机。** 一致性约束仍要求 student 遵循较强的局部轨迹关系，限制了极低 NFE。Distribution Matching Distillation（DMD）直接让 few-step generator 的输出分布接近 teacher 分布，允许 student 采用不同于 teacher 的短路径。

**具体做法。** 从 student 输出构造加噪样本 ẑ_t，并分别用真实/teacher score 与 student-induced fake score 评估；优化 student 使两者接近，同时训练辅助 fake score/critic。其抽象目标可写为：

> **公式 18：分布蒸馏目标**　L_DMD = D(P_student(ẑ_t) || P_teacher(ẑ_t))

非流式方法对完整 clip 做一至四步生成；流式方法则把视频切成 causal chunks。CausVid 用双向 teacher 蒸馏 causal student，但训练时依赖真实历史，存在 exposure bias；Self-Forcing 在训练中让 student 使用自己生成的历史和 KV cache，缩小训练—推理差距。后续方法进一步沿四条路线发展：causal attention/window、带噪自回归 rollout、运动/奖励感知 DMD target，以及从双向预训练到 causal 部署的多阶段 curriculum。

**计算影响。** 这是目前最有能力进入 1–4 NFE 的路线。对 50 NFE teacher，4 NFE 的主干理论减少 92%，一步则减少 98%。但 streaming 模型每个 chunk 还需执行若干 NFE，并产生历史 attention；总计算取决于 chunk 数、window 和 cache 策略。

**带宽影响。** 少步化降低每 clip 的权重和 activation 总流量；流式部署却会增加 KV cache 的持续读写。历史越长，当前 query 读取的 K/V 越多，带宽可能从“重复 NFE”转移到“持续历史访问”。

**存储与显存影响。** 离线 few-step 推理的峰值显存变化有限。流式模型需要长期保存 KV，若每层、每头均保存全部历史，cache 容量近似随 chunk 数线性增长；window、attention sink、compressed memory 和 eviction 因此成为必要组件。训练通常还需要 teacher、fake score network 和 rollout state，成本高于普通 fine-tuning。

**意义与边界。** Distribution distillation 是实时视频和 world model 的核心基础，但稳定性工程与损失本身同等重要。错误不会只在当前 chunk 出现，而会进入下一 chunk 的条件分布。因此，应以 20 秒、1 分钟甚至更长 rollout 评估，而不能只看 5 秒 VBench。

### 5.3 Adversarial Distillation：少步模型的感知质量补偿器

**动机。** 极少步 student 容易产生过平滑纹理、运动幅度不足和视觉锐度下降。对抗损失能直接惩罚生成分布与真实视频在感知层面的差异。

**具体做法。** 训练 discriminator D_ψ 区分真实视频 z_0 与 student 输出 G_θ(ε)，student 则最大化被判为真实的概率：

> **公式 19：对抗补偿**　L_G,adv = -E[log D_ψ(G_θ(ε))]

ADD、LADD、DMD2 等通常把 adversarial loss 叠加在 consistency 或 DMD 之上；独立对抗蒸馏因 GAN 式不稳定而较少作为第一阶段。Seaweed-APT 等更可信的路径是先得到稳定少步模型，再以对抗目标做后期锐化。

**计算、带宽和存储影响。** 推理时 discriminator 被丢弃，因此几乎不增加部署 FLOPs、带宽或显存；训练时需要 discriminator 前后向、额外视频特征和真实样本，训练算力、显存与数据 IO 增加。它本身不减少 NFE，而是帮助更激进的 NFE 压缩保持质量。

**意义与边界。** 对抗蒸馏应被理解为 quality recovery，而不是独立的效率机制。对高动态视频，它能提升清晰度和运动感；但也可能放大伪影、造成训练震荡，必须与分布或一致性目标共同约束。

### 5.4 2026 年新趋势：转移匹配与逐步动态结构

Transition Matching Distillation（TMD，2026）不再要求一个完整 14B 主干在每个内部更新中重复运行，而是把模型拆成“多数早期层构成的语义主干”和“少数末层构成的轻量 flow head”。每个外层 transition 只运行一次主干，flow head 在共享语义表征上执行多个较便宜的内部 flow updates。其意义是把少步模型内部仍存在的重复计算进一步拆解：

> **公式 20：TMD 近似成本**　C_TMD ≈ S_outer·C_backbone + S_outer·K_inner·C_head，且 C_head≪C_backbone

这减少主干计算和权重读取，但需缓存语义表征，因而会增加局部 activation 驻留和读取。它适合主干极大、head 相对轻量的 DiT。

Dynamic-in-Few-Step（2026 年 7 月）进一步指出，不同少步 timestep 对网络深度和 block 的需求不同。该方法在 4-step distillation 中联合学习 step-specific structured sparsity，形成 Mixture-of-Models；在 Wan-14B 上额外移除 24% per-step FLOPs，带来 1.2× wall-clock 增益，并相对 50-step teacher 达到 30×。它说明下一阶段不是“先蒸馏、再独立剪枝”，而是联合优化 S 与 C_step。代价是保存多个 step-specific 子结构、增加调度元数据，并要求专用 inference engine 才能兑现结构稀疏。

## 6. Efficient Attention：减少长视频中的 N² 计算与内存流量

Efficient attention 的出发点是：Video DiT 的 N 随帧数和像素分辨率增长；full self-attention 需要计算 N_q×N_k 的相关性。其核心成本为 2N_qN_kd，若显式存储 score/probability，还产生 O(N_qN_k) 中间矩阵。该类方法分为 IO-aware exact attention、静态稀疏、动态稀疏，以及 linear/hybrid attention。

### 6.1 FlashAttention：不改模型数学，只改数据移动

**动机。** 标准 attention 即使 FLOPs 可接受，也会因反复读写 HBM、显式 materialize N×N 矩阵而受带宽和显存限制。

**具体做法。** 把 Q/K/V 切成可放入 SRAM/shared memory 的 tiles，在片上分块计算 score、online softmax 和 value accumulation，仅写回最终输出。数学结果仍为 exact attention。

**计算影响。** 理论 attention FLOPs 基本不变，但减少 kernel 间操作和无效内存等待，实际吞吐提高。FlashAttention-3 还利用异步执行和低精度路径提高 H100 利用率。

**带宽影响。** 避免把完整 score/probability 写入 HBM，将 IO complexity 从近似 O(N²) 数据写回降到以 Q/K/V/O 分块读写为主；它主要优化 bytes，不是减少注意力连接数。

**存储影响。** 不再保存完整 N² attention matrix，attention workspace 显著下降，使更长序列成为可能；模型权重和其他 activation 不变。

**意义。** FlashAttention 是后续 sparse/hybrid attention 的系统基线。任何新方法若只与朴素 attention 比较而不与 FlashAttention/SageAttention 比较，都可能夸大收益。

### 6.2 Static Sparse Attention：以规则性换适应性

**动机。** 视频注意力存在强时空局部性、对角带、首帧/关键帧全局连接和周期性结构。若这些模式在不同输入间稳定，可预先设定 mask，跳过大部分不重要的 QK 连接。

**具体做法。** Efficient-vDiT、GRAT、PAROAttention、Radial Attention、Sparse-vDiT 和 W.A.L.T 分别采用 tile locality、grouped block、重排后的对角带、径向模式、条纹模式或 block diagonal。Sparse-vDiT 还离线搜索每层每头最适合的稀疏 pattern，并把相同策略的 heads 融合为统一 kernel。

设 keep ratio 为 ρ，则：

> **公式 21：静态 KV 稀疏 attention**　C_att,sparse ≈ ρ·2N_qN_kd，M_score≈ρ·N_qN_k·b_a/8

**计算影响。** attention 主项近似按 ρ 下降；QKV projection 和 FFN 不变。Sparse-vDiT 在 CogVideoX1.5、HunyuanVideo、Wan2.1 上分别报告理论 FLOP 降幅 2.09×、2.38×、1.67×，实际推理为 1.76×、1.85×、1.58×。

**带宽影响。** 规则 block mask 可减少 K/V tile 和 score 的 HBM 流量，并维持合并访问；若 pattern 太碎，则索引、metadata 和不连续 gather 会降低有效带宽。

**存储影响。** attention workspace 随有效 block 数下降；固定 mask 元数据很小。模型权重不变，通常无需额外 cache。

**意义与边界。** 静态稀疏易部署、kernel 规则、跨硬件可复现，但无法针对当前视频恢复被 mask 丢弃的重要长程依赖。短 clip 上表现稳定的局部先验，在快速镜头切换、远距离物体重现和超训练时长视频上可能失效。

### 6.3 Dynamic Sparse Attention：让稀疏模式随内容变化

**动机。** 不同视频、层、head 和 timestep 的重要 token 不同。动态路由可以保留真正相关的 K/V，在相同稀疏率下通常比固定 mask 更好。

**具体做法。** 路由分三类：计算式路由用压缩 Q/K 或低比特 score 预测重要 block；先验式路由利用 pose、optical flow 或 object mask；统计式路由复用历史 log-sum-exp、attention indices 或 timestep 稳定性。RainFusion2.0 用 block mean、3D window permutation 和 first-frame sink；HASTE 用 Temporal Mask Reuse 避免每个 timestep 重算 mask，再按 head 的误差敏感性分配全局稀疏预算。

动态稀疏的真实成本应写成：

> **公式 22：动态稀疏净成本**　C_dynamic = C_route + ρC_att,dense + C_gather/scatter + C_metadata

只有 C_route 与不规则执行开销足够小，ρ 带来的理论收益才能兑现。

**计算影响。** KV-only sparsity 保留所有 query，attention 主项按 ρ 降低，质量相对安全；QKV/token sparsity 同时减少 query，可让后续投影与 FFN 也下降，但路由错误影响更大。RainFusion2.0 在 80% sparsity 下端到端 1.5–1.8×；HASTE 在 720p 达到最高 1.93×。

**带宽影响。** 被裁剪 K/V 减少 HBM 读取，但动态索引会造成不连续访问；mask prediction 还会读 Q/K 代理。mask reuse 可以同时减少计算和元数据流量。KV-only 稀疏若用于 streaming，也可降低历史 cache 的有效读取量。

**存储影响。** score/workspace 下降；需保存 mask、indices、per-head budget 或历史统计。通常 metadata 远小于 K/V，但在小 block、长序列和多层多头下不可忽略。

**意义与边界。** 动态稀疏的学术上限高于静态稀疏，工程下限却更低：没有 tile-friendly kernel 时，较少 FLOPs 可能变成更慢的执行。评价必须同时报告 attention kernel、路由、端到端时延和 peak VRAM。

### 6.4 Linear 与 Hybrid Attention：把复杂度从 N² 改为 N

**动机。** 当视频长度远超训练 horizon 时，即使 80% sparsity，ρN² 仍然是二次复杂度。Linear attention 用核映射或状态递推改变乘法次序，使 K/V 先聚合为固定大小状态。

> **公式 23：线性注意力**　Attention(Q,K,V)≈φ(Q)[φ(K)ᵀV]，复杂度 O(Nd²) 而非 O(N²d)

**具体做法。** 纯 linear attention 用可分离 kernel/state-space 结构；hybrid attention 保留少量 full/sparse heads 或局部窗口，并以 linear branch 承担低频、远程残差。训练型方法通常需要从头训练或对 pretrained DiT 做 attention surgery 和恢复训练。

**计算影响。** 长序列下理论扩展性最好；但 d² state update、normalization 和额外 branch 仍有成本。N 不够大时，优化良好的 FlashAttention 可能更快。

**带宽与存储影响。** 不存 N² score，workspace 从二次降为线性或固定状态；历史可压成 recurrent state，降低 KV cache 增长。代价是更多小矩阵/状态读写，可能偏带宽受限。

**意义与边界。** 线性化会压缩 pairwise 交互，容易过度平滑长程运动、削弱身份召回。综述判断更可靠的路线是 sparse 为主、linear 作为被裁剪 token 的低成本补偿分支，而不是全面替换 attention。

### 6.5 2026 年新进展：长视频、流式 KV 与硬件协同

LVSA（2026）将结构化窗口与 rotating global anchors 结合，消除固定网格在超训练时长下的偏置；配合 FlashInfer kernel，在 Wan2.1/HunyuanVideo 上报告最高约 2.98–3.33× compute reduction，并使原本单卡 OOM 的更长 HunyuanVideo 生成可运行。它的意义不仅是速度，还在于把稀疏 attention 作为“扩展上下文容量”的方法。

Sparse Forcing（2026）把 persistent salient blocks 和 local block sparsity 原生训练进 autoregressive diffusion，并提供 Persistent Block-Sparse Attention kernel。它在 5 秒生成上降低 peak KV-cache 42%，20 秒和 1 分钟时加速提高到 1.22×和 1.27×。这说明流式 attention 的关键指标应随 horizon 变化：短视频收益可能一般，长视频因 KV 受控而逐渐放大。

Light Forcing（2026）用 Chunk-Aware Growth 分配不同历史 chunk 的稀疏率，并做 frame/block 两级选择；单独 sparse attention 端到端约 1.2–1.3×，与 FP8 和 LightVAE 组合后在 RTX 5090 达到 2.3×及 19.7 FPS。这个结果直接支持“少步、稀疏、低精度、轻量 VAE 必须协同设计”的趋势。

## 7. Model Compression：降低参数、激活与 latent 序列负载

Model compression 包含四种资源作用完全不同的技术：QAT/PTQ 主要降低 bitwidth；VAE compression 降低进入 DiT 的 token 数；token pruning 动态缩短序列；model pruning 移除网络结构。不能把它们笼统理解为“模型变小”。

### 7.1 Quantization-Aware Training（QAT）

**动机。** 低比特可近似按位宽降低权重/activation 容量和搬运量，并利用 FP8/INT8/FP4 tensor core 提高吞吐；但 Video Diffusion 的 activation 含离群值，且不同 timestep 的分布显著变化，直接量化会产生闪烁和轨迹偏移。

**具体做法。** QAT 在训练前向中插入 fake quantizer Q(X;s)，让模型在量化噪声下更新权重，并通过 teacher alignment、token saliency、timestep-aware scale 和 mixed precision 恢复质量。QuantSparse、S2Q-VDiT、Q-VDiT 的共同思想是把量化视为误差补偿问题，而不是简单舍入。

> **公式 24：QAT 对齐目标**　L_QAT = L_task(G_θ^quant) + λ·d(G_θ^quant, G_teacher)

**计算影响。** 部署硬件有原生低比特 GEMM 时，线性层/FFN/投影吞吐提高；否则 dequantize、packing 和 scale 会抵消收益。QAT 训练成本高于普通 fine-tuning。

**带宽影响。** W8 相对 BF16 将权重 bytes 减半，W4 理论降至 1/4；activation quantization 还减少层间 HBM 与跨卡通信。scale、zero-point 和混合精度 fallback 会使实际降幅略小。

**存储影响。** checkpoint 和 resident weights 显著缩小，可能使模型从 offload 变为完全常驻；这会产生非线性的端到端收益。训练需维护高精度 master weights 和量化参数，训练存储不一定下降。

**意义与边界。** 当目标为 W4A4/NVFP4 等激进精度，QAT 通常比 PTQ 更可靠；代价是需要数据、训练资源和目标硬件确定性。

### 7.2 Post-Training Quantization（PTQ）

**动机。** 大多数视频基础模型训练成本过高，部署方无法重新训练；PTQ 希望只用少量 calibration 视频和少量参数优化完成量化。

**具体做法。** 主要包括三个方向：transform-then-quantize，通过 rotation、smoothing 或 log-domain 变换压平离群值；timestep-aware quantization，为不同 t 选择 scale/bitwidth；layer/token sensitivity allocation，把较高精度留给 motion-sensitive 层、首尾 timestep 和异常 token。

> **公式 25：timestep-aware PTQ**　s_t=Π_s(t)，b_t=Π_b(t)，min Σ_t ||G_quant(z_t,t;s_t,b_t)-G_fp(z_t,t)||²

**资源影响。** 推理阶段与 QAT 类似：权重/activation bytes 降低，原生 kernel 下算力提高；校准阶段成本远低于重新训练。PTQ 的模型存储可直接下降，但为多个 timestep 保存多组 scale/bit allocation 会增加少量 metadata。

**意义与边界。** PTQ 是当前最可部署的 backbone compression。其上限取决于 calibration 是否覆盖真实分辨率、运动类型和 timestep；只用静态图片或短 clip 校准，长视频可能出现累积量化误差。

### 7.3 2026 年动态混合精度：6Bit-Diffusion

6Bit-Diffusion 根据 block 输入输出 residual 与量化敏感性的相关性，在推理时动态选择 NVFP4 或 INT8；对时间上稳定的 block 再用 Temporal Delta Cache 跳过计算，报告 1.92×端到端加速、内存占用降至约 1/3.32。

其资源意义是：NVFP4 主要减少容量、HBM bytes 和 GEMM 时间，INT8 为敏感层提供误差保险，TDC 再减少实际执行 block 数。代价是 predictor、双 kernel path、cache 和调度分支。如果硬件不能高效混合 NVFP4/INT8，算法收益会下降。

### 7.4 VAE Compression：在进入 DiT 之前减少 N

**动机。** 量化只改变每个元素的 bitwidth，不能改变 token 数。VAE compression 同时减少 encoder/decoder 中间空间和 DiT 的序列长度，收益会跨所有 L 层和 S 次 NFE 传播。

**具体做法。** 第一类保持 image-VAE 兼容并增强时间建模，如 CV-VAE、IV-VAE；第二类通过 omni-dimensional、wavelet 或轻量结构实现更强时空压缩，如 OD-VAE、WF-VAE、LeanVAE；第三类将静态结构与动态运动分解，如 VidTwin、Hi-VAE、DC-VideoGen。FSVideo 的 FSAE 则采用 64×64×4 下采样、128 channels、专门 latent regularization 和 refiner。

设 token 比例 α=N_new/N_old，则线性层近似降为 α，attention 主项降为 α²：

> **公式 26：VAE 压缩对 DiT 的放大收益**　C_new/C_old ≈ q_lin·α + q_att·α² + q_fixed

**计算影响。** 所有 DiT block、所有 NFE 均受益；但更强 VAE、decoder 和 refiner 会增加固定成本。FSVideo 的 42.3×双 H100 实测说明 token reduction 可远大于普通 sparse kernel 的收益。

**带宽影响。** 更短序列减少 Q/K/V、FFN activation、跨卡 all-to-all 和 cache bytes；高通道 latent 会部分抵消输入输出张量的减少。decoder 接近 RGB 端仍需处理大量像素级特征。

**存储影响。** DiT activation/workspace 大幅下降；VAE/refiner 权重可能增加。两阶段模型会增加持久化 checkpoint，并产生模型切换或双模型常驻问题。

**意义与边界。** 这是架构级最高杠杆，但压缩丢失发生在整条生成链上游；身份、运动或高频细节一旦未编码，后续 DiT 很难凭空恢复。必须同时报告 VAE reconstruction 与最终 generation quality。

### 7.5 Token Pruning：选择、合并或延迟计算 token

**动机。** 同一 latent 网格中，背景、低运动区域和相邻帧冗余程度不同。动态减少 token 可同时降低 attention 和后续 FFN。

**具体做法。** Astraea/FastVID 用 saliency、motion 或 density 评分；FullDiT2/VGDFR 动态选择保留率；AsymRnR 保留近期和高运动 token；frame context packing 更倾向 merge 而不是硬删除。

**资源影响。** 若 N 降至 αN，后续 attention 理论降至 α²、线性层降至 α；但 scoring、sort、gather/scatter 和 token restoration 都增加计算与不规则 HBM 访问。保存 merge map、indices 和恢复信息会增加 metadata。

**意义与边界。** 它比 KV-only sparse attention 更激进，但当前说服力较弱，因为被删除 query 不再更新，错误会直接映射为画面缺失或运动不连续。只有结构化 token packing 和融合 kernel 才可能获得稳定端到端收益。

### 7.6 Model Pruning：移除 block、head、channel 或 FFN

**动机。** 大型 Video DiT 在层深和 FFN 宽度上存在冗余；少量层可能对特定 timestep 或运动模式贡献较低。

**具体做法。** 以 teacher 为参照评估 block sensitivity，保留深层 motion-sensitive blocks，迭代删除较不重要 block，再用 online distillation 恢复；也可联合 head/FFN/channel 做多粒度结构化剪枝。

**计算影响。** 删除完整 block 可近似按保留率 r_l 降低 attention 和 FFN；结构化 channel/head 剪枝可缩小 GEMM。非结构化零权重若没有 sparse GEMM kernel，通常只减少参数统计而不减少 wall time。

**带宽与存储影响。** 删除 block 同时减少权重读取、activation 和 checkpoint 大小；模型变浅还减少通信同步次数。多套 timestep-specific 子模型则可能增加总体磁盘存储，但每次只激活一套。

**意义与边界。** 视频模型的深层往往承担运动与全局一致性，简单按权重幅值剪枝风险高。当前可信路线是 sensitivity-guided structured pruning + distillation repair + 专用 engine。

## 8. Cache、Trajectory 与 Parallelism：消除跨 timestep 和跨 chunk 的重复工作

### 8.1 Feature Cache：复用相邻去噪步的中间特征

**动机。** 相邻 timestep 的 latent 变化有限，许多 block 输出高度相似；每一步都完整重算浪费计算。

**具体做法。** 对第 l 层缓存先前特征 C_(t-Δ)^l，根据 latent/feature 相似度、预测误差或固定 schedule 决定 reuse/refresh：

> **公式 27：特征复用门控**　F̃_t^l = g_t^l C_(t-Δ)^l + (1-g_t^l)F_t^l

方法可分为 full reuse、partial refresh 和 error-compensated reuse。FasterCache、TeaCache、DiCache、BWCache、Sortblock 等工作的核心差别在于评分信号、刷新粒度和误差补偿。

**计算影响。** cache hit 时跳过部分 block 或近似其 residual；收益与 hit rate、被跳过 block 占比相关。相似度预测和 correction 会消耗部分 FLOPs。

**带宽影响。** 少读被跳过 block 的权重，但需要读写 cached feature。若 feature 很大且 block 算术强度高，cache 有利；若被替代 block 很轻，单纯搬运 feature 可能更慢。

**存储影响。** 显存增加约为缓存层数×token×hidden×bytes；多 timestep、多分辨率 cache 可能显著。cache 若挤出 resident weights 并触发 offload，会产生反效果。

**意义与边界。** Feature cache 是训练自由且易插入的加速方式，但最大风险是误差跨 timestep 积累。可靠方案必须动态刷新，并对高运动层/时段保守。

### 8.2 KV Cache：用历史存储换取流式重算减少

**动机。** 自回归/causal 视频生成中，新 chunk 需要访问历史。若每次重算历史 K/V，计算巨大；缓存 K/V 可避免重复 projection。

> **公式 28：KV cache 容量**　M_KV ≈ 2·L·N_history·d_kv·b_a/8

若按 batch、head 显式展开，还需乘 B 和 KV head 数对应维度。容量随历史 token 线性增长；当前 query 对历史的 attention 计算也随 N_history 增长。

**具体做法。** Sliding window 和 eviction 固定预算；compressed memory 把历史汇总为 memory tokens；selective KV 按相关性保留关键帧、对象或 persistent blocks；attention sink 始终保存首帧/关键条件。

**计算影响。** 避免历史 K/V projection 重算，但当前 Q 对历史 K/V 的 score 仍需计算；若做 selective KV，attention 计算随有效 history 减少。

**带宽影响。** KV cache 是典型的持续 HBM 读取负载。长视频中，计算可能从 projection-bound 转为 cache-bandwidth-bound；量化 KV、分层存储和局部窗口可以降低 bytes。

**存储影响。** 峰值显存随 cache budget 增加。Sparse Forcing 报告降低 42% peak KV-cache footprint，说明 native sparse memory 能同时改善容量和长时速度。

**意义与边界。** KV cache 是流式生成的必要设施，而不是无代价加速。错误 eviction 可能在很久之后表现为身份漂移，因此评估必须覆盖延迟失效。

### 8.3 Noise/State Modification：局部改变状态而不重写全轨迹

**动机。** 部分长视频一致性可通过起始噪声、局部 latent 或跨帧对应关系实现，无需训练新模型或改变全部采样步骤。

**具体做法。** FreeNoise 重排初始噪声并做窗口融合；TokenFlow 根据帧间对应替换 diffusion features；Latent-Shift 用无参数通道 shift 引入时间耦合；Free4D 在特定步骤融合或替换 reference state。

**资源影响。** 新增操作通常为 O(Nd) 的 shift、copy、warp 或 fuse，算力低但偏内存带宽受限；需保存 reference/window state，增加少量到中等 activation 存储。它不直接减少主干 NFE，主要通过允许更稳定的窗口生成或避免额外修复步骤间接提升效率。

**意义与边界。** 训练自由、轻量、适合 editing/long-video extension；但先验与场景不匹配时容易产生拼接或运动伪影，通用加速上限有限。

### 8.4 Trajectory Modification：重新分配整条去噪路径的预算

**动机。** 不同 timestep 对内容、结构和细节的重要性不同，均匀步长或单一路径可能浪费计算。

**具体做法。** 一类优化 timestep placement、adaptive sub-step、leap 或局部 transition approximation；另一类将生成改为 coarse-to-fine、content-motion decomposition 或 high-low-high resolution 多阶段路径。FSVideo 的低分辨率 base + 8-NFE refiner 属于多阶段预算重分配。

**计算影响。** 将昂贵高分辨率计算集中在少数后期步骤，其余在低分辨率或轻量 head 完成；理论收益取决于每阶段 N、S 和模型大小之和，而不是简单步数。

**带宽与存储影响。** 低分辨率阶段减少 activation 和通信；多阶段需传递 latent、保存多个 checkpoint，并可能增加模型切换 IO。若 base/refiner 无法同时驻留，offload 会侵蚀收益。

**意义与边界。** 轨迹重设计比局部状态编辑上限更高，但全局校准错误会影响整段视频。应报告每阶段成本和端到端质量，不应只报告主干某一阶段。

### 8.5 Parallel Computation：用更多设备换取时延或吞吐

**动机。** 长序列和大模型单卡无法容纳；Video DiT 又包含 sequence、patch、CFG、timestep/chunk 等多种可并行维度。

**具体做法。** xDiT 统一 sequence、patch 和 CFG parallel；PipeDiT 将 denoising/decoding pipeline 化；Block Cascading 并行 temporal blocks；db-SP 对 sparse attention 做负载均衡；StreamFusion 用 topology-aware sequence parallel、Torus Attention 和 one-sided communication 重叠跨机 all-to-all。

**计算影响。** 单卡 FLOPs 下降，但全局 FLOPs 通常不变，甚至因 padding/重复边界增加。收益由并行效率 E_G 决定。

**带宽影响。** all-to-all、all-gather、reduce-scatter 和 pipeline activation transfer 增加。节点内 NVLink 与跨节点网络差异巨大，拓扑感知和 overlap 比理论并行度更重要。

**存储影响。** 参数/activation/cache 可分片，单卡显存下降；通信 buffer 和 replicated states 增加。更高并行度也会增加总 GPU-seconds，未必降低成本。

**意义与边界。** 并行化解决“容量与时延”，不等于算法更高效。报告必须同时给 wall time、GPU 数、GPU-seconds、网络规模和扩展效率。

### 8.6 VAE Decoder 与服务系统：加速后的新瓶颈

主干经过少步、稀疏和量化后，decoder、文本编码、模型加载和排队占比上升。Flash-VAED 通过 channel pruning 和 causal 3D convolution 的 stage-wise dominant operator optimization，在 Wan/LTX decoder 上约 6×，端到端最高 36%。这符合 Amdahl 定律：decoder 只有在主干被充分加速后才值得优先优化。

服务层还包括 shape-aware batching、token bucket scheduling、checkpoint prefetch、CUDA graph/compile、异步 VAE 和流水线。它们不改变模型 FLOPs，却直接影响 GPU 利用率、tail latency 和实际吞吐。

## 9. 四类方法的统一资源分析与选择原则

### 9.1 三种资源并非同步下降

- 少步蒸馏主要降低总计算和总流量，不显著降低单次峰值显存。
- 量化主要降低容量和 bytes，只有硬件原生 kernel 才稳定降低 wall time。
- 稀疏 attention 降低 attention compute/workspace，但可能增加路由与不规则访存。
- Feature cache 降低计算，却增加显存和 cache 带宽。
- KV cache 降低历史重算，却增加长期驻留存储和持续读取。
- VAE compression 同时降低 token、activation、attention 和通信，但可能增加 decoder/refiner。
- 多卡并行降低单卡存储压力，却提高互联带宽需求和总设备成本。

### 9.2 方法选择应由瓶颈类型决定

| 当前主瓶颈 | 优先方法 | 不应首先采用的方案 | 原因 |
|---|---|---|---|
| NFE/总前向次数 | 4–8 步 distillation、CFG distillation | 单独做小比例 pruning | 减少 S 的杠杆更直接 |
| 长序列 attention | FlashAttention、块稀疏、LVSA/HASTE | 非结构化权重剪枝 | 目标应是 N² 与 HBM traffic |
| 权重无法常驻 | FP8/INT8/PTQ、分片、compressed backbone | 增加 feature cache | cache 可能进一步挤占 HBM |
| Activation OOM | VAE/token compression、FlashAttention、sequence parallel | 仅 weight-only quantization | 权重变小不一定解决 activation |
| PCIe offload | 量化、常驻布局、减少 NFE | 复杂动态路由 | 应先减少搬运 bytes 与次数 |
| 流式 KV 增长 | window/eviction、compressed/selective KV、Sparse Forcing | 保存全历史 | 容量与读取成本随时长增长 |
| VAE decoder 占比高 | Flash-VAED、tiling/slicing、轻量 decoder | 继续只优化 DiT attention | 瓶颈已经迁移 |
| 多节点通信高 | topology-aware SP、overlap、负载均衡 | 更细粒度不规则 sparsity | 可能加剧 all-to-all 与 rank imbalance |

### 9.3 评价 efficient diffusion 的最低证据要求

一项可信的 efficient video diffusion 工作至少应报告：相同输出帧数/像素分辨率、真实 NFE、精度、GPU 型号与数量、是否包含 VAE/text/offload、peak VRAM、端到端 latency、局部 kernel latency、理论 FLOPs、质量与长时稳定性。训练型方法还应报告蒸馏/恢复训练成本和额外 checkpoint。

最常见的误导包括：用理论 sparsity 代替 wall time；用 diffusion steps 代替 NFE；用双卡 latency 与单卡 baseline 比较；只测短视频而声称长视频稳定；只报告权重容量而忽略 activation/cache；只报告主干而排除 VAE 和 offload。

## 10. 代表性方法逐项解析：从论文算法到硬件负载

本章不再按“大类”概括，而是选取能够代表技术演化路径的方法逐一拆解。每项分析均区分算法所减少的理论运算、实际设备上减少的数据搬运，以及为实现加速新引入的状态、元数据和训练成本。

### 10.1 VideoLCM 与 Motion Consistency Model：为什么视频一致性蒸馏必须显式处理运动

**问题定义。** 普通 consistency distillation 约束同一去噪轨迹上 z_t 与 z_s 的预测结果一致。对静态图像，这足以保证内容和纹理大体保持；对视频，如果损失主要由大量低运动像素主导，student 可以通过减小运动幅度获得较低平均误差，形成“看起来清晰但不动”的退化解。

**核心观察。** 视频质量包含两个不完全一致的目标：每帧外观质量与跨帧运动一致性。若只在 latent 或像素空间做逐元素距离，快速运动区域在总损失中占比可能很小，但感知重要性很高。因此 VideoLCM、MCM、DCM 等方法在一致性目标之外引入 motion-aware representation。VideoLCM 更接近直接 latent 对齐；MCM 使用可学习运动提取器；DCM 使用时间 latent difference，使损失直接感知相邻帧变化。

**算法流程。** 首先从 teacher 的多步轨迹采样两个噪声时刻 t>s；其次由 EMA teacher 将 z_t 推进到 z_s；然后 student 分别预测两状态的 clean latent；最后同时最小化外观距离和运动距离：

> **公式 29：运动一致性蒸馏**　L = d(ẑ_0^t,ẑ_0^s) + λ_m d(M(ẑ_0^t),M(ẑ_0^s))

M 可以是帧差、光流特征或可学习网络。λ_m 过小会回到静态偏置，过大则可能牺牲单帧细节。

**计算影响。** 推理阶段 M 被移除，仍只执行 few-step student，因此加速主要来自 NFE 减少。训练阶段需要 teacher 前向、student 前向和 motion extractor；若 M 是光流网络，训练 FLOPs 和显存会明显增加。若 M 只是 temporal difference，额外计算近似 O(Nd)，相对主干较小。

**带宽与存储影响。** 推理的权重和 activation 总流量随 NFE 减少。训练需要同时保留两个 timestep 的输出及运动特征，activation 存储和 HBM 读写高于纯一致性蒸馏。EMA teacher 还带来约一份模型权重的持久化存储。

**意义与局限。** 这类方法证明“少步视频”不能只优化去噪误差，还必须把运动作为独立监督对象。其局限是 motion extractor 的偏置：光流偏好局部对应，可能难以描述镜头运动、遮挡和物体生成消失。

### 10.2 DMD、DMD2 与视频 Distribution Distillation：为什么它能进入 1–4 步

**问题定义。** 轨迹一致性仍要求 student 与 teacher 的局部动力学相似。当步数压到 1–4 时，student 不可能逐段复现 teacher 的几十步路径，因此必须直接优化最终分布。

**核心做法。** DMD 将当前 student 输出重新加噪得到 ẑ_t，并比较 teacher score 与 fake score。Teacher score 指向真实/teacher 分布的高密度区域，fake score 描述 student 当前分布；二者差形成更新 generator 的方向。DMD2 通过改进 fake score 训练、回归项和对抗机制提高稳定性。

> **公式 30：DMD 梯度的抽象形式**　∇_θL_DMD ∝ E_t[(s_fake(ẑ_t,t)-s_real(ẑ_t,t))·∂G_θ/∂θ]

它不是要求每个 z_t 对应固定 teacher target，而是让整个 student-induced distribution 向 teacher distribution 移动，因此允许短得多的生成路径。

**计算影响。** 部署推理可降到 1–4 NFE。训练却非常昂贵：需要 generator、teacher score model、fake score model，有时还需 discriminator；每次更新可能包含多次大模型前向。训练成本可能远高于普通 fine-tuning，但这是一次性成本，适合高调用量服务摊销。

**带宽与存储影响。** 部署时总 HBM 流量按 NFE 显著减少，峰值显存仍由一次 generator 前向决定。训练同时驻留多份 14B 模型通常不可行，需要 FSDP/ZeRO、CPU offload 或分阶段更新；模型与 optimizer checkpoint 可达到数百 GB 乃至 TB 级。

**失效条件。** Fake score 跟不上 generator 时，梯度方向失真；视频中的少量高运动模式易发生 mode dropping；一步模型还可能把 teacher 的多解分布压成保守运动。因而实际系统多采用 4 步而不是一步，并叠加 reward/adversarial correction。

### 10.3 CausVid 与 Self-Forcing：实时视频为什么不仅是“少步”问题

**CausVid 的动机。** 双向 Video DiT 生成整个 clip 后才能输出，首帧时延高且不能无限延长。CausVid 将双向 teacher 蒸馏为 causal student，使第 i 个 chunk 只依赖历史 chunks。

**CausVid 的训练—推理差距。** 训练时 student 条件是 ground-truth history z_data^{<i}，推理时条件却是自身生成历史 z_gen^{<i}。小误差进入下一 chunk 后持续放大，即 exposure bias。短 clip 指标可能正常，长视频会逐渐漂移。

**Self-Forcing 的改进。** 在训练阶段显式展开 autoregressive rollout，让 student 使用自己生成的历史 KV cache。第 i 个 chunk 的生成可写为：

> **公式 31：自回归少步 rollout**　z_i^K →G(·|C_<i) z_i^{K-1} → … → z_i^0，C_i=Update(C_<i,Proj(z_i^0))

这样训练分布更接近推理，但 rollout 长度增加会成倍提高训练计算与 activation/KV 存储。

**资源分析。** 与离线 full-clip 方案相比，causal chunk 降低首帧时延和单次 activation 工作集；但总时长不变时，主干总 FLOPs未必下降。KV cache 使历史 projection 不必重算，却增加 M_KV≈2LN_historyd_kv·b/8 的长期显存和 HBM 读取。窗口太小会丢长期身份，窗口太大则失去流式容量优势。

**意义。** Self-Forcing 把实时生成的核心从“每秒能做多少 FLOPs”转为“模型能否在自身错误分布上稳定运行”。因此实时系统必须共同优化 NFE、chunk size、KV budget 和 rollout training。

### 10.4 TMD：用大主干一次表征加多个轻量内部更新

**动机。** 即使已经只有 4 个 outer steps，每一步仍完整运行几十层 14B DiT。相邻内部 flow updates 可能共享高层语义，不必重复执行全部主干。

**模型分解。** TMD 将原模型前面大多数层作为 semantic backbone，把最后少数层改造成 conditional flow head。每个 outer transition 运行一次 backbone 得到 h_t；flow head 在 h_t 条件下执行 K 次内部状态更新：

> **公式 32：TMD 内部转移**　h_t=B(z_t,t)，u_0=z_t，u_(k+1)=u_k+Δ_kH(u_k,h_t,t)

**计算分析。** 原始 K 次完整更新为 K(C_B+C_H)，TMD 为 C_B+KC_H；理论加速为 K(C_B+C_H)/(C_B+KC_H)。当 C_B≫C_H、K 较大时收益显著。

**带宽分析。** Backbone 权重由每个内部更新读取 K 次变为一次；但 h_t 必须在 K 次 head update 中保持并反复读取。若 h_t 很大，TMD 可能从计算受限转为 feature-bandwidth 受限，需要片上重用或 fusion。

**存储分析。** 需要保存 h_t、flow head 状态和额外 head 权重；峰值 activation 可能略增，但总体权重读取减少。训练还需把 pretrained backbone 适配为条件 flow map，并进行 DMD rollout。

**意义。** TMD 说明“减少 NFE”之外还可以分解一次 NFE 内部的重复语义计算，是蒸馏与 feature reuse 之间的桥梁。

### 10.5 Sparse-vDiT：为何离线搜索的静态稀疏能比动态路由更容易落地

**核心观察。** 不同 Video DiT 的注意力图反复出现 diagonal、multi-diagonal 和 vertical-stripe 模式，而且 pattern 与 layer depth、head index 的相关性高于与具体输入的相关性。这意味着可以离线确定每层每头的 pattern，不必在线预测 mask。

**算法流程。** 首先在 calibration videos 上统计每个 head 的 attention pattern；然后以质量误差和硬件 latency 为目标搜索 dense、diagonal、multi-diagonal、vertical stripe 或 skip；最后将相同 pattern 的 heads 分组并调用专门 sparse kernel。

> **公式 33：硬件感知 pattern 选择**　p*_(l,h)=argmin_p [Error_(l,h)(p)+λ·Latency_kernel(p,N)]

**计算影响。** 被 skip 的 head 完全省去 attention；其余 head 只计算 pattern 对应 blocks。论文报告理论 FLOP reduction 1.67–2.38×，实际 1.58–1.85×，折损来自 QKV/FFN 不变、kernel 和非 attention 占比。

**带宽影响。** 静态 block layout 允许预先排布 contiguous tiles，减少 K/V 与 score traffic；无在线 routing，metadata 读取很小。head grouping 还能减少 kernel launches。

**存储影响。** attention workspace 减少；每层每头只需保存少量 pattern ID 和参数。无需保存历史统计或动态 indices。

**局限。** Calibration distribution 与部署视频差异大时，固定 pattern 不能恢复被丢弃交互；长视频超出训练 horizon 后，静态局部结构可能导致循环和冻结。

### 10.6 RainFusion2.0：动态稀疏如何降低 mask prediction 开销

**问题。** 在线动态稀疏若先近似计算完整 QK score 再选 top-p，路由本身接近 dense attention，理论收益消失；逐 token top-k 又产生极不规则访问。

**具体做法。** RainFusion2.0 将 token 分为 3D spatiotemporal blocks，以 block mean q̂_i、k̂_j 估计块间相关性；用 3D window permutation 把时空相似 token 排到相邻位置；再强制所有 query 连接第一帧 key，形成 first-frame sink。

> **公式 34：块级代理评分**　ŝ_ij=(mean(Q_i)·mean(K_j))/√d，Mask_ij=I(ŝ_ij≥τ)

若每块含 B 个 token，代理 score 矩阵规模由 N² 降为 (N/B)²；随后只对保留 block 执行精确 attention。

**计算影响。** 80% sparsity 意味 attention 主项理论保留约 20%，但 permutation、block mean、mask 和非 attention 模块存在，端到端为 1.5–1.8×。

**带宽影响。** Block mean 只需顺序归约，规则 block mask 适合 GPU/NPU；permutation 若物理重排 tensor 会产生额外读写，因此实现应尽量使用 layout/index 复用。first-frame sink 增加少量固定 K/V 读取，换取身份稳定。

**存储影响。** score workspace 与 K/V 有效读取下降；需保存 permutation map 和 block mask，远小于 dense N² matrix。该方法尤其适合 N 很大、attention 占比较高的 720p workload。

### 10.7 HASTE：按 head 分配稀疏预算并跨 timestep 复用 mask

**核心观察一。** 相邻 denoising timestep 的 Q/K 漂移有限，反复预测 mask 是冗余计算。HASTE 监控 query-key drift，仅在变化超过阈值时更新 mask。

**核心观察二。** 不同 heads 对稀疏误差的敏感性差异大。统一 top-p 会在不敏感 head 上浪费预算，在敏感 head 上损伤质量。HASTE 通过 calibration 测量每个 head 的 output error，在给定全局计算预算下分配 ρ_h：

> **公式 35：Head-wise 稀疏预算**　min_{ρ_h} Σ_h E_h(ρ_h)，s.t. Σ_h Cost_h(ρ_h)≤C_budget

**计算影响。** Mask reuse 直接减少 C_route，head-wise allocation 在相同平均 sparsity 下减少质量损失，从而允许更激进预算；720p 最高 1.93×。

**带宽影响。** 复用 mask 减少代理 Q/K 和 mask metadata 的重复读取；不同 head 的不等 block 数可能造成 warp/rank load imbalance，需要 bucket 和融合。

**存储影响。** 保存上一 timestep mask、drift statistics 和 per-head budget；相对 K/V 较小，但比静态稀疏复杂。

**意义。** HASTE 把动态稀疏从“每次重新找 top-k”推进到跨 timestep 控制问题，更符合 diffusion 的迭代冗余。

### 10.8 LVSA：长视频稀疏为什么需要旋转全局锚点

**问题。** 固定 local window 在训练长度内有效，但生成时长扩大数倍后，不同窗口缺少跨区信息，模型可能产生周期循环或静止重复。固定 global tokens 又会形成长期偏置。

**做法。** LVSA 保留结构化局部窗口，并让 global anchors 随层或时间旋转，使远距离块在不同计算阶段获得连接机会。它仍是 training-free block sparse，配合 FlashInfer kernel。

**计算与存储。** 在 horizon 扩大时，dense attention 的 N² 增长最明显，而 LVSA 每个 query 只连接固定局部块与少量 anchors，复杂度接近 O(Nw+Na)。论文在多模型上报告约 2.98–3.33× compute reduction，并使原本单卡 OOM 的更长生成可运行。Workspace 和有效 K/V 读取同步下降，anchor metadata 很小。

**质量意义。** Rotating anchors 不是单纯加速技巧，而是用有限连接覆盖全局时间轴。其失效风险是 anchor rotation 周期与视频运动周期耦合，或关键对象恰好在未连接窗口中长期缺失。

### 10.9 Sparse Forcing：把 KV 记忆压缩和局部稀疏联合训练

**核心观察。** 自回归视频的历史 attention 并非均匀分布：少数 salient blocks 在多个 chunks 中持续被关注，构成隐式长期记忆；当前滑动窗口内部则呈局部 block sparsity。

**具体做法。** 模型学习哪些历史 blocks 应成为 persistent memory，哪些可以淘汰；局部窗口只动态选择邻域 blocks。PBSA kernel 将 sparse attention、persistent memory update 和 cache management 融合，避免独立 gather/scatter。

> **公式 36：受预算约束的持久化记忆**　C_i = TopK_persistent(C_(i-1)∪K_i, score)，|C_i|≤K_budget

**计算影响。** 当前 query 不再访问全部历史 KV；时长越长，相对 dense history 的节省越大。论文 5秒为1.11–1.17×，20秒1.22×，1分钟1.27×，体现 horizon-dependent gain。

**带宽影响。** 历史 KV 读取量受固定 budget 控制，PBSA fusion 减少 indices 和 cache update 的额外 traffic。若 score 需要扫描全部历史，收益会消失，因此 persistent update 必须增量执行。

**存储影响。** Peak KV-cache 下降42%；另需 persistent score、age 和 block metadata。固定预算使显存从随历史线性增长变为近似有界。

**意义。** 这是稀疏 attention 与 cache management 真正融合的代表，比在 dense KV cache 上临时加 mask 更适合一分钟级流式生成。

### 10.10 Q-DiT、ViDiT-Q 与 timestep-aware PTQ：为什么视频量化不能只校准一次

**问题。** Diffusion 在高噪声 timestep、结构形成阶段和低噪声细化阶段的 activation 范围、outlier 和敏感性不同。单一 scale s 用于所有 t，会在某些阶段浪费动态范围，在另一些阶段饱和。

**做法。** Calibration 收集多个 timestep、层和 token 的统计；对权重使用 per-channel/group scale，对 activation 使用 timestep-aware smoothing/rotation；敏感层或首尾阶段保留更高 bitwidth。

> **公式 37：量化误差预算**　min_{b_l(t),s_l(t)} Σ_{t,l} w_{t,l}||X_l(t)-Dequant(Q(X_l(t);s_l(t),b_l(t)))||²

**计算影响。** W8A8/FP8 在原生 tensor core 上可提高 GEMM 吞吐；W4A8 主要降低权重带宽。若每 timestep 动态切换 kernel，调度和编译变体增加，应把连续 timesteps bucket 到有限配置。

**带宽影响。** W8 相对 BF16 权重流量约减半；A8 也减少 activation 和 all-to-all bytes。Rotation/smoothing 通常可离线折叠进权重，否则会增加在线 GEMM/读写。

**存储影响。** 模型体积近似按 bitwidth 下降；per-group scales 和多 timestep profiles 增加少量 metadata。量化使 14B 模型从 28 GB 降到约14 GB（8 bit），可能直接消除 offload。

**局限。** Calibration prompt、运动分布和分辨率不充分时，量化误差会在长 rollout 中积累；逐帧指标可能看不出闪烁，需要 temporal metric 和人工评估。

### 10.11 6Bit-Diffusion：动态 NVFP4/INT8 与 Temporal Delta Cache 的联合收益

**核心观察。** 某 block 的输入—输出 residual 幅度与其内部线性层的量化敏感性相关；相邻 timesteps 中 block residual 又具有时间稳定性。

**流程。** 轻量 predictor 根据 residual 选择 NVFP4 或 INT8；稳定 block 使用 NVFP4，敏感 block 保留 INT8。若 residual 与历史足够接近，TDC 直接复用 cached delta，跳过 block：

> **公式 38：Delta cache**　y_t=x_t+Δ_t，若 d(Δ_t,Δ_(t-1))<τ，则 y_t≈x_t+C_Δ

**计算影响。** 量化加速实际执行的 GEMM，TDC 减少执行 block 数，两者近似相乘；论文端到端1.92×。

**带宽影响。** NVFP4/INT8 降低权重与 activation bytes；TDC 少读 block 权重，但多读 cached delta。对算术强度高的 FFN/attention projection，通常划算；对轻层可能变成 cache copy 限制。

**存储影响。** 低比特权重显著缩小；delta cache 与 predictor state 增加显存。报告内存占用降至基线约1/3.32，说明权重/activation 压缩超过 cache 增量。

### 10.12 FSVideo/FSAE：为什么压缩 latent 网格比压缩权重更能改变数量级

**问题。** 14B 模型很大，但 Video DiT 的主要重复成本还来自每层、每 NFE 都处理巨大 token 序列。Weight quantization 只改变每个参数的 bytes，不改变 N。

**做法。** FSAE 将像素视频映射为时间4×、空间64×64下采样的128-channel latent；通过专门 regularization 降低 latent intrinsic dimension；FSAE-Lite 减少靠近 RGB 端的大 feature channels，并用 group-causal convolution；base DiT 在低分辨率 latent 生成，再由 CNN upscaler 和8-NFE refiner补细节。

**计算链。** 若网格 token 比例 α≈1/60，则 DiT 线性项近似降60倍、attention项在同 patch 假设下理论可降约3600倍；实际模型宽度、patch、refiner和非attention成本使最终远低于这个上限。2×H100 实测42.3×仍说明其数量级优势。

**带宽链。** Q/K/V、FFN activation和sequence-parallel communication随N下降；但128 latent channels增加输入投影数据，refiner引入第二阶段读写。两14B权重若不能同时常驻，会出现模型切换带宽。

**存储链。** 主干activation/workspace大降；BF16两14B checkpoint理论56GB。FSAE decoder的大像素特征仍是峰值来源，因此需要tiling/slicing。

**质量代价。** 高压缩VAE的重建PSNR/FVD弱于8×8×4 VAE；refiner不是可选装饰，而是对上游信息损失的补偿。评估必须同时看VAE reconstruction、generation quality和总GPU-seconds。

### 10.13 TeaCache、FasterCache 与 DiCache：Feature Cache 的三种控制思想

**共同动机。** 相邻 timesteps 中 block features 高度相关，尤其在轨迹平稳阶段。不同 cache 方法的本质差别不是“是否缓存”，而是如何判断当前误差仍在可接受范围内。

**Schedule/prior-based。** TeaCache 类方法利用 timestep embedding 或预设 schedule 判断哪些步骤可复用，开销小、规则性好，但对prompt和运动自适应弱。

**Similarity/residual-based。** FasterCache 等依据相邻 feature/residual 差异触发刷新，可针对当前视频；需要计算距离并维护历史状态。

**Model-decided/forecast-based。** DiCache 让模型或轻量 predictor 判断 cache，或预测未来 feature 再以误差控制刷新；适应性强但引入 predictor FLOPs。

> **公式 39：Cache 是否有净收益**　ΔT = T_skipped-block - T_score - T_cache-read/write - T_correction

**计算影响。** 命中率高且跳过的是大 FFN/attention block 时收益最大。若只跳过轻层，score与copy可能超过重算。

**带宽影响。** 少读被跳过block权重，但缓存完整B×N×d feature需要大量HBM traffic。可缓存residual、低秩表示或量化feature降低bytes。

**存储影响。** 保存一层BF16 feature需要2BNd bytes；缓存k层/多timestep近似乘k。高分辨率视频中cache容量必须纳入与权重争抢HBM的预算。

**质量边界。** Cache误差会被后续denoising继续使用，不是独立噪声。应在高运动、镜头切换和少步模型上单独调阈值，因为少步模型每一步的重要性更高。

### 10.14 StreamFusion 与 xDiT：为什么并行方案的中心是通信而不是GPU数量

**xDiT的贡献。** DiT推理可沿sequence、spatial patch、CFG branch和pipeline等维度并行。xDiT提供统一引擎，使不同模型能组合parallel strategies。

**通信问题。** Sequence parallel需要交换K/V或attention结果；CFG parallel在条件/无条件分支结束时合并；patch parallel需要边界或全局attention通信。GPU数G增加后，每卡计算约降1/G，但通信不会同比下降。

**StreamFusion的做法。** 根据节点内与节点间带宽差异做topology-aware partition；Torus Attention把跨机all-to-all与本地attention计算重叠；one-sided communication减少sender-receiver同步。

> **公式 40：多卡层时延**　T_layer≈max(C_layer/(G·P_eff), D_comm/BW_link)+T_sync+T_imbalance

**计算影响。** 算法总FLOPs基本不变，wall time下降；padding和重复边界可能增加总FLOPs。

**带宽影响。** 通信是新增主成本。低比特activation、sparse balanced partition和overlap能降低/隐藏D_comm；不规则dynamic sparsity可能让不同rank工作量失衡。

**存储影响。** 参数和activation分片降低单卡峰值，但通信buffer、replicated norms/text states增加。评估必须给GPU-seconds；8卡1秒不一定比1卡6秒更省资源。

### 10.15 Flash-VAED：为什么主干加速后必须重新测端到端占比

**动机。** 原始系统中DiT可能占90%时延，VAE decoder只占10%；若DiT被加速10倍，新的占比变为DiT约47%、decoder约53%，decoder成为主瓶颈。

**做法。** Flash-VAED用independence-aware channel pruning移除decoder冗余通道，并针对不同stage中的dominant operators优化causal 3D convolution；通过三阶段distillation保持原latent distribution兼容。

**计算影响。** Decoder局部约6×，但端到端最高36%。代入Amdahl公式可反推decoder在优化前已占相当比例，且其他模块仍不变。

**带宽与存储影响。** 靠近RGB端的feature map尺寸大，channel pruning同时减少convolution FLOPs、activation HBM traffic和峰值显存；轻量decoder权重也下降。Decoder distillation增加训练成本但不增加部署模型。

**意义。** Efficient diffusion必须反复profile。任何局部优化成功后，原来的性能模型都会失效；下一轮应针对新占比，而不是继续优化已经不主导的attention。

## 11. FSVideo 案例：深 latent 压缩如何重塑负载

### 11.1 设计逻辑

FSVideo 的系统逻辑是先用 FSAE 在空间维度进行 64×64 下采样、在时间维度进行 4×下采样，并以 128 个 latent channels 保持表达能力；随后在高度压缩的潜在空间中训练 14B base DiT，再通过 CNN latent upscaler 和 8-NFE 的 14B refiner 恢复高分辨率细节。Layer Memory 让当前层可从此前层的 K/V 表征中动态聚合，以增强网络深度方向的信息复用。

这种设计不是把模型变小，而是让大模型处理数量更少、单个 token 信息密度更高的表示。它把负载从“每一步处理巨量 token”转移到高通道 latent 的输入投影、更复杂的 VAE decoder、第二个 refiner checkpoint 和两阶段调度。

### 11.2 量化负载变化

以 121 帧、720×1280 像素的视频为例，先估算 VAE 输出的 latent 网格单元数；这里的“网格单元”不是输出像素，也不一定等于最终 DiT token：

| 方案 | 时空下采样率 | latent 网格单元数（近似） | 相对 8×8×4 |
|---|---|---:|---:|
| Wan 类 VAE | 8×8×4 | 31×90×160 = 446,400 | 1.0 |
| FSAE | 64×64×4 | 31×12×20 = 7,440 | 1/60 |

严格使用向上取整后，结果还取决于 padding 和 causal VAE 的首帧处理；若忽略边界并按连续比例计算，64×64 相比 8×8 的 latent 网格单元数理论上减少 64 倍。FSAE 将 latent channels 提高到 128，因此张量元素数量并不会随网格缩小而减少 64 倍。论文定义的总体压缩比为“输入视频张量元素数 ÷ latent 张量元素数”，FSAE 为 384:1，而论文列出的典型 8×8×4 VAE 为 48:1。对 DiT 而言，attention 的 N² 项由 patchify 后的 token 数决定；latent channel 数则主要影响输入/输出投影和 latent 表达能力。

### 11.3 实测与边界

FSVideo 在 BF16、FlashAttention 3、5 秒 720×1280 24 fps 条件下报告：

| 硬件/设置 | Wan2.1-I2V-14B | FSVideo | 结果 |
|---|---:|---:|---:|
| 1×H100 80 GB | OOM | 76.6 s（参数 offload） | FSVideo 可运行但受 IO 影响 |
| 2×H100 | 822.1 s | 19.4 s | 42.3× |
| 假设无显存约束的单卡估计 | 1607.5 s | 27.4 s | 58.7×（估计值） |

注意：FSVideo 使用 60 base NFE + 8 refiner NFE，Wan 为 60 NFE，NFE 接近但网络和 latent 空间不同，所以该结果证明的是“每 NFE 成本”显著下降，而不是单纯蒸馏收益。58.7×是论文估计，不应与实测同等看待。

FSVideo 也揭示了三项代价：一是高压缩 VAE 重建质量更难，FSAE-Standard 在 Inter-4K 的 PSNR 28.96、FVD 256.62，弱于低压缩 Hunyuan/Wan VAE；二是需要 refiner 补细节；三是两个 14B checkpoint 增加持久化存储和模型切换压力。因此它更像负载重分配与全栈优化，而非无代价压缩。

## 12. RainFusion2.0 案例：硬件友好的动态块稀疏

RainFusion2.0 解决动态稀疏的两个常见问题：mask 预测本身太贵，以及 GPU 之外的硬件难以执行不规则索引。其三项设计分别对应系统约束：

- block-wise mean 代表 token：以较低成本估计 block 重要性，降低路由算力和元数据。
- 3D spatiotemporal-aware permutation：把相似 token 聚到相邻 block，提高块内同质性，同时形成规则 tile。
- first-frame sink：强制保留与首帧 token 的连接，保护 I2V 的身份、结构和全局条件。

在 80% sparsity 下 1.5–1.8×的端到端收益说明，块稀疏能同时减少 score/value 计算和相关 HBM traffic，但不会等比例减少 Q 投影、FFN、VAE 与调度。论文还指出可与量化等正交方法组合；工程上应先验证 kernel 是否对目标 GPU/NPU 的 block size、layout 和精度原生支持，否则理论 sparsity 不等于 wall-clock speedup。

## 13. 技术对算力、存储与带宽的影响矩阵

| 技术 | 算力 | 峰值显存/存储 | HBM/IO/网络带宽 | 主要反作用 |
|---|---|---|---|---|
| Solver/少步蒸馏 | NFE 近线性下降 | 推理峰值变化有限；训练/多 checkpoint 增加 | 总读取次数下降 | 少步轨迹误差、训练成本 |
| CFG distillation | 去掉双前向，常接近 2×主干收益上限 | 条件状态略简化 | 权重与 activation 总读写下降 | guidance 灵活性降低 |
| VAE/token 深压缩 | 线性层随 N 降、attention 可随 N² 降 | activation 大降；VAE/refiner 权重可增加 | HBM 与跨卡 tensor 大降 | 重建损失、decoder/refiner 变重 |
| FlashAttention | FLOPs 基本不变或略降 | 不存 N² matrix，workspace 大降 | HBM traffic 大降 | 仍是 dense exact compute |
| 结构化稀疏 attention | attention 主项按 keep ratio 降 | score/workspace 降；mask 元数据增加 | KV/score traffic 降 | 路由、负载不均、质量风险 |
| Linear/hybrid attention | 理论由 N² 向 N 降 | 中间状态下降 | traffic 下降 | 长程运动与细节可能退化 |
| FP8/INT8/低比特 | tensor core 支持时吞吐提高 | 权重/activation 按位宽下降 | 权重、activation、通信 bytes 下降 | scale/outlier、解包与校准 |
| 结构化剪枝 | GEMM/层数直接下降 | 权重和 activation 下降 | 权重读取下降 | 恢复训练、形状利用率 |
| Feature cache | 跳过部分 block/NFE | cache 增加 | 计算读写减少但 cache traffic 增加 | drift、命中判断开销 |
| KV cache/流式窗口 | 历史重算下降 | 随窗口/历史增长 | 持续读写 cache | 长时容量和带宽上界 |
| Sequence/context parallel | 单卡计算和 activation 分摊 | 单卡下降、全局不一定下降 | all-to-all 等通信增加 | 拓扑、同步、扩展效率 |
| VAE decoder pruning/distill | decoder 算力下降 | decoder activation/权重下降 | RGB 邻近大 feature traffic 降 | 主干加速后才显著 |

## 14. 组合加速：如何避免“数字相乘、效果相消”

### 14.1 理想乘法模型

若基线成本为 S×C(N,b)，采用 NFE 比例 α_s、token 比例 α_n、精度 bytes 比例 α_b、稀疏 keep ratio ρ，则理想主干成本可写成：

C_new / C_old ≈ α_s × [线性项 α_n + attention 项 ρ α_n²] × 精度/硬件效率修正。

> **公式 16：组合加速的归一化模型**　R_compute ≈ α_s · [q_lin α_n + q_att ρ α_n² + q_other]

其中 α_s=S_new/S_old，α_n=N_new/N_old，ρ 是 sparse attention keep ratio；q_lin、q_att、q_other 分别是基线中线性层、attention 和不受这些优化影响部分的占比，三者之和为 1。理论主干加速约为 1/R_compute，但还需加入 VAE、通信、offload 和路由开销才能得到端到端结果。

举例：若基线 q_lin=0.35、q_att=0.55、q_other=0.10，采用 8/40 步蒸馏（α_s=0.2）、token 减半（α_n=0.5）和 50% attention keep ratio（ρ=0.5），则 R_compute≈0.2×[0.35×0.5+0.55×0.5×0.5²+0.10]≈0.069，主干理论约 14.5×。但若主干原本只占端到端 80%，最终上限约为 1/[0.2+0.8/14.5]≈3.92×。这个例子直观展示了“主干乘法加速”与“端到端瓶颈迁移”的差异。

这说明 token 压缩与 sparse attention 在 attention 项上会重复作用；当 token 已压得很低，attention 占比可能下降，继续提高 sparsity 的边际收益变小。相反，NFE 压缩对所有每步模块都生效，通常与 token 压缩更接近正交。

### 14.2 误差与系统开销并非乘法独立

少步蒸馏、低比特、token 丢弃、cache 复用都引入近似误差。单独调优时每项都“质量可接受”，组合后可能在相邻 timestep 累积并表现为闪烁、身份漂移、运动冻结或纹理崩坏。系统开销也会相互影响：稀疏路由可能使用 FP16 score，破坏全链路 FP8；cache 占用 HBM 后可能挤出常驻权重；sequence parallel 上的不规则稀疏会造成 rank imbalance。

因此组合顺序应采用逐层 profile 与质量回归：先建立 exact/BF16 基线；加入 token 或 NFE 主杠杆；再量化；再引入结构化稀疏；最后增加 cache 和并行通信优化。每一步重新测量 kernel breakdown、peak VRAM、HBM bytes、通信占比和长视频质量，不应直接套用各论文 speedup 的乘积。

## 15. 部署场景建议

### 15.1 单卡 24–48 GB

目标首先是“能常驻、少 offload”。优先使用 1–7B 或量化 14B、顺序加载子模型、VAE tiling/slicing、4–8 步蒸馏和 compressed latent。若仍依赖 PCIe offload，继续做 sparse attention 的收益可能被权重搬运掩盖。应记录每 clip host-device bytes 和模型切换次数。

### 15.2 单卡 80 GB 或双卡 NVLink

可让 14B BF16/FP8 主干常驻，并用 context/sequence parallel 承载高分辨率。优先使用 FlashAttention 3、FP8、规则 block sparsity 和 CFG distillation。双 14B 两阶段模型要评估并发驻留与顺序切换；若双卡分别驻留 base/refiner，需权衡 pipeline 空闲与 latent 传输。

### 15.3 多节点高吞吐服务

服务端的目标不只是单请求 latency，还包括 batching、SLO 和 GPU-hour/video。Video diffusion 的请求形状差异大，动态分辨率/时长会造成 padding 浪费。应按 token bucket 调度，使用 topology-aware sequence parallel，并将跨节点 all-to-all 与 attention/FFN 重叠。StreamFusion 的结果说明，网络拓扑感知本身可提供 1.35–1.77×的系统收益。

### 15.4 实时或长视频

使用 causal/chunk-wise 生成、few-step distillation、滑动窗口 KV、首帧/关键帧 sink、周期性 full refresh 和分层 cache。核心验收指标应为首帧 <1 s、持续 fps、固定显存上界、分钟级身份与运动稳定性，而不是只测 5 秒 clip。

## 16. 建议的基准测试与容量规划方法

建议每个候选模型固定 480p、720p 两档，5 s、10 s、30 s 三档，分别测试 BF16、FP8/INT8，以及 dense/sparse。每组记录：

1. 输入与 latent 的 F/H/W、patch、visual token N；
2. diffusion steps、CFG 实现和真实 NFE；
3. 权重格式、常驻比例、CPU-GPU transfer bytes；
4. peak allocated/reserved VRAM、cache 和通信 buffer；
5. DiT、VAE encode/decode、text encoder、通信、offload、调度的分段时间；
6. HBM throughput、tensor-core utilization、kernel launch 数、all-to-all 时间；
7. 首帧、总时延、吞吐、实时因子、能耗；
8. VBench/FVD/LPIPS/PSNR 与长时人工评测。

容量规划不要直接从参数量推算。最低应使用：

VRAM_peak ≈ resident weights + max(active block workspace + activations + attention buffers + VAE features + cache + communication buffers) + 10–20% runtime margin。

对服务容量，则以 GPU-seconds/video 为核心：GPU_seconds = GPU_count × wall_time。FSVideo 的双卡 19.4 s 等于约 38.8 GPU-s/video；与单卡方案比较时应同时看 latency 与 GPU-seconds，不能仅看 wall-clock speedup。

## 17. 未来 12–24 个月的技术趋势判断

1. **高压缩、可生成的 tokenizer/VAE 将继续前移为核心架构。** 目标从重建指标转向“在少 token 下易于 diffusion 建模”，并与 refiner、decoder co-design。
2. **4–8 步将成为高质量服务的主流折中，一步模型服务特定实时场景。** 研究重点转向稳定分布蒸馏、causal rollout 和质量恢复。
3. **稀疏 attention 将从算法 mask 竞争转向 kernel、layout 和负载均衡竞争。** mask reuse、head-wise budget、block/tile 对齐和跨 NPU/GPU 可移植性更重要。
4. **timestep-aware mixed precision 会替代全模型固定 bitwidth。** diffusion 不同 timestep、layer、token 的敏感性不同，动态 FP4/FP8/INT8 组合更可能保持质量。
5. **cache 会从“固定间隔复用”转向误差控制的分层缓存。** block residual、KV、feature 和 trajectory 将统一纳入容量预算与刷新策略。
6. **瓶颈继续向 VAE decoder、通信和调度迁移。** 主干优化越成功，端到端系统越需要 Flash-VAED 类 decoder 优化和 StreamFusion 类通信优化。
7. **MoE 可能扩大模型容量但不会自动降低系统成本。** active FLOPs 可下降，但专家权重存储、all-to-all、负载不均、prefetch 和 expert cache 会成为新瓶颈。

## 18. 结论

Video diffusion 的负载本质是一个三维扩展问题：token 决定单步序列规模，NFE 决定重复深度，memory movement 决定硬件能否兑现理论 FLOPs。高分辨率与长视频使 full attention、activation 和通信快速增长；大参数模型则使权重常驻与 offload 成为硬约束。

Efficient diffusion 已经形成明确的全栈路线：蒸馏减少 NFE，VAE/token 压缩降低所有层的输入规模，Flash/稀疏/混合 attention 降低 N² 和 HBM traffic，量化/剪枝降低参数与 activation bytes，cache/trajectory 减少重复工作，分布式和 decoder 优化处理瓶颈迁移。真正可部署的方案不是选择其中一个最高 speedup，而是以 profile 为依据组合两条正交轴，并对显存、带宽、通信和质量做共同约束。

就当前证据而言，最具确定性的方向是：**深 latent/token 压缩 + 4–8 步蒸馏 + 硬件原生 FP8/INT8 + 规则块稀疏 + 受控 cache**。它们分别作用于 N、S、bytes 和重复计算，具有较强的互补性。其工程成败取决于三点：近似误差是否在长时域内受控，稀疏/低比特是否有真实 kernel 支持，以及加速后是否及时处理 VAE 与通信的新瓶颈。

## 参考资料

### 本地核心材料

1. Chen et al. RainFusion2.0: Temporal-Spatial Awareness and Hardware-Efficient Block-wise Sparse Attention. arXiv:2512.24086v2, 2025. https://arxiv.org/abs/2512.24086
2. FSVideo Team et al. FSVideo: Fast Speed Video Diffusion Model in a Highly-Compressed Latent Space. arXiv:2602.02092v1, 2026. https://arxiv.org/abs/2602.02092
3. Shao et al. Efficient Video Diffusion Models: Advancements and Challenges. arXiv:2604.15911v1, 2026. https://arxiv.org/abs/2604.15911

### 联网补充的一手资料

4. Zheng et al. HASTE: Training-Free Video Diffusion Acceleration via Head-Wise Adaptive Sparse Attention. arXiv:2605.14513, 2026. https://arxiv.org/abs/2605.14513
5. Su et al. 6Bit-Diffusion: Inference-Time Mixed-Precision Quantization for Video Diffusion Models. arXiv:2603.18742, 2026. https://arxiv.org/abs/2603.18742
6. Yang et al. StreamFusion: Scalable Sequence Parallelism for Distributed Inference of Diffusion Transformers on GPUs. arXiv:2601.20273, 2026. https://arxiv.org/abs/2601.20273
7. Yuan and Li. Accelerating Video Generation Inference with Sequential-Parallel 3D Positional Encoding Using a Global Time Index. arXiv:2603.06664, 2026. https://arxiv.org/abs/2603.06664
8. Zhu et al. Flash-VAED: Plug-and-Play VAE Decoders for Efficient Video Generation. arXiv:2602.19161, 2026. https://arxiv.org/abs/2602.19161
9. Chen et al. Sparse-vDiT: Unleashing the Power of Sparse Attention to Accelerate Video Diffusion Transformers. arXiv:2506.03065, 2025. https://arxiv.org/abs/2506.03065
10. Glorian et al. LVSA: Training-Free Sparse Attention for Long Video Diffusion. arXiv:2605.31057, 2026. https://arxiv.org/abs/2605.31057
11. Xu et al. Sparse Forcing: Native Trainable Sparse Attention for Real-time Autoregressive Diffusion Video Generation. arXiv:2604.21221, 2026. https://arxiv.org/abs/2604.21221
12. Lv et al. Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention. arXiv:2602.04789, 2026. https://arxiv.org/abs/2602.04789
13. Nie et al. Transition Matching Distillation for Fast Video Generation. arXiv:2601.09881, 2026. https://arxiv.org/abs/2601.09881
14. Cheng et al. Dynamic-in-Few-Step: Unifying Dynamic Computation and Few-Step Distillation for Efficient Video Generation. arXiv:2607.06631, 2026. https://arxiv.org/abs/2607.06631
15. Zhao et al. minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models. arXiv:2605.30263, 2026. https://arxiv.org/abs/2605.30263

## 附录 A：数字解释与限制

- 本报告对 token 的示例计算基于论文披露的 VAE 下采样率；不同模型的 causal 边界、padding、patchify 和条件 token 会改变绝对值。
- 参数体积采用 P×bitwidth/8 的理论下界，未计量化元数据、对齐、embedding、norm 和 runtime buffer。
- 论文 speedup 来自不同硬件、实现和质量约束，不构成横向排行榜。42.3×、1.5–1.8×、1.92×等数字只在各自实验条件下成立。
- 综述与若干 2026 年论文仍为 arXiv 预印本；报告将其作为最新技术趋势证据，而非已被大规模生产验证的结论。
- “存储”在报告中分别指持久化 checkpoint/磁盘、主存和 GPU HBM；三者容量与带宽不可互换。

## 附录 B：快速决策清单

- 模型是否能完整常驻目标 GPU？若不能，先量化、分片或换更少 token 的架构。
- 当前 wall time 中 DiT、VAE、offload、通信各占多少？只优化占比最大的部分。
- 使用的是 diffusion steps 还是 NFE？CFG 是否双前向？
- sparse attention 的 mask 是否 tile-aligned？路由与 permutation 占比多少？
- cache 增加了多少 HBM，是否挤出权重或降低 batch？
- 多卡 speedup 是否同时报告 GPU-seconds 与跨节点流量？
- 质量评估是否覆盖高运动、镜头切换、长时身份稳定和闪烁，而非只看逐帧指标？
