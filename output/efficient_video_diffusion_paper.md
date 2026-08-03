# 高效视频扩散模型：算法演进、系统负载与计算—带宽—存储协同优化

## 摘要

视频扩散模型依靠扩散Transformer在高维时空潜在空间中进行迭代去噪，已经成为高质量视频生成的重要技术路线。然而，视频分辨率、帧数与采样步数的共同增长，使其推理成本远高于图像生成，并形成由计算吞吐、显存容量、片外带宽和跨设备通信共同决定的系统瓶颈。现有研究通常以采样步数、理论浮点运算量或局部算子加速比衡量效率，却容易忽略两类关键事实：其一，不同方法改变的资源维度并不相同，减少网络调用次数不必然降低峰值显存，减少理论FLOPs也不必然降低数据搬运；其二，多种加速技术叠加后会发生瓶颈迁移，扩散主干被加速后，视频自编码器、KV缓存、参数卸载与多卡通信可能成为新的主导因素。本文以《Efficient Video Diffusion Models: Advancements and Challenges》的分类体系为基础，将高效视频扩散方法归纳为采样步蒸馏、高效注意力、模型压缩以及缓存与轨迹优化四类，并在统一负载模型下分析其技术动机、算法机制及对计算、带宽和存储的影响。在此基础上，本文进一步讨论2026年以来的代表性进展，包括Transition Matching Distillation、HASTE、LVSA、Sparse Forcing、6Bit-Diffusion、Dynamic-in-Few-Step、FSVideo与StreamFusion。本文的核心观点是，高效视频扩散并非单一的模型压缩问题，而是网络调用次数、时空token规模、数值精度、状态复用和硬件执行规则性之间的联合优化问题。面向实际部署，只有同时控制每次生成的总计算量、HBM与互联传输字节数以及峰值驻留状态，算法层面的效率改进才能稳定转化为端到端收益。

**关键词：** 视频扩散模型；扩散Transformer；高效生成；稀疏注意力；少步蒸馏；模型量化；特征缓存；系统协同设计

## 1 引言

视频生成模型正在从短时、低分辨率的离线合成走向高分辨率、长时域和交互式生成。与图像扩散相比，视频扩散不仅需要在空间维度上生成大量视觉内容，还必须在时间维度上保持运动、身份、场景布局和镜头演化的一致性。为了获得较高的生成质量，主流系统通常先使用视频变分自编码器将像素视频压缩到潜在空间，再由大规模扩散Transformer（Diffusion Transformer，DiT）执行数十次去噪网络调用，最后通过视频解码器恢复RGB帧。这一流程同时放大了三个维度的成本：视频分辨率和时长决定时空token数量，DiT深度与宽度决定单次前向成本，扩散采样过程决定相同主干被重复执行的次数。

这一负载结构使视频扩散面临不同于传统视觉网络的部署困难。首先，全时空注意力的计算量随token数平方增长，当分辨率或帧数增加时，attention score、KV访问和中间工作区会迅速膨胀。其次，十亿到百亿参数级DiT的权重本身已接近单卡显存容量；如果权重无法常驻高带宽显存，参数卸载会把计算问题转化为PCIe或NVLink搬运问题。再次，流式视频生成虽然通过因果分块降低了首帧时延，却需要长期保存历史KV状态，因而将一次性的激活峰值转化为持续增长的缓存容量和带宽压力。最后，当少步蒸馏、稀疏注意力和量化显著加速DiT后，原先占比较低的视频自编码器解码、跨卡通信和服务调度会根据Amdahl定律成为新的系统瓶颈。

已有高效扩散研究从不同角度缓解上述问题。采样步蒸馏减少去噪网络调用次数；高效注意力降低长序列中的二次复杂度和内存流量；量化、剪枝与潜在空间压缩减少参数、激活或token规模；特征缓存和KV缓存复用跨时间步或跨视频块的历史计算；并行推理则通过多设备分摊单卡容量和时延。然而，这些方法往往使用不同的实验条件和效率指标。局部attention kernel的加速比不能直接等同于完整视频生成速度，模型权重的压缩比不能代表峰值显存变化，双卡时延也不能与单卡基线直接比较。缺少统一的资源分析框架，是当前高效视频扩散研究难以形成可部署结论的重要原因。

本文围绕三个问题展开。第一，视频扩散的计算、带宽和存储负载如何由分辨率、时长、模型规模与采样深度共同决定？第二，四类高效扩散方法分别改变了负载模型中的哪一项，又引入了哪些新的开销？第三，最新技术为何逐渐从独立优化转向少步、稀疏、低精度、缓存和硬件kernel的联合设计？本文并不把不同方法简单排列为速度榜单，而是分析其作用对象、资源转移和适用边界。本文的主要贡献包括：建立面向视频DiT的统一负载模型；以计算、带宽和存储三个维度重新解释四类高效方法；总结2026年最新研究所体现的联合优化趋势；提出面向论文评测和工程部署的统一指标体系与研究议程。

## 2 视频扩散模型的负载形成机制

### 2.1 从像素视频到时空token

设输出视频包含F帧，每帧的像素高度与宽度分别为H和W。视频自编码器在时间、高度和宽度方向的下采样率分别为r_t、r_h和r_w，DiT在潜在网格上的patch大小为p_t、p_h和p_w，则视觉token数可近似表示为

> **式（1）**　N = ⌈F/(r_t p_t)⌉⌈H/(r_h p_h)⌉⌈W/(r_w p_w)⌉。

该表达式区分了三个容易混淆的概念：H×W描述输出像素分辨率，自编码器输出的是潜在网格单元，潜在网格进一步patchify后才形成DiT token。计算负载直接由token数N决定，而不是直接由像素总数决定，但像素分辨率通过VAE压缩率和patch大小间接决定N。若高度和宽度各扩大两倍，N近似扩大四倍；在全注意力区域，相关性计算可扩大约十六倍。若视频时长扩大两倍，N近似扩大两倍，而全注意力计算近似扩大四倍。因此，视频扩散从480p扩展到720p或从5秒扩展到长视频时，时延和显存通常呈非线性增长。

### 2.2 单次DiT前向的计算结构

设Transformer隐藏维度为d，前馈网络扩张比为m。忽略常数较小的归一化和条件注入，一个Transformer块的主要计算量可写为

> **式（2）**　C_block ≈ (4+2m)Nd² + 2N²d。

第一项包含Q、K、V、输出投影与前馈网络，随N线性增长；第二项包含attention score和value aggregation，随N平方增长。令两项相等，可得到全注意力开始占据主导的大致临界点N*≈(2+m)d。当m=4、d=4096时，N*约为24576。这个结果说明，“Video DiT一定由attention主导”并不总是成立：潜在空间压缩很强、token数较小时，宽模型的线性层和FFN仍可能占主要计算；随着分辨率和时长增长，N²项才迅速接管。

若模型包含L个Transformer块，真实网络调用次数为S，则扩散主干总计算近似为

> **式（3）**　C_DiT ≈ SL[(4+2m)Nd²+2N²d]。

这里使用网络调用次数（number of function evaluations，NFE）而非扩散步数，是因为classifier-free guidance可能对条件与无条件分支分别执行一次前向，使一个采样步对应两次NFE。式（3）揭示了高效视频扩散的两个基本优化轴：采样步蒸馏主要减少S，潜在空间压缩、稀疏注意力、量化与剪枝主要减少每次前向的成本。

### 2.3 存储层次与数据移动

推理峰值显存不是参数量的简单函数，而是常驻权重、激活、attention工作区、VAE特征、缓存、通信buffer和运行时碎片的叠加：

> **式（4）**　M_peak≈M_weight+M_act+M_attn+M_VAE+M_cache+M_comm+M_runtime。

P个参数、每个权重b_w bit时，权重容量下界为M_weight=P b_w/8。14B模型的BF16权重约为28 GB，8 bit约为14 GB，4 bit约为7 GB，但这些数字不含量化scale、embedding、激活和工作区。训练阶段还需保存梯度、高精度master weights与Adam一、二阶状态，朴素混合精度训练通常达到约16—20 bytes/parameter，因此14B模型仅训练状态就可能需要224—280 GB。

从系统角度看，容量与带宽不可分割。如果14B BF16权重无法常驻GPU，而每次NFE都需要从主存搬运28 GB参数，那么60 NFE对应的理论传输下界为1.68 TB。设常驻比例为f_resident，参数卸载量可写为

> **式（5）**　D_offload≥S(1-f_resident)M_weight。

当生成时延目标为T时，链路平均带宽至少为D_offload/T。若T为60秒，则完全卸载情况下仅参数搬运就需要约28 GB/s，已经接近高端PCIe链路的有效上限，且尚未计入激活、VAE和协议开销。这解释了为什么量化在显存临界场景中会产生非线性收益：它不仅减少GEMM成本，还可能使模型从频繁卸载转为完全常驻。

硬件执行上限可用Roofline关系描述：Performance≤min(P_peak,BW·AI)，其中AI为FLOPs/byte。FlashAttention主要通过减少HBM字节数提高AI；量化同时降低字节数并提高低精度峰值算力；不规则稀疏虽然减少FLOPs，却可能因访存离散和负载不均降低有效带宽。因此，理论计算下降不必然转化为端到端加速。

## 3 高效视频扩散方法的统一分类

高效视频扩散研究可归纳为四个相互关联的范式。采样步蒸馏压缩去噪深度，直接减少式（3）中的S；高效注意力改变N²项的实现或有效连接数量；模型压缩降低参数位宽、网络结构或进入DiT的token数；缓存与轨迹优化则利用跨时间步、跨层和跨视频块的冗余，避免重复计算或重新分配计算预算。这四类方法并非互斥。少步蒸馏与潜在空间压缩作用于不同乘数，具有近似乘法潜力；量化与稀疏注意力共同改变计算和带宽；流式蒸馏必须与KV缓存管理联合设计；主干加速后又需要VAE与通信优化处理瓶颈迁移。

从资源角度看，四类方法的差异比算法名称更重要。采样步蒸馏显著降低每个视频的总计算和总数据流量，但通常不显著降低一次前向的峰值显存。稀疏注意力降低attention计算与工作区，却可能增加路由、索引和不规则访存。量化降低权重和激活字节数，只有在硬件具有原生低比特kernel时才稳定提高吞吐。Feature cache以额外显存和缓存流量换取少算若干block，KV cache以历史存储换取不重复生成K/V。多卡并行降低单卡容量，却提高互联带宽和同步成本。后文将沿这一资源逻辑讨论各类方法。

## 4 采样步蒸馏：压缩去噪深度

### 4.1 一致性蒸馏与视频运动约束

一致性蒸馏的基本假设是，同一去噪轨迹上的不同噪声状态应被映射到相同的干净样本。在线student对较晚状态z_t进行预测，EMA teacher将其推进到较早状态z_s，再约束二者的一致性。该目标不再要求student复现所有中间步骤，因此能够把几十次网络调用压缩到4—8次。

视频场景的困难在于逐元素一致性损失容易被低运动区域主导。若大部分画面为背景，student通过减小运动幅度也能获得较低平均误差，最终产生清晰但近静态的视频。VideoLCM、Motion Consistency Model和DCM因此引入运动表征M(·)，形成L=d(ẑ_0^t,ẑ_0^s)+λ_m d(M(ẑ_0^t),M(ẑ_0^s))。M可以是帧差、光流特征或可学习运动提取器。这个改动的意义不是额外加速，而是使模型能够承受更激进的NFE减少。

一致性蒸馏的推理收益近似与NFE减少成正比。由于单次前向的网络和token结构没有改变，峰值显存变化有限，但权重、激活和attention数据的总读取次数显著下降。如果基线需要参数卸载，NFE减少也会等比例减少跨PCIe传输。其代价集中在训练阶段：student、EMA teacher和运动网络需要额外计算与存储。总体而言，一致性蒸馏稳定、适合可控生成和4—8步服务，但其保守目标限制了进入一步生成的能力。

### 4.2 分布蒸馏与极少步生成

Distribution Matching Distillation不再要求student遵循teacher的局部轨迹，而是直接匹配二者的输出分布。DMD从student输出重新构造加噪状态，利用teacher score与student-induced fake score之差更新generator。由于student可以学习不同于teacher的短路径，DMD及DMD2成为1—4步视频生成的主要路线。

分布蒸馏把推理计算压缩到极低水平，却显著提高训练复杂度。训练通常同时涉及generator、teacher score model和fake score model，并可能引入discriminator或reward model。百亿参数视频模型很难将这些组件同时放入单卡，必须依赖FSDP、ZeRO、参数卸载或交替更新。因而分布蒸馏体现了典型的“高训练投入换低推理成本”，适合推理调用量大、能够摊销一次性训练成本的部署。

在流式视频中，分布蒸馏还需解决因果生成的误差累积。CausVid将双向teacher蒸馏为causal student，但训练时使用真实历史、推理时使用自身生成历史，存在exposure bias。Self-Forcing在训练中展开自回归rollout，使student条件于自己生成的KV缓存，从而缩小训练与推理分布差异。这个机制提高长时稳定性，却使训练计算、KV状态和activation随rollout长度上升。由此可见，实时视频的核心不仅是少步，还包括因果训练、缓存预算与误差控制。

### 4.3 对抗蒸馏与感知质量恢复

极少步模型常出现纹理过平滑、运动减弱和感知锐度下降。对抗蒸馏通过discriminator区分真实视频与student输出，为generator提供感知层面的分布约束。由于discriminator仅在训练中使用，对抗目标几乎不增加部署计算、带宽和显存，其主要作用是补偿少步蒸馏的质量损失，而不是独立减少NFE。

独立GAN式训练在视频基础模型上不稳定，因此ADD、LADD和DMD2等方法通常把对抗损失叠加在一致性或分布蒸馏之后。该路线能够增强细节与运动感，但也可能放大伪影和模式崩溃。更可靠的策略是先获得稳定的4步或少步student，再进行有限的对抗与偏好优化。

### 4.4 采样步蒸馏的新进展

Transition Matching Distillation进一步分解了一次少步转移内部的重复计算。TMD将预训练DiT拆为承担主要语义提取的backbone和由少量末层构成的flow head。每个外层transition只执行一次大backbone，flow head在共享语义特征上完成多个内部更新。若backbone成本为C_B、head成本为C_H、内部更新次数为K，则原始成本K(C_B+C_H)被改写为C_B+KC_H。当C_B远大于C_H时，可显著减少计算与权重读取；代价是语义特征必须在内部更新期间保持驻留，可能形成新的activation带宽压力。

Dynamic-in-Few-Step则将少步蒸馏与逐timestep结构化剪枝联合优化。传统方法先把teacher蒸馏为固定4步模型，再对所有步骤使用相同网络；该工作认为不同噪声阶段所需block不同，因而学习step-specific Mixture-of-Models。其在Wan-14B上相对4步模型额外移除24%的per-step FLOPs并获得1.2倍wall-clock增益，相对50步teacher达到约30倍。该结果表明，高效视频扩散正从独立压缩S或C_step转向二者的联合优化，但专用执行引擎与结构调度成为兑现收益的必要条件。

## 5 高效注意力：从精确IO优化到内容自适应稀疏

### 5.1 FlashAttention与IO感知执行

FlashAttention不改变attention的数学定义，而是改变Q、K、V在GPU存储层次中的移动方式。标准实现可能把N×N score和softmax概率写入HBM，长序列下中间数据远大于模型输出。FlashAttention将Q、K、V切成可放入片上SRAM的tiles，在片上完成分块score、online softmax和value累积，仅将最终结果写回HBM。其理论FLOPs基本不变，但显著降低HBM流量和attention工作区。

因此，FlashAttention应被视为所有稀疏或线性方法的系统基线。若新方法只与显式materialize attention matrix的朴素实现比较，所得加速比无法代表现代Video DiT部署。FlashAttention-3进一步利用异步执行与低精度路径提高H100利用率，说明精确attention仍可通过软硬件协同获得较高效率。

### 5.2 静态稀疏注意力

静态稀疏利用视频attention中反复出现的时空局部、对角带、条纹或block-diagonal结构，预先规定每个query可以访问的key blocks。若keep ratio为ρ，attention主项由2N_qN_kd降为约2ρN_qN_kd，score工作区也近似按ρ下降。由于mask固定，系统可以预先排布tile、融合相同pattern的heads并避免在线路由。

Sparse-vDiT通过分析不同层和head的attention map，发现pattern与层深、head位置的相关性高于与输入内容的相关性，因此在校准集上进行硬件感知搜索，为每个head选择dense、对角、多对角、条纹或跳过策略。该方法在CogVideoX1.5、HunyuanVideo和Wan2.1上报告1.67—2.38倍理论FLOP reduction，实际推理加速为1.58—1.85倍。理论与实际差距来自QKV投影和FFN不变、attention并非完整链路的全部成本，以及kernel启动和非理想利用率。

静态稀疏的优势是执行规则、元数据小和跨硬件可复现，缺点是无法根据当前视频恢复被mask遗漏的长程依赖。超出训练时长时，固定局部窗口容易产生循环和冻结。LVSA为此在结构化窗口之外引入旋转全局锚点，使远距离块在不同层或时间获得连接机会。配合FlashInfer kernel，LVSA在多个模型上报告约2.98—3.33倍compute reduction，并使原本单卡OOM的更长视频生成成为可能。这说明稀疏attention的价值不仅在于降低时延，也在于扩大可处理的时间上下文。

### 5.3 动态稀疏注意力

动态稀疏根据当前输入、层、head和timestep选择有效K/V，理论上在相同稀疏率下比固定mask更能保存重要交互。然而，其真实成本应表示为C_route+ρC_dense+C_gather/scatter+C_metadata。若mask预测接近完整QK计算，或索引造成严重不连续访存，稀疏算法可能减少FLOPs却无法降低wall time。

RainFusion2.0通过块级代理降低路由开销。该方法将token划分为三维时空blocks，以block mean作为代表Q和K估计块间相关性，并通过时空感知重排提高块内相似度；first-frame sink则保留所有query与首帧key的连接，以维持图生视频中的身份和结构。若每块包含B个token，代理score规模从N²降为(N/B)²。该方法在80% sparsity下获得1.5—1.8倍端到端加速，说明attention主项即使理论保留20%，整个系统也仍受FFN、投影、VAE和路由限制。

HASTE进一步利用扩散过程的跨timestep稳定性。相邻步骤的Q/K漂移较小时，Temporal Mask Reuse直接复用上一步mask；Error-guided Budgeted Calibration则根据各head的输出误差分配不同稀疏预算，而非使用统一top-p。该方法在720p上获得最高1.93倍加速。其意义在于，动态稀疏不再只是当前前向中的token选择问题，而成为跨timestep的控制与预算分配问题。

在流式模型中，稀疏attention还必须与KV缓存共同设计。Sparse Forcing观察到少数显著visual blocks会在多个chunks中持续受到关注，形成隐式长期记忆；该方法学习保留persistent blocks，并在滑动窗口内选择局部邻域，使用PBSA kernel融合稀疏attention和cache update。其峰值KV缓存降低42%，且20秒和1分钟视频的加速高于5秒，表明缓存受控带来的收益会随时长增长。

### 5.4 线性与混合注意力

即使ρ很小，ρN²在分钟级视频上仍是二次复杂度。线性attention使用核映射改变乘法次序，将φ(K)^TV预先聚合，再与φ(Q)相乘，使复杂度从O(N²d)降至O(Nd²)。它同时避免N²工作区，并可将历史压缩成固定状态。然而，pairwise交互被低秩或核化压缩，容易过度平滑远程运动并削弱身份召回。

因此，近期更可行的方向是hybrid attention：局部或重要token使用精确/稀疏attention，低重要token通过linear branch提供低成本残差。该设计将线性attention从全量替代方案转变为稀疏信息的补偿机制。其最终效率取决于分支比例、融合kernel和状态读写；当N不够大时，优化良好的FlashAttention仍可能快于线性实现。

## 6 模型压缩：精度、结构与潜在空间

### 6.1 量化感知训练与训练后量化

量化通过降低权重和激活位宽减少模型容量、HBM流量和跨卡通信，并在硬件支持时提高低精度GEMM吞吐。视频扩散量化的困难在于activation分布随timestep显著变化，且不同层、token与运动区域对误差的敏感性不同。静态scale可能在高噪声阶段浪费动态范围，在细节恢复阶段产生饱和，最终表现为跨帧闪烁而非单帧明显失真。

QAT在训练前向中注入fake quantization，使模型学习吸收低比特误差，并使用teacher alignment、token saliency与timestep-aware scale恢复质量。其适合NVFP4或W4A4等激进精度，但训练成本较高。PTQ则冻结pretrained backbone，只用少量校准数据优化scale、rotation、smoothing与bit allocation，是当前更可部署的路径。Q-DiT、ViDiT-Q等方法为不同timestep选择不同量化配置，抽象目标是最小化所有层与时间步的加权重建误差。

量化对资源的影响需分开理解。Weight-only 8 bit可把14B模型权重从约28 GB降至14 GB，主要减少容量与权重读取；activation 8 bit还降低层间HBM和all-to-all字节数。若低比特kernel不原生，dequantization和packing可能抵消计算收益。若量化使模型从无法常驻变为完全常驻，避免参数卸载带来的收益则可能远大于GEMM本身的加速。

6Bit-Diffusion代表了动态混合精度的新趋势。该方法利用block输入—输出residual与量化敏感性的相关性，在推理时为稳定层选择NVFP4、为敏感层保留INT8，并使用Temporal Delta Cache跳过跨timestep变化很小的block。量化减少实际执行层的字节数与计算，Delta Cache减少执行次数；其报告1.92倍端到端加速和约3.32倍内存压缩。代价是predictor、双kernel路径、cache状态与调度复杂性，体现出量化正从统一bitwidth转向与动态计算联合设计。

### 6.2 VAE与潜在空间压缩

VAE压缩作用于DiT之前，因而具有比权重量化更高的乘法层级。若新旧token比例为α，DiT线性项近似降为α，attention项近似降为α²，且这一收益作用于所有L层和S次NFE。CV-VAE、IV-VAE试图保持图像VAE先验并增强时间压缩；OD-VAE、WF-VAE与LeanVAE通过全维度、wavelet或轻量结构提高压缩效率；VidTwin、Hi-VAE与DC-VideoGen进一步分离静态结构和动态运动。

FSVideo将这一方向推进到64×64×4下采样和128-channel latent。对于121帧、720×1280视频，8×8×4 VAE约产生446400个潜在网格单元，而64×64×4 VAE在向上取整后约产生7440个，减少约60倍。FSVideo在2张H100上将Wan2.1的822.1秒降至19.4秒，报告42.3倍加速。由于两者NFE接近，该结果主要说明每次前向token成本的下降。

潜在空间压缩并非无损。FSAE需要更多latent channels、专门正则化、CNN upscaler与8-NFE refiner恢复细节；两个14B主干增加持久化存储和模型切换压力。高压缩VAE一旦丢失运动、身份或高频信息，后续DiT很难恢复。因此，VAE研究必须同时评估重建质量、潜在空间可生成性和最终视频质量，不能只报告下采样倍数。

### 6.3 Token与模型结构剪枝

Token pruning根据显著性、运动或密度删除、合并或延迟处理冗余token。若token数降为αN，后续attention理论降为α²，线性层降为α；但importance scoring、sort、gather/scatter和token restoration会增加不规则计算与HBM访问。视频中的query token被删除后不再更新，错误可直接表现为局部画面缺失或运动不连续，因此当前token pruning的质量—速度上限通常弱于KV-only sparse attention。

Model pruning直接移除DiT blocks、heads、channels或FFN宽度。结构化block pruning可同时减少计算、权重读取、activation和同步次数，非结构化零权重若没有稀疏GEMM支持则很难降低wall time。视频模型深层常承担运动与全局一致性，简单按幅值删除风险较高，较可靠的方案是sensitivity-guided pruning与distillation repair。Dynamic-in-Few-Step进一步表明，不同timestep应使用不同子结构，未来模型剪枝将更接近动态执行策略而非固定小模型。

## 7 缓存、轨迹与并行执行

### 7.1 Feature Cache

Feature cache利用相邻去噪步骤的中间特征冗余。设F_t^l为第t步第l层新计算特征，C_(t-Δ)^l为历史缓存，则系统通过门控g_t^l决定使用缓存还是刷新：F̃_t^l=g_t^lC_(t-Δ)^l+(1-g_t^l)F_t^l。TeaCache类方法使用timestep先验或固定schedule，FasterCache使用特征或residual相似度，DiCache类方法让模型或预测器决定刷新。

Feature cache减少被跳过block的FLOPs和权重读取，却增加缓存特征的HBM读写与显存。其净收益为被跳过计算时间减去评分、缓存读写与误差补偿时间。如果缓存的是B×N×d的BF16特征，单层容量约为2BNd bytes；在高分辨率视频中，多层、多timestep缓存可能挤占常驻权重空间并触发offload。因此，cache不是无条件的算力换速度，而是计算、显存和带宽之间的交换。

缓存误差还会沿后续去噪步骤传播。可靠方法从固定间隔复用转向动态刷新、局部更新和显式误差补偿。高运动区域、镜头切换和少步模型需要更保守的阈值，因为每一步对最终轨迹的影响更大。

### 7.2 KV Cache与流式记忆

自回归视频生成中，KV cache避免为历史chunks重复计算K/V projection。其容量近似为M_KV≈2LN_historyd_kv b_a/8，随历史token数线性增长；当前query访问历史K/V的attention计算与带宽也随历史增加。因此，KV cache解决了历史重算，却引入长期容量和持续读取问题。

实际系统通过滑动窗口与eviction固定预算，通过compressed memory把历史汇总为少量memory tokens，或通过selective KV保留关键帧、对象与长期persistent blocks。首帧或关键帧attention sink用于维持身份和场景布局。错误淘汰可能不会立即失败，而是在多个chunks之后表现为身份漂移和拼接断裂，因而必须进行长时评估。

### 7.3 轨迹修改与多阶段生成

Noise/state modification通过重排初始噪声、窗口融合、跨帧token替换或latent shift改善长视频一致性，新增操作通常为O(Nd)，计算量低但偏内存带宽受限。Trajectory modification则重新分配整条去噪路径的预算，例如优化timestep位置、使用adaptive sub-step，或采用低分辨率生成与高分辨率refiner的coarse-to-fine流程。后者能够把昂贵的高分辨率计算限制在少数步骤，但需要多个checkpoint、latent传输和模型切换。

FSVideo的base—refiner结构即属于轨迹和分辨率共同重设计。评价这类系统不能只看base DiT的NFE，而应求和每个阶段的模型规模、token数、NFE、VAE与模型加载时间。

### 7.4 分布式推理

当单卡无法容纳长序列或大模型时，sequence、context、patch、tensor和CFG parallel可以分摊计算和activation。多卡层时延可近似写为T_layer≈max(C_layer/(GP_eff),D_comm/BW_link)+T_sync+T_imbalance。增加GPU数量只能降低计算项，all-to-all、all-gather、reduce-scatter和同步不会按1/G下降。

xDiT提供多种DiT并行维度的统一执行框架。StreamFusion进一步针对节点内与节点间带宽差异采用拓扑感知sequence parallel，使用Torus Attention重叠跨机all-to-all与本地计算，并以one-sided communication减少同步，报告平均1.35倍、最高1.77倍优于既有方案。这一结果表明，多卡扩展的中心不是并行维度本身，而是通信拓扑、算通重叠和负载均衡。

并行系统应同时报告wall-clock latency与GPU-seconds。8张GPU用1秒完成的任务消耗8 GPU-s，不一定比1张GPU用6秒更经济。动态稀疏还可能使不同rank的有效blocks不均，最慢rank决定整层时间，因此算法稀疏必须与分布式负载均衡共同设计。

## 8 计算、带宽与存储的统一影响分析

不同加速方法对三种资源的作用并不同步。少步蒸馏减少总前向次数，因此显著降低总计算和总数据流量，但一次前向的峰值工作集几乎不变。FlashAttention降低HBM流量和attention工作区而不减少数学连接；结构化稀疏同时减少连接和工作区，却引入mask与执行规则性问题。量化首先减少权重与激活字节数，在原生kernel上才进一步降低计算时延。VAE压缩减少token数，对线性层、attention、缓存与通信同时生效，但可能增加decoder和refiner。Feature cache减少计算却增加显存与缓存读写，KV cache减少历史projection重算却形成随时长增长的驻留状态。多卡并行降低单卡容量，却增加互联流量和总设备成本。

为了估计组合加速，设基线中线性层、attention和不受优化部分的占比分别为q_lin、q_att和q_other，少步比例α_s=S_new/S_old，token比例α_n=N_new/N_old，attention keep ratio为ρ，则主干归一化成本可近似表示为

> **式（6）**　R≈α_s(q_linα_n+q_attρα_n²+q_other)。

该式展示了少步与token压缩的乘法潜力，也显示token压缩与稀疏attention会在N²项上产生部分重叠。若主干原本只占端到端时延比例q，即使主干加速a倍，完整系统加速仍受Amdahl定律限制：Speedup=1/[(1-q)+q/a]。Flash-VAED约6倍decoder加速只带来最高36%的端到端改善，正是瓶颈迁移的实例。

资源优化还存在非连续效应。当显存需求略高于设备容量时，系统可能被迫启用参数卸载、更多分片或activation重算，性能会突然下降，而不是平滑变化。量化或VAE压缩即使只减少一部分容量，也可能使系统跨过“可完全常驻”的临界点，获得远大于局部FLOPs变化的收益。反过来，加入Feature cache可能挤出权重并触发offload，使理论少算反而变慢。

## 9 最新趋势与研究挑战

截至2026年，Video Diffusion效率研究呈现出从独立模块向联合设计演化的趋势。Light Forcing将自回归稀疏attention与FP8和轻量VAE组合，在RTX 5090上报告2.3倍加速与19.7 FPS；Sparse Forcing把持久化KV记忆和局部块稀疏原生训练进模型；6Bit-Diffusion联合动态精度与Delta Cache；Dynamic-in-Few-Step联合少步蒸馏和step-specific结构；TMD在一次少步转移内部共享backbone语义。这些工作共同说明，单一维度的优化空间正在收窄，未来收益来自NFE、token、bitwidth、cache和kernel的协同。

然而，联合优化会叠加误差。少步蒸馏、低比特、稀疏连接和cache复用各自看似质量可接受，组合后却可能在相邻timestep与chunks中累积为闪烁、运动冻结和身份漂移。未来需要把误差预算作为显式约束，在不同timestep、层、head和视频区域之间分配精度、稀疏率与刷新频率，而不是独立设定超参数。

长视频与实时生成仍是最困难的场景。首帧时延、持续帧率和固定显存上界必须同时满足；因果rollout的exposure bias、KV记忆增长和长时身份稳定不能通过短视频VBench充分衡量。LVSA提出用于识别循环视频失效的VQeval，也反映出现有指标可能奖励重复或近静态输出。未来评测应覆盖20秒、1分钟和多镜头视频，报告延迟失效与缓存预算。

硬件—算法协同也是关键。动态token选择如果不能映射为tile-aligned blocks，理论稀疏会被gather/scatter和负载不均抵消；FP4/FP8只有在目标设备具备原生kernel时才产生稳定收益；多卡稀疏attention还必须平衡各rank工作量。可部署方法应同时发布算法、kernel、编译配置和完整端到端profile，而不是只给理论FLOPs。

最后，领域需要统一评测口径。最低限度应报告输出帧数与像素分辨率、潜在网格与token数、真实NFE、CFG实现、精度、GPU型号和数量、是否包含VAE/text/offload、峰值显存、主干与端到端时延、GPU-seconds、通信规模以及长时质量。只有在这些条件一致时，不同方法的速度和资源收益才具有可比性。

## 10 结论

高效视频扩散的本质不是简单减少FLOPs，而是控制网络调用次数、时空token规模和数据移动。采样步蒸馏减少重复深度，高效注意力降低长序列中的二次计算与HBM流量，量化和剪枝降低模型与激活成本，VAE压缩从上游缩短整个生成链的序列，缓存和轨迹方法复用跨步骤与跨块状态，分布式系统则在容量与通信之间重新分配负载。

这些方法不存在脱离硬件与场景的统一最优解。权重无法常驻时，应优先量化与布局；activation或attention OOM时，应优先VAE/token压缩、FlashAttention和sequence parallel；NFE主导时，4—8步蒸馏的收益最大；流式历史增长时，必须控制KV预算；主干充分加速后，应重新优化VAE decoder、通信和服务调度。理论上近似正交的方法可以产生乘法收益，但误差、元数据和瓶颈迁移使端到端结果远小于局部加速的简单乘积。

未来高效视频扩散将从“给固定模型增加一个加速模块”转向“联合学习何时计算、计算哪些token、使用何种精度、保留哪些历史以及如何映射到硬件”。只有将生成质量、长时稳定、计算吞吐、带宽和存储放入同一优化问题，视频扩散才可能从昂贵的离线生成技术发展为可持续部署的实时生成基础设施。

## 参考文献

1. Shao S, Bai L, Wan P, et al. Efficient Video Diffusion Models: Advancements and Challenges. arXiv:2604.15911, 2026. https://arxiv.org/abs/2604.15911
2. Chen A, Liu Y, Huang J, et al. RainFusion2.0: Temporal-Spatial Awareness and Hardware-Efficient Block-wise Sparse Attention. arXiv:2512.24086, 2025. https://arxiv.org/abs/2512.24086
3. FSVideo Team, Chen Q, Fang Z, et al. FSVideo: Fast Speed Video Diffusion Model in a Highly-Compressed Latent Space. arXiv:2602.02092, 2026. https://arxiv.org/abs/2602.02092
4. Chen P, Zeng X, Zhao M, et al. Sparse-vDiT: Unleashing the Power of Sparse Attention to Accelerate Video Diffusion Transformers. arXiv:2506.03065, 2025. https://arxiv.org/abs/2506.03065
5. Zheng X, Ma Y, Xu J, et al. HASTE: Training-Free Video Diffusion Acceleration via Head-Wise Adaptive Sparse Attention. arXiv:2605.14513, 2026. https://arxiv.org/abs/2605.14513
6. Glorian G, Lamprou I, Zhang Z, et al. LVSA: Training-Free Sparse Attention for Long Video Diffusion. arXiv:2605.31057, 2026. https://arxiv.org/abs/2605.31057
7. Xu B, Du Y, Liu Z, et al. Sparse Forcing: Native Trainable Sparse Attention for Real-time Autoregressive Diffusion Video Generation. arXiv:2604.21221, 2026. https://arxiv.org/abs/2604.21221
8. Su R, Zhang J, Yuan Z, et al. 6Bit-Diffusion: Inference-Time Mixed-Precision Quantization for Video Diffusion Models. arXiv:2603.18742, 2026. https://arxiv.org/abs/2603.18742
9. Nie W, Berner J, Ma N, et al. Transition Matching Distillation for Fast Video Generation. arXiv:2601.09881, 2026. https://arxiv.org/abs/2601.09881
10. Cheng Y, Yao S, Qi Z, et al. Dynamic-in-Few-Step: Unifying Dynamic Computation and Few-Step Distillation for Efficient Video Generation. arXiv:2607.06631, 2026. https://arxiv.org/abs/2607.06631
11. Lv C, Shi Y, Huang Y, et al. Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention. arXiv:2602.04789, 2026. https://arxiv.org/abs/2602.04789
12. Yang J, Wu J, Ding Y, et al. StreamFusion: Scalable Sequence Parallelism for Distributed Inference of Diffusion Transformers on GPUs. arXiv:2601.20273, 2026. https://arxiv.org/abs/2601.20273
13. Zhu L, Huang Y, Ge X, et al. Flash-VAED: Plug-and-Play VAE Decoders for Efficient Video Generation. arXiv:2602.19161, 2026. https://arxiv.org/abs/2602.19161
