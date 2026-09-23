# MP-OPD：基于多 Prompt 专家的 On-Policy 蒸馏设计

## 1. 文档目的

本文定义一个基于 P-OPD 的新项目：**MP-OPD（Multi-Prompt Expert On-Policy Distillation）**。

MP-OPD 面向推荐场景。它使用多个共享同一 `teacher_base`、但接受不同方向专用 prompt 的条件专家，在一次 student 回答上提供联合指导。第一版不训练专家、不为每个专家保存独立 checkpoint，也不训练额外 router。专家差异首先来自 prompt 中明确写入的方向侧重；第二天真实点击等 ground truth 可以作为训练期可选证据附加到对应专家 prompt，但不是专家成立的必要条件。

本文是实现前的设计规格，主要回答：

- 多个专家如何在不训练的情况下产生不同信号；
- 如何在 student 的 top-k 候选词表上计算多专家概率；
- 如何进行两层 softmax 融合；
- 如何构造稳定、带 anchor 的目标分布；
- 需要如何扩展现有 P-OPD 数据流和代码模块；
- 如何验证实现的数学正确性、信息隔离和训练稳定性。

具体的逐文件实施顺序将在本设计确认后写入独立的 `plan.md`。

### 开发与验证环境约束

当前本地设备不具备 MP-OPD 所需的 Python/CUDA 运行环境和模型权重，因此本地只进行仓库结构、静态内容、配置语法以及不依赖项目环境的检查。依赖 PyTorch、verl、Ray、FSDP、vLLM、GPU 或实际模型权重的单元测试、集成测试和训练 smoke test，必须将代码同步到具备环境和模型的服务器后执行。任何尚未在服务器运行的命令都必须明确标记为“未执行”，不能据此声称训练链路已经通过验证。

## 2. 背景与动机

原始 P-OPD 使用一个经过强化学习的代理专家，并通过以下相对变化指导 student：

$$
\Delta_\phi
=
\log\pi_\phi^+
-
\log\pi_\phi
$$

其中 $\pi_\phi^+$ 是训练后的代理专家，$\pi_\phi$ 是代理的初始模型。

推荐场景中的可用监督并不只有一种。一次推荐结果可能同时受到以下方向影响：

- **追打**：延续用户近期高强度兴趣；
- **长期**：符合长期稳定偏好；
- **复购**：识别存在重复购买或周期消费可能的商品；
- **泛化兴趣**：从已有兴趣扩展到相关但未直接点击的内容。

每个方向都有稳定的关注目标，例如复购专家需要更重视周期消费与重复购买，长期专家需要更重视跨时间窗口的稳定偏好。MP-OPD 将这种方向说明写入专用 prompt，使同一个冻结模型形成多个条件专家。若训练样本还带有第二天真实点击或行为反馈，则把它作为可选的训练期证据写入对应专家 prompt。最后将这些专家对同一条 student 回答的判断融合成蒸馏目标。

因此，每次训练包含 **五种 prompt 视图**：一个纯净 prompt，以及追打、长期、复购、泛化兴趣四个专用 prompt。这是五种输入视图，不是五个独立模型。

## 3. 核心约束

第一版 MP-OPD 遵循以下约束：

1. Student、student base 和 teacher base 都只使用同一个纯净 prompt。
2. 方向说明和可选未来证据只允许进入对应冻结专家的训练期 forward。
3. 所有专家共享同一份模型权重，不训练专家，也不复制四份 checkpoint。
4. 所有专家评估同一条 student response，不分别生成专家回答。
5. 每个回答位置使用 student 当前策略的 top-k 候选 token。
6. 所有专家必须在完全相同的候选 token ID 上计算概率。
7. 第一层 softmax 沿专家维计算，输入为专家相对变化的绝对值。
8. 第一层加权求和保留相对变化的真实正负号。
9. 第二层 softmax 沿 top-k token 维计算，输入包含 student-base 概率和有符号融合增量。
10. 蒸馏目标必须 `detach`，只有 student 更新。

## 4. 模型角色

| 角色 | 符号 | 是否训练 | 输入 |
| --- | --- | ---: | --- |
| Student 当前策略 | $\pi_\theta$ | 是 | 纯净 prompt $x$ |
| Student base / anchor | $\pi_{\mathrm{ref}}$ | 否 | 纯净 prompt $x$ |
| Teacher base | $\pi_\phi$ | 否 | 纯净 prompt $x$ |
| 第 $e$ 个条件专家 | $\pi_{\phi,e}^{\mathrm{spec}}$ | 否 | 专用 prompt $P_e(x,s_e,z_e)$ |

所有条件专家 $\pi_{\phi,e}^{\mathrm{spec}}$ 与 teacher base $\pi_\phi$ 使用同一份冻结权重。$s_e$ 是必需的方向侧重说明，$z_e$ 是可选的训练期证据；它们的区别仅来自输入上下文。

Student anchor 和专家基础模型可以指向同一个 checkpoint，也可以是不同规模的模型。第一版要求它们与 student 使用相同 tokenizer 和词表，从而可以直接共享 student top-k token ID。跨 tokenizer 支持不属于第一版范围。

## 5. 五种 Prompt 视图与专家上下文

设纯净 prompt 为 $x$，第 $e$ 个专家的方向说明为 $s_e$，可选证据为 $z_e$。$s_e$ 默认来自配置级 `expert_instructions`，样本可以选择覆盖；$z_e$ 来自样本数据。默认五种 prompt 视图为：

```text
clean         纯净 prompt：student、student_base、teacher_base 共用
chasing       追打专用 prompt
long_term     长期兴趣专用 prompt
repurchase    复购专用 prompt
generalized   泛化兴趣专用 prompt
```

每个专家输入由可配置模板构造。方向说明必须存在，训练期证据可以不存在。未提供任何 `expert_contexts` 的样本仍会使用四个配置级方向说明生成四种专家 prompt：

```text
[普通用户上下文 x]

[专家方向说明]
你负责从「复购」角度评估推荐回答。请加强关注重复购买、消耗周期和历史复购间隔。

[可选训练期证据]
该用户第二天在复购方向上的真实点击为：...

[Student 已生成的回答前缀 y_<t]
```

专家不会重新生成完整回答。训练系统使用 teacher forcing，将同一条 student response 的前缀接到四种专家 prompt 后，并查询相同 top-k 候选 token 的条件概率。即使某个方向没有 ground truth，只要该专家被启用且方向说明可用，它仍然是有效专家。

## 6. Top-k 候选集合

对于 student response 的第 $t$ 个位置，从 student 当前策略选出 top-k 候选：

$$
C_t
=
\{v_{t,1},v_{t,2},\ldots,v_{t,K}\}
$$

默认建议从 `K=10` 开始。

需要保存：

- `student_topk_ids`：形状 `[B, T, K]`；
- `student_topk_log_probs`：形状 `[B, T, K]`；
- `student_base_topk_log_probs`：形状 `[B, T, K]`；
- `teacher_base_topk_log_probs`：形状 `[B, T, K]`；
- `specialized_expert_topk_log_probs`：形状 `[B, T, K, E]`；
- `expert_mask`：形状 `[B, E]`，表示该样本哪些专家有效。

候选集由当前 student 动态产生。专家和 base 只在这些候选 ID 上计算 log-prob，避免保存完整词表分布。

## 7. 多专家相对变化

对于候选 token $v_{t,j}$ 和专家 $e$，定义专用 prompt 相对纯净 prompt 带来的变化：

$$
\Delta_{e,t,j}
=
\log\pi_{\phi,e}^{\mathrm{spec}}
(v_{t,j}\mid P_e(x,s_e,z_e),y_{<t})
-
\log\pi_\phi
(v_{t,j}\mid x,y_{<t})
$$

这一定义隔离了冻结模型本身已有的语言偏好，保留第 $e$ 个方向 prompt（以及其可选证据）带来的概率变化。

- $\Delta_{e,t,j}>0$：该专家支持提高此 token 的概率；
- $\Delta_{e,t,j}<0$：该专家支持降低此 token 的概率；
- $\Delta_{e,t,j}\approx0$：该方向 prompt 对这个 token 没有明显影响。

## 8. 第一层 softmax：专家维融合

第一层 softmax 固定位置 $t$ 和候选 token $j$，沿专家维 $e$ 计算。

### 8.1 使用绝对变化计算专家权重

$$
w_{e,t,j}
=
\operatorname{softmax}_e
\left(
\frac{|\Delta_{e,t,j}|}{\tau_E}
+ \log \rho_e
\right)
$$

其中：

- $\tau_E>0$ 是专家 softmax 温度；
- $\rho_e$ 是可选的静态专家先验，第一版默认所有专家相等；
- 不可用专家在 softmax 前被 mask 为负无穷；
- 权重计算使用绝对值，因此强烈支持和强烈反对都代表高影响力。

### 8.2 使用真实有符号变化进行融合

$$
g_{t,j}
=
\sum_{e=1}^{E}
w_{e,t,j}\Delta_{e,t,j}
$$

权重只决定影响力，实际融合仍使用带符号的 $\Delta$：

- 多个专家共同支持时，$g_{t,j}$ 为正；
- 多个专家共同反对时，$g_{t,j}$ 为负；
- 专家意见冲突时，正负信号相互抵消；
- 每个候选 token 可以得到完全不同的专家权重。

所有专家融合计算均在 `torch.no_grad()` 下执行，结果在进入 loss 前显式 `detach`。

## 9. 第二层 softmax：top-k 目标分布

不能直接对 $g_{t,j}$ 做 token softmax。若所有专家增量都为零，直接 softmax 会得到均匀分布，从而无端拉平 student。

MP-OPD 使用 student anchor 作为目标分布的基准：

$$
q_{t,j}
=
\operatorname{softmax}_j
\left(
\log\pi_{\mathrm{ref}}(v_{t,j}\mid x,y_{<t})
+ \frac{g_{t,j}}{\lambda\tau_T}
\right)
$$

其中：

- $\lambda>0$ 控制专家信号相对于 anchor 的强度；
- $\tau_T>0$ 控制 top-k 目标分布的尖锐程度；
- 第二层使用真实有符号的 $g_{t,j}$；
- 当 $g=0$ 时，目标退化为 student-base 在 top-k 内的归一化分布。

Student 在相同候选集合上的归一化分布为：

$$
p^\theta_{t,j}
=
\operatorname{softmax}_j
\left(
\log\pi_\theta(v_{t,j}\mid x,y_{<t})
\right)
$$

第一版损失为：

$$
\mathcal L_{\mathrm{MP\text{-}OPD}}
=
\frac{1}{\sum_t m_t}
\sum_t m_t
D_{\mathrm{KL}}
\left(
q_t\;\|\;p^\theta_t
\right)
$$

$m_t$ 是 response mask。实现时可使用：

$$
D_{\mathrm{KL}}(q\|p)
=
\sum_j q_j(\log q_j-\log p_j)
$$

`q` 和构造 `q` 所用的所有冻结模型输出必须停止梯度。

## 10. 完整数据流

```text
训练样本
  ├── 纯净 prompt x
  └── 四组专家上下文 {(s_e, optional z_e)}
            ↓
Student 使用 x 生成 response y
            ↓
Student 对每个 response 位置选择 top-k token ID
            ↓
Student base 在纯净 prompt 上计算 top-k log-prob
            ↓
Teacher base 在纯净 prompt 上计算公共基线 log-prob
            ↓
同一个 teacher base 对 B×E 个专用 prompt 做 forward
            ↓
恢复为 [B,T,K,E]，计算每个专家的 Δ
            ↓
专家维 softmax(abs(Δ)) + 有符号加权
            ↓
融合增量 g：[B,T,K]
            ↓
token 维 softmax(anchor_logp + g/λ)
            ↓
目标分布 q：[B,T,K]
            ↓
KL(q || student_topk_distribution)
            ↓
仅更新 Student
```

## 11. 数据格式

四个方向说明属于模型行为配置，而不是逐样本标签。建议在 `algorithm.mp_opd` 中配置：

```yaml
expert_instructions:
  chasing: "加强关注近期高强度兴趣及其延续性"
  long_term: "加强关注跨时间窗口稳定出现的长期偏好"
  repurchase: "加强关注重复购买、消耗周期和历史复购间隔"
  generalized: "加强关注相邻品类和可迁移的泛化兴趣"
```

现有 verl Parquet 行只需增加逐样本开关、可选 instruction override 和可选 evidence：

```json
{
  "data_source": "recommendation",
  "prompt": [
    {
      "role": "user",
      "content": "用户历史与推荐请求"
    }
  ],
  "ability": "Recommendation",
  "reward_model": {
    "ground_truth": "用于总体离线评测的真实行为"
  },
  "extra_info": {
    "expert_contexts": {
      "chasing": {
        "enabled": true,
        "evidence_available": true,
        "evidence": "追打方向的第二天真实点击"
      },
      "long_term": {
        "enabled": true,
        "evidence_available": false
      },
      "repurchase": {
        "enabled": true,
        "evidence_available": true,
        "evidence": []
      },
      "generalized": {
        "enabled": true,
        "instruction_override": "本样本额外强调跨品类迁移，但避免过度扩散",
        "evidence_available": true,
        "evidence": "泛化兴趣方向的第二天真实点击"
      }
    }
  }
}
```

`enabled` 与 `evidence_available` 表达不同含义：

- `enabled=false` 表示该专家被禁用，应从专家 softmax 中排除；
- `enabled=true, evidence_available=false` 表示没有额外 ground truth，但专家仍凭方向说明参与融合；
- `enabled=true, evidence_available=true, evidence=[]` 表示已经观测、但没有对应行为；这是有效证据，不等同于缺失。
- 未提供某专家的逐样本 context 时，默认 `enabled=true, evidence_available=false`，并使用配置级方向说明；
- 非空 `instruction_override` 仅覆盖当前样本的配置级方向说明。

第一版要求每个配置专家必须有非空的有效方向说明。无法构造专用 prompt（例如超长且无法安全截断）时，才把该样本上的该专家 mask 掉。数据加载后应产生固定专家顺序，避免 Python 字典顺序或数据源差异导致专家维错位。

## 12. Prompt 构造与信息隔离

应新增独立的 expert prompt builder，而不是在原始 `prompt` 上原地修改。它需要输出：

- 纯净 prompt token；
- 每个专家的专用 prompt token；
- 每个专家的 response 起始 offset；
- 专家可用性 mask；
- 用于日志的专家名称列表。

必须满足以下信息隔离条件：

1. Student rollout 输入不包含 `expert_contexts`；
2. Student 当前策略、student base 和 teacher base 都只接收纯净 prompt；
3. 只有对应的冻结专家 forward 可以访问 $s_e$ 和可选 $z_e$；
4. 验证和线上推理不依赖专家方向说明或未来证据；
5. 日志不得默认打印原始用户行为和未来点击内容。

测试中应直接断言三个纯净分支的 input IDs 完全一致，且不包含专家方向模板或证据 token。

## 13. 批处理与显存设计

四个专家不对应四份模型，而是一个共享模型上的四组输入。实现建议：

1. 将 `[B,E]` 有效专家 prompt 展平为 `[B×E]`；
2. 按 `expert_forward_micro_batch_size` 分块 forward；
3. 仅收集 `student_topk_ids` 指定 token 的 log-prob；
4. 按样本和专家索引恢复 `[B,T,K,E]`；
5. 立即计算融合结果，并在可行时释放四维中间张量。

主要额外成本来自：

- 专用 prompt 的额外长度；
- $E$ 倍冻结模型 forward；
- `[B,T,K,E]` 的临时 log-prob 张量。

第一版应优先保证数学正确性，再增加并发 forward、缓存公共前缀或 chunked expert execution。

## 14. 与现有 P-OPD 的关系

现有 P-OPD 已经提供以下可复用能力：

- student on-policy rollout；
- `opd_top_k` 候选提取；
- 在指定 token ID 上计算冻结模型 log-prob；
- student-base anchor；
- FSDP worker 和动态 micro-batch；
- response mask、checkpoint、验证和指标记录。

现有 `multi_teacher_distill` 不能直接满足 MP-OPD。它主要根据样本字段在教师之间进行选择，属于“一条样本选择一个教师”，而 MP-OPD 需要“一条样本同时使用全部有效专家，并在每个候选 token 上自适应融合”。

建议新增独立的 MP-OPD 路径，避免继续扩展旧的 `multi_teacher_distill` 条件分支。

## 15. 预计代码边界

### 15.1 数据层

[`rl_dataset.py`](P-OPD/verl/verl/utils/dataset/rl_dataset.py) 需要读取并保留：

- `expert_contexts`；
- 固定专家顺序；
- `expert_mask`；
- 可选专家模板参数。

### 15.2 Prompt 与对齐层

建议新增独立模块，例如：

```text
verl/trainer/ppo/expert_prompt_utils.py
```

职责包括：

- 构造纯净与专用 expert prompts；
- 展平和恢复专家 batch；
- 维护 response offset；
- 校验 tokenizer 和候选 token ID；
- 生成专家 mask。

### 15.3 Worker 层

[`fsdp_workers.py`](P-OPD/verl/verl/workers/fsdp_workers.py) 需要新增 MP-OPD 概率准备 RPC，返回：

- student top-k IDs；
- student 当前 top-k log-prob；
- student anchor top-k log-prob；
- teacher base top-k log-prob；
- specialized experts top-k log-prob；
- timing metrics。

### 15.4 Trainer 层

[`ray_trainer.py`](P-OPD/verl/verl/trainer/ppo/ray_trainer.py) 需要：

- 识别 `train_mode=multi_prompt_distill`；
- 在 rollout 后构建 specialized expert batch；
- 调用 MP-OPD 概率准备；
- 传递 `[B,T,K,E]` 概率和 mask；
- 记录专家权重、冲突度和 top-k 指标；
- 保证验证路径不注入专家方向说明或可选 evidence。

### 15.5 Actor 与 loss 层

[`dp_actor.py`](P-OPD/verl/verl/workers/actor/dp_actor.py) 应新增独立的 `update_policy_mpopd()`，负责：

- 计算专家相对增量；
- 专家维绝对值 softmax；
- 有符号融合；
- 构造 anchored target distribution；
- 计算 top-k KL loss；
- 记录数值稳定性指标。

KL 的纯张量逻辑建议拆成可单元测试的函数，放入 [`core_algos.py`](P-OPD/verl/verl/trainer/ppo/core_algos.py) 或独立 loss 模块。

### 15.6 配置与脚本

需要增加的核心配置包括：

```yaml
algorithm:
  train_mode: multi_prompt_distill
  mp_opd:
    expert_names: [chasing, long_term, repurchase, generalized]
    expert_instructions:
      chasing: "加强关注近期高强度兴趣及其延续性"
      long_term: "加强关注跨时间窗口稳定出现的长期偏好"
      repurchase: "加强关注重复购买、消耗周期和历史复购间隔"
      generalized: "加强关注相邻品类和可迁移的泛化兴趣"
    top_k: 10
    expert_temperature: 1.0
    token_temperature: 1.0
    lambda_value: 1.0
    expert_priors: null
    expert_forward_micro_batch_size: 0
    skip_samples_without_active_experts: true
```

推荐新增独立启动脚本，而不是修改数学或代码脚本的默认行为：

```text
scripts/mpopd_recommendation.sh
```

## 16. 专家可用性与异常处理

### 16.1 部分专家不可用

`enabled=false` 或 prompt 构造失败的专家，在专家维 softmax 前 mask 掉，不参与分母。配置缺少必要方向说明属于启动配置错误，应直接 fail fast；缺少可选 evidence 本身不会使专家失效。至少一个专家有效时正常计算。

### 16.2 所有专家不可用

默认跳过该样本的 MP-OPD loss，并记录 `mpopd/all_experts_unavailable_count`。不应静默退化成普通 anchor 蒸馏，因为这会改变训练数据含义。

### 16.3 Prompt 超长

不能任意截断方向说明或 evidence 而保留不完整语义。Prompt builder 应使用可配置策略：

- 对用户历史、方向说明和可选 evidence 分别设置预算；
- 保证 response prefix 不被截断；
- 超过预算仍无法构造时跳过该专家；
- 所有专家均失败时按“全部不可用”处理。

### 16.4 数值异常

实现需要：

- 使用稳定的 `log_softmax`；
- softmax 前减最大值；
- 将无效专家设为适合 dtype 的负无穷；
- 断言 $\tau_E>0$、$\tau_T>0$、$\lambda>0$；
- 检查所有目标概率有限且和为 1；
- 出现 NaN/Inf 时跳过 micro-batch 并记录明确错误指标，不能继续传播梯度。

## 17. 训练指标

至少记录以下指标：

### 17.1 Loss 与分布

- `mpopd/kl_loss`
- `mpopd/target_entropy`
- `mpopd/student_topk_entropy`
- `mpopd/topk_probability_mass`
- `mpopd/target_max_probability`

### 17.2 专家信号

- 每个专家的 `delta_mean`、`delta_abs_mean`；
- 每个专家的平均 softmax 权重；
- 专家权重熵；
- 最大专家权重；
- 正负信号比例；
- 专家冲突率；
- 有效专家数量分布。

### 17.3 性能

- clean-base forward 时间；
- specialized expert forward 总时间；
- 每个专家或 chunk 的 forward 时间；
- 融合计算时间；
- MP-OPD update 时间；
- GPU 峰值显存。

专家冲突可以初步定义为：同一个候选 token 上同时存在有效的正 $\Delta$ 和负 $\Delta$。

## 18. 测试策略

### 18.1 数学单元测试

必须覆盖：

1. 第一层 softmax 确实使用 $|\Delta|$；
2. 加权求和使用原始有符号 $\Delta$；
3. 专家 mask 不影响有效专家归一化；
4. $g=0$ 时目标分布等于 anchor 的 top-k 归一化分布；
5. 正增量提高对应 token 的目标概率；
6. 负增量降低对应 token 的目标概率；
7. 目标分布每行和为 1；
8. KL 在 student 分布等于目标分布时为 0；
9. target tensor 不带梯度；
10. padding token 不贡献 loss。

### 18.2 数据与 Prompt 测试

必须覆盖：

- 固定专家顺序；
- 禁用专家、缺少 evidence 与已观测空 evidence 的区别；
- 专家方向说明和 evidence 未进入三个纯净 prompt 分支；
- 不同长度专家 prompt 的 response offset 正确；
- 展平 `[B,E]` 与恢复 `[B,T,K,E]` 可逆；
- 超长 prompt 的处理符合配置。

### 18.3 Worker 集成测试

使用小模型和短序列验证：

- 所有概率张量形状正确；
- 所有专家使用相同 top-k IDs；
- 专用 prompt 与纯净 prompt 相同时 $\Delta\approx0$；
- 改变一个专家的方向说明或 evidence 只改变该专家切片；
- 冻结模型不产生梯度。

### 18.4 端到端 smoke test

构造极小推荐数据集，运行 1～2 个训练 step，验证：

- rollout 成功；
- MP-OPD loss 有限；
- student 参数发生变化；
- expert/base 参数不变；
- checkpoint 能保存和恢复；
- 不使用 expert context 的验证可以独立运行。

## 19. 第一版验收标准

MP-OPD 第一版完成需要同时满足：

1. 一个冻结 teacher base 可处理任意配置数量的 prompt-conditioned experts；
2. Student 每个回答位置能生成 top-k 候选；
3. 所有专家能在相同候选 token 上返回概率；
4. 得到形状正确的 `[B,T,K,E]` 专家概率；
5. 按绝对变化计算专家 softmax 权重；
6. 按实际有符号变化完成融合；
7. 生成 anchored top-k target distribution；
8. 通过 KL loss 只更新 student；
9. Student、student base、teacher base、验证和推理不存在专家 prompt 或未来信息泄漏；
10. 单元测试、集成测试和最小训练 smoke test 通过；
11. 关键数学、专家行为和性能指标可观测；
12. `top_k=1`、单专家和零增量等边界情况有明确行为。

## 20. 第一版非目标

以下内容不进入第一版：

- 训练独立专家 checkpoint；
- 可学习 gating/router 网络；
- 专家分别生成回答后再合并；
- 跨 tokenizer 的 top-k token 映射；
- 完整词表级多专家蒸馏；
- 在线推理阶段访问未来点击；
- 自动搜索专家 prompt；
- 同时重构现有 GRPO/P-OPD 全部训练路径。

## 21. 后续可扩展方向

第一版稳定后可以研究：

- learned router 替代参数无关的绝对增量 softmax；
- 用户级或场景级专家先验 $\rho_e$；
- 专家权重的熵正则与防塌缩约束；
- 将 top-k 外概率建模为 residual bucket；
- 根据 top-k 覆盖质量动态调整 $K$；
- 跨 tokenizer 的字符串或 token span 对齐；
- 将多个未来时间窗口作为额外专家；
- 结合 GRPO 或推荐业务 reward 的混合训练；
- 对专家冲突进行 Pareto 或约束优化，而不仅是有符号抵消。

## 22. 设计总结

MP-OPD 将“专家”定义为共享冻结 `teacher_base` 权重、但接受不同方向专用 prompt 的条件策略；ground truth 是可选训练期证据，而不是专家身份本身。每次训练共有一个纯净 prompt 和四个专用 prompt。Student 只根据纯净 prompt 生成一次回答；student base 与 teacher base 也使用该纯净 prompt，四个专家则对同一回答在 student top-k 候选空间中进行 teacher-forcing 评分。

每个候选 token 先沿专家维根据相对变化的绝对值计算影响力，再用真实有符号变化融合；随后将融合增量叠加到 student-base 分布上，沿 top-k token 维构造 anchored target distribution，并通过 KL 蒸馏更新 student。

这保留了 P-OPD“迁移相对变化、使用冻结 anchor”的核心思想，同时将单代理更新扩展为一次回答上的多方向、逐 token、逐候选词表自适应融合。
