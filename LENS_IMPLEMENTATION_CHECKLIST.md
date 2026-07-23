# LENS 改进方案 - 实施检查清单与代码对接细节

## Part I: 新模块创建清单

### Module 1: BatchRankingEvaluator
**文件**: `src/sirchmunk/learnings/batch_ranking_evaluator.py`
**大小**: ~320 行

**依赖**:
- `sirchmunk.llm.openai_chat.OpenAIChat`
- `sirchmunk.utils.create_logger`
- `dataclasses.dataclass`

**接口**:
```python
class BatchRankingEvaluator:
    async def rank_batch(
        samples: List[SampleWindow],
        query: str,
        keywords: Optional[List[str]] = None,
    ) -> List[SampleWindow]
    
    async def rank_by_chunks(
        samples: List[SampleWindow],
        query: str,
        chunk_size: int = 5,
    ) -> List[SampleWindow]
```

**与 evidence_processor.py 的对接**:
- 替代行 375-389 的 `_evaluate_batch` 方法
- 调用点: `get_roi()` 第 522 行 (原 Round 1 评估)

**Prompt 设计** (新增到 prompts.py):
```
LISTWISE_RANKING_PROMPT = """
### Task: Rank documents by relevance (Listwise Ranking)

Query: {query}

Documents:
{doc_snippets}

Rank from most to least relevant.
Return JSON: {{"ranking": [1,2,...], "reasons": [...]}}
"""
```

---

### Module 2: MultiArmNavigator
**文件**: `src/sirchmunk/learnings/multi_arm_navigator.py`
**大小**: ~520 行

**新数据类**:
```python
@dataclass
class Arm:
    arm_id: str
    observations: List[float] = field(default_factory=list)
    center: int = 0  # Document position
    
    def posterior_mean(self) -> float
    def posterior_variance(self) -> float
    def confidence_interval(self, level: float = 0.95) -> Tuple[float, float]
```

**依赖**:
- `scipy.stats.t` (t-interval)
- `numpy` (基础计算)

**与 evidence_processor.py 的对接**:
- 替代行 102: `self.top_k_seeds = 2` 改为 `self.navigator = MultiArmNavigator(...)`
- 替代行 241-294: `_sample_gaussian` 方法调用改为 `navigator.allocate_and_explore()`
- 修改 `get_roi()` 主循环结构 (第 470-540 行)

**新的主循环流程** (伪代码):
```python
# Initialize arms (Round 1)
initial_samples = await navigator.initialize_arms(self, query)
evaluated = await batch_ranker.rank_batch(initial_samples, query)

# Multi-round exploration
for r in range(2, max_rounds + 1):
    samples = await navigator.allocate_and_explore(
        sampler=self, query=query, budget=5
    )
    evaluated = await batch_ranker.rank_batch(samples, query)
    all_candidates.extend(evaluated)
    # Update arms with new scores...
```

---

### Module 3: ReasoningChainExploiter
**文件**: `src/sirchmunk/learnings/reasoning_chain_exploiter.py`
**大小**: ~280 行

**核心方法**:
```python
class ReasoningChainExploiter:
    def extract_signals(
        reasoning: str,
        score: float,
        query: str,
    ) -> Dict[str, Any]
    
    async def generate_targeted_samples(
        signals: Dict[str, Any],
        sampler: MonteCarloEvidenceSampling,
        query: str,
        num_samples: int = 3,
    ) -> List[SampleWindow]
```

**与 evidence_processor.py 的对接**:
- 在 `get_roi()` 第 522-528 行 (评估后处理) 后插入:
```python
for sample in evaluated:
    signals = reasoning_exploiter.extract_signals(
        sample.reasoning, sample.score, query
    )
    if signals:
        targeted = await reasoning_exploiter.generate_targeted_samples(
            signals, self, query
        )
        current_samples.extend(targeted)
```

**信号模式** (在 __init__ 中编译):
```python
SIGNAL_PATTERNS = {
    "year": r"(19|20)\d{2}",
    "segment": r"(segment|division|subsidiary|unit)",
    "table_type": r"(Income Statement|Balance Sheet|Cash Flow)",
    "missing": r"(not found|missing|unavailable|不存在)",
}
```

---

### Module 4: AdaptiveProposalMixer
**文件**: `src/sirchmunk/learnings/adaptive_proposal_mixer.py`
**大小**: ~200 行

**核心方法**:
```python
class AdaptiveProposalMixer:
    def compute_lambda(
        round_num: int,
        posterior_entropy: float,
        token_budget_remaining: float,
    ) -> float
    
    async def propose(
        sampler: MonteCarloEvidenceSampling,
        query: str,
        round_num: int,
        num_samples: int = 5,
        lambda_t: float = 0.5,
    ) -> List[SampleWindow]
```

**与 evidence_processor.py 的对接**:
- 在 `get_roi()` 主循环 (第 475-517 行) 替换采样决策逻辑:
```python
# OLD (第 475-517 行):
if r == 1:
    fuzz_samples = await self._get_fuzzy_anchors(...)
    current_samples.extend(fuzz_samples)
    random_samples = self._sample_stratified_supplement(...)
    current_samples.extend(random_samples)
else:
    current_samples = self._sample_gaussian(valid_seeds, r)

# NEW:
lambda_t = self.proposal_mixer.compute_lambda(
    r, posterior_entropy=..., token_budget_remaining=...
)
current_samples = await self.proposal_mixer.propose(
    self, query, r, num_samples=5, lambda_t=lambda_t
)
```

**lambda 计算公式**:
```python
def compute_lambda(self, round_num, posterior_entropy, token_budget):
    base = max(0.15, 0.8 ** round_num)
    entropy_factor = 0.5 + 0.5 * sigmoid(posterior_entropy - 2.0)
    budget_factor = token_budget / 10000
    return base * entropy_factor * budget_factor
```

---

### Module 5: StatisticalStopDecider
**文件**: `src/sirchmunk/learnings/statistical_stop_decider.py`
**大小**: ~350 行

**数据类**:
```python
@dataclass
class ConfidenceEstimate:
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    
    def is_confident(self, threshold: float = 8.0) -> bool:
        return self.ci_lower >= threshold
```

**核心方法**:
```python
async def should_stop(
    all_candidates: List[SampleWindow],
    query: str,
    round_num: int,
    max_rounds: int = 3,
    tokens_remaining: float = float('inf'),
    multi_source_intent: float = 0.0,
) -> Tuple[bool, str]
```

**与 evidence_processor.py 的对接**:
- 替代行 534-540 的硬阈值检查:
```python
# OLD:
if top_seeds and top_seeds[0].score >= confidence_threshold:
    if self.verbose:
        await self._log.info(...)
    break

# NEW:
should_stop, reason = await self.stop_decider.should_stop(
    all_candidates, query, r, self.max_rounds,
    tokens_remaining=tokens_remaining,
    multi_source_intent=multi_source_intent,
)
if should_stop:
    await self._log.info(f"Stopping: {reason}")
    break
```

---

## Part II: 修改现有文件

### evidence_processor.py 修改清单

**第 1-14 行** (import 部分):
```diff
+ from sirchmunk.learnings.batch_ranking_evaluator import BatchRankingEvaluator
+ from sirchmunk.learnings.multi_arm_navigator import MultiArmNavigator, Arm
+ from sirchmunk.learnings.reasoning_chain_exploiter import ReasoningChainExploiter
+ from sirchmunk.learnings.adaptive_proposal_mixer import AdaptiveProposalMixer
+ from sirchmunk.learnings.statistical_stop_decider import StatisticalStopDecider
```

**第 71-108 行** (__init__ 方法):
```diff
def __init__(self, ...):
    # ... 现有代码 ...
+   self.batch_ranker = BatchRankingEvaluator(llm=llm, log_callback=log_callback)
+   self.navigator = MultiArmNavigator(llm=llm, k_arms=4, log_callback=log_callback)
+   self.reasoning_exploiter = ReasoningChainExploiter(log_callback=log_callback)
+   self.proposal_mixer = AdaptiveProposalMixer()
+   self.stop_decider = StatisticalStopDecider()
```

**第 119-192 行** (_get_fuzzy_anchors 方法):
```diff
- results = process.extract(
-     query=f"{query} {' '.join(keywords)}".strip(),
-     choices=list(chunk_texts),
-     scorer=fuzz.token_set_ratio,
-     limit=int(self.fuzz_candidates_num * 2),
-     score_cutoff=None,
- )

+ # Use HybridFuzzyMatcher instead
+ matcher = HybridFuzzyMatcher(use_embedding=False)  # TODO: init in __init__
+ results = []
+ for i, chunk in enumerate(chunk_texts):
+     score = matcher.score_hybrid(f"{query} {' '.join(keywords)}", chunk)
+     if score >= threshold:
+         results.append((chunk, score, i))
+ results.sort(key=lambda x: x[1], reverse=True)
+ results = results[:int(self.fuzz_candidates_num * 2)]
```

**第 296-326 行** (_evaluate_sample_async 方法):
```diff
# Keep method but mark as DEPRECATED
# async def _evaluate_sample_async(self, sample, query):
#     """DEPRECATED: Use BatchRankingEvaluator.rank_batch() instead."""
```

**第 375-389 行** (_evaluate_batch 方法):
```diff
async def _evaluate_batch(
    self, samples: List[SampleWindow], query: str
) -> List[SampleWindow]:
-   if self.verbose:
-       await self._log.info(f"   Evaluating {len(samples)} samples with LLM...")
-   tasks = [self._evaluate_sample_async(s, query) for s in samples]
-   evaluated_samples = await asyncio.gather(*tasks)
-   return list(evaluated_samples)

+   if self.verbose:
+       await self._log.info(f"   Batch ranking {len(samples)} samples...")
+   return await self.batch_ranker.rank_batch(samples, query)
```

**第 421-601 行** (get_roi 方法):
🔴 **主要重构** - 详见下文"get_roi 重构指南"

---

### prompts.py 修改清单

**添加新 prompt** (在文件末尾):

```python
LISTWISE_RANKING = """### Task: Rank documents by relevance to the query.

Query: {query}

Documents:
{doc_snippets}

### Instructions:
1. Rank these documents from MOST to LEAST relevant.
2. Consider direct answer, specificity, precision.
3. Return JSON:

{{"ranking": [1, 2, ...], "reasons": ["Reason 1", "Reason 2", ...]}}
"""
```

---

### knowledge_base.py 修改清单

**第 115-200 行** (_extract_evidence_for_file 方法):

```diff
# 无需修改主体逻辑，但需要传递 multi_source_intent:

+ # NEW: Extract multi-source intent from keywords
+ multi_source_intent = keywords.get("_multi_source_intent", 0.0)

  sampler = MonteCarloEvidenceSampling(...)
  roi_result: RoiResult = await sampler.get_roi(
      query=query,
      keywords=keywords,
      confidence_threshold=confidence_threshold,
      top_k=top_k_snippets,
+     multi_source_intent=multi_source_intent,  # NEW
  )
```

---

## Part III: get_roi 重构指南

**原始流程** (第 421-601 行):
```
Round Loop (1 to max_rounds):
  └─ if r==1: fuzz + random
  └─ else: gaussian focusing
  └─ evaluate_batch (逐个 LLM)
  └─ top-2 selection
  └─ hard threshold check
```

**新流程**:
```
Initialize (Round 1):
  └─ multi_arm_nav.initialize_arms()
  └─ batch_ranker.rank_batch()

Exploration Loop (Round 2 to max_rounds):
  ├─ compute λ(t) via proposal_mixer
  ├─ propose_mix(λ, exploration vs exploitation)
  ├─ extract_signals + generate_targeted()
  ├─ batch_ranker.rank_batch()
  ├─ navigator.update_arms()
  └─ stop_decider.should_stop()
      └─ if true: break

Final:
  └─ generate_summary()
```

**关键变量**:
```python
# NEW in get_roi
multi_source_intent = 0.0  # From query keywords analysis
posterior_entropy = 0.0    # Computed from arm variances
tokens_used = 0            # Track cumulative tokens
tokens_remaining = 20000   # Budget (GPT-4: ~$0.30 per query)
```

**伪代码框架**:

```python
async def get_roi(self, query, keywords, ...):
    # ... initial validation ...
    
    # NEW: Extract query metadata
    multi_source_intent = kwargs.get("multi_source_intent", 0.0)
    
    all_candidates: List[SampleWindow] = []
    
    # === NEW: Round 1 with Arms ===
    if self.verbose:
        await self._log.info(f"Initializing {self.navigator.k_arms} arms...")
    
    initial_samples = await self.navigator.initialize_arms(self, query)
    
    if self.verbose:
        await self._log.info(f"   Batch ranking {len(initial_samples)} initial samples...")
    
    evaluated = await self.batch_ranker.rank_batch(initial_samples, query)
    all_candidates.extend(evaluated)
    
    # Update arms with initial observations
    for i, sample in enumerate(evaluated):
        if i < self.navigator.k_arms:
            self.navigator.arms[i].add_observation(sample.score)
    
    # === Exploration Loops (Rounds 2-3) ===
    for r in range(2, self.max_rounds + 1):
        if self.verbose:
            await self._log.info(f"--- Round {r}/{self.max_rounds} ---")
        
        current_samples = []
        
        # Compute adaptive proposal mixture
        posterior_entropy = self.navigator._entropy()
        lambda_t = self.proposal_mixer.compute_lambda(
            r, posterior_entropy=posterior_entropy,
            token_budget_remaining=token_budget
        )
        
        if self.verbose:
            await self._log.info(f"   λ(t) = {lambda_t:.3f} (exploration ratio)")
        
        # Propose samples with mixture
        proposed = await self.proposal_mixer.propose(
            self, query, r, num_samples=5, lambda_t=lambda_t
        )
        current_samples.extend(proposed)
        
        # Extract signals from high-confidence samples and target exploration
        for sample in all_candidates[-5:]:
            if sample.score >= 4.0 and sample.score < 8.0:
                signals = self.reasoning_exploiter.extract_signals(
                    sample.reasoning, sample.score, query
                )
                if signals:
                    targeted = await self.reasoning_exploiter.generate_targeted_samples(
                        signals, self, query, num_samples=2
                    )
                    current_samples.extend(targeted)
        
        # Batch ranking
        if current_samples:
            if self.verbose:
                await self._log.info(f"   Batch ranking {len(current_samples)} samples...")
            evaluated = await self.batch_ranker.rank_batch(current_samples, query)
            all_candidates.extend(evaluated)
            
            # Log results
            for s in evaluated:
                await self._log.info(
                    f"  [Pos {s.start_idx:6d}] Score: {s.score:.1f} | {s.reasoning[:40]}..."
                )
            
            # Update navigator arms
            for sample in evaluated:
                # Find closest arm center and update
                closest_arm_idx = self._find_closest_arm(sample)
                self.navigator.arms[closest_arm_idx].add_observation(sample.score)
        
        # Statistical stop criterion
        should_stop, reason = await self.stop_decider.should_stop(
            all_candidates, query, r, self.max_rounds,
            tokens_remaining=token_budget,
            multi_source_intent=multi_source_intent,
        )
        
        if should_stop:
            if self.verbose:
                await self._log.info(f"Early stopping: {reason}")
            break
    
    # === Final Processing ===
    if not all_candidates:
        return RoiResult(
            summary="Could not retrieve relevant content.",
            is_found=False,
            snippets=[],
        )
    
    relevant_candidates = [c for c in all_candidates if c.score >= 4.0]
    if not relevant_candidates:
        best = all_candidates[0]
        return RoiResult(
            summary="No exact answer found.",
            is_found=False,
            snippets=[{...}],
        )
    
    final_candidates = relevant_candidates[:self.top_k_seeds]
    summary = await self._generate_summary(final_candidates, query)
    
    roi_snippets = [
        {
            "snippet": c.content,
            "start": c.start_idx,
            "end": c.end_idx,
            "score": c.score,
            "reasoning": c.reasoning,
        }
        for c in final_candidates
    ]
    
    return RoiResult(
        summary=summary,
        is_found=True,
        snippets=roi_snippets,
    )
```

---

## Part IV: 测试计划

### Unit Tests

**test_batch_ranking_evaluator.py**:
```python
@pytest.mark.asyncio
async def test_rank_batch_order():
    """Verify ranking is consistent and valid."""
    samples = [
        SampleWindow(..., content="exact match", score=0),
        SampleWindow(..., content="partial match", score=0),
        SampleWindow(..., content="no match", score=0),
    ]
    ranker = BatchRankingEvaluator(llm=mock_llm)
    ranked = await ranker.rank_batch(samples, query="exact match test")
    assert ranked[0].score > ranked[1].score > ranked[2].score
```

**test_multi_arm_navigator.py**:
```python
def test_ids_allocation():
    """Verify allocation favors high-uncertainty arms."""
    nav = MultiArmNavigator(k_arms=4)
    nav.arms[0].add_observation(9.0)  # Confident
    nav.arms[1].add_observation(5.0)
    nav.arms[1].add_observation(6.0)  # High variance
    
    allocation = nav._ids_allocation(budget=10)
    assert allocation[1] > allocation[0]  # More samples to uncertain arm
```

### Integration Tests

**test_get_roi_integration.py**:
```python
@pytest.mark.asyncio
async def test_get_roi_with_new_components():
    """End-to-end test of improved get_roi."""
    doc = "文本内容 2021年资本支出为500万... 2022年为600万..."
    mces = MonteCarloEvidenceSampling(llm=openai_chat, doc_content=doc)
    
    result = await mces.get_roi(
        query="公司多年资本支出",
        keywords={"capital expenditure": 8.0},
        multi_source_intent=0.7,  # Multi-year query
    )
    
    # Verify multi-hop discovery
    assert len(result.snippets) >= 2
    assert any("2021" in s["reasoning"] for s in result.snippets)
    assert any("2022" in s["reasoning"] for s in result.snippets)
```

---

## Part V: 部署检查

- [ ] 所有 5 个新模块创建并通过 linting
- [ ] evidence_processor.py 导入正确无循环依赖
- [ ] prompts.py 新 prompt 格式验证
- [ ] Unit tests 全部通过
- [ ] Integration tests 全部通过
- [ ] FinanceBench mini (10-20 queries) 基准测试
- [ ] 代码审查 (focus: LLM prompt 合理性、Bayesian 计算正确性)
- [ ] 文档更新 (API reference + migration guide)

