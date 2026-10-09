# Real LLM Paired Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real-model V1/V2 paired-evaluation closed loop on the frozen 100 Cases: paired runs → deterministic scoring → E1–E10 attribution → cluster-aware statistics, where the system prompt is the sole independent variable.

**Architecture:** A new paired experiment runs alongside (without modifying) the Phase 3 Deterministic Baseline. Both versions use the same Volcano Ark model, parameters, tools, data, Cases, Runner, max-steps and timeout; only the versioned system prompt differs. The run process sees only `RunnableCase.user_input`; full Evaluation Cases (Gold) are loaded exclusively by the offline scoring/analysis processes. Artifacts are versioned (`artifact_schema_version="2.0.0"`) and published via Phase-3-style temp-dir → verify → file-lock → atomic-rename.

**Tech Stack:** Python 3.11+, Pydantic v2, NumPy, PyYAML, filelock, stdlib `urllib` (OpenAI-compatible transport), `pytest`, `ruff`.

**Design spec:** `docs/superpowers/specs/2026-10-09-real-llm-paired-evaluation-design.md` (authoritative when this plan and the spec ever appear to diverge).

---

## File Structure

New / modified units and their single responsibility:

```text
app/prompts/
  __init__.py            # re-export v1/v2 snapshots
  base.py                # PromptSnapshot model, SHA-256, builder
  v1.py                  # frozen Real LLM V1 prompt text + snapshot
  v2.py                  # frozen Real LLM V2 prompt text + snapshot
app/experiments/
  llm_settings.py        # ProviderSettings from env + strict preflight
  llm_artifacts.py       # Artifact v2 models, canonical bytes, verify, stage/commit writer
  pilot_cases.py         # frozen 20 pilot Case IDs + selection
  llm_run.py             # PairedRunCoordinator (pilot/formal), per-case isolated execution
  comparisons.py         # comparison manifest + pair fairness validation
  llm_cli.py             # real-llm-pilot / real-llm-formal argparse entry points
app/evaluation/
  __init__.py
  freeze.py              # FrozenConfig + versioned SHA-256 freeze list
  models.py              # full Gold-bearing EvaluationCase + EvaluationCaseBundle
  loader.py              # integrity-checked full loader (proves runnable projection)
  normalize.py           # date/product/metric/number normalization helpers
  scorer.py              # deterministic 10-dimension scorer + Task Success gate
  evaluation_artifacts.py # evaluation manifest/case_scores/summary writer + reader
  cli.py                 # offline scoring entry point
app/error_analysis/
  __init__.py
  taxonomy.py            # E1–E10 labels + earliest-causal attribution rules
  rca.py                 # RCA record builder
  artifacts.py           # RCA jsonl + Error Shift matrix writer
  cli.py                 # offline error-analysis entry point
app/statistics/
  __init__.py
  config.py              # frozen statistics config (seed/iterations/CI/min clusters)
  pairing.py             # pairing + fairness gate, cluster index, technical-failure rate
  bootstrap.py           # cluster bootstrap (resample whole clusters)
  mcnemar.py             # cluster all-pass aggregation + Exact McNemar
  paired.py              # descriptive paired metrics, latency/token aggregation
  cli.py                 # statistics aggregation entry point + artifact
configs/evaluation/
  statistics_v1.yaml     # frozen statistics parameters
  rubric_v1.yaml         # frozen rubric: metric aliases, task core facts, gates
  error_taxonomy_v1.yaml # frozen E1–E10 definitions and attribution rules
```

Modification points (no drive-by changes):

```text
app/llm/openai_compatible.py  # add immutable optional system_prompt; first turn system+user
pyproject.toml                # add console scripts: real-llm-pilot, real-llm-formal,
                              #   evaluate-runs, analyze-errors, aggregate-statistics
Makefile                      # add targets: real-llm-pilot, real-llm-formal
.env.example                  # document LLM_* vars (already present; verify only)
```

Hard rules for every task:

- TDD: write failing test → run and confirm the failure → implement → run and confirm pass → commit.
- Never print, log, snapshot, fixture or persist the API key, Authorization header, provider error body, Gold, expected answers or Case labels inside Run artifacts.
- Do not modify Phase 3 models (`app/experiments/artifacts.py`) or its frozen-asset tests. After each task run the full suite once and keep it green.
- Commits are scoped to the files of the current task only. `书籍/` is never staged.

Convenience command used throughout (run from repo root):

```bash
PYTHON=python3
```

---

# Batch A — Real-model run infrastructure and Pilot

Gate: Pilot technical gate passes (per-version technical failure ≤ 10%); V1/V2 prompts may be revised on Development evidence; no formal effect is published.

---

### Task 1: Prompt package scaffold and PromptSnapshot

**Files:**
- Create: `app/prompts/__init__.py`
- Create: `app/prompts/base.py`
- Test: `tests/test_prompts_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts_base.py
import pytest

from app.prompts.base import (
    PromptSnapshot,
    compute_prompt_sha256,
    make_prompt_snapshot,
)


def test_compute_prompt_sha256_is_utf8_sha256():
    assert (
        compute_prompt_sha256("你好")
        == "670d9743542cae3ea7ebe36af56bd53648b0a1126162e78d81a32934a711302e"
    )


def test_make_prompt_snapshot_hashes_text_and_is_frozen():
    snapshot = make_prompt_snapshot(
        name="real-llm-v1",
        version="1.0.0",
        text="系统提示文本",
        change_note="initial",
        source_file="app/prompts/v1.py",
    )
    assert isinstance(snapshot, PromptSnapshot)
    assert snapshot.prompt_sha256 == compute_prompt_sha256("系统提示文本")
    with pytest.raises(Exception):
        snapshot.prompt_text = "other"


def test_prompt_snapshot_rejects_blank_identity():
    with pytest.raises(Exception):
        make_prompt_snapshot(
            name=" ",
            version="1.0.0",
            text="x",
            change_note="n",
            source_file="app/prompts/v1.py",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prompts_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.prompts'`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/prompts/base.py
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field


class PromptModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )


def compute_prompt_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class PromptSnapshot(PromptModel):
    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    change_note: str = Field(min_length=1)
    source_file: str = Field(min_length=1)


def make_prompt_snapshot(
    *,
    name: str,
    version: str,
    text: str,
    change_note: str,
    source_file: str,
) -> PromptSnapshot:
    return PromptSnapshot(
        prompt_name=name,
        prompt_version=version,
        prompt_text=text,
        prompt_sha256=compute_prompt_sha256(text),
        change_note=change_note,
        source_file=source_file,
    )
```

```python
# app/prompts/__init__.py
"""Frozen real-model system prompts."""

__all__: list[str] = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prompts_base.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/prompts/__init__.py app/prompts/base.py tests/test_prompts_base.py
git commit -m "feat(prompts): add frozen prompt snapshot model"
```

---

### Task 2: Real LLM V1 prompt

**Files:**
- Create: `app/prompts/v1.py`
- Modify: `app/prompts/__init__.py`
- Test: `tests/test_prompts_v1.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts_v1.py
from app.prompts.base import compute_prompt_sha256
from app.prompts.v1 import (
    V1_PROMPT_NAME,
    V1_PROMPT_VERSION,
    v1_prompt_snapshot,
)

FORBIDDEN = (
    "CASE_",
    "gold",
    "Gold",
    "expected",
    "score",
    "evidence_id",
    "E1",
    "tolerance",
    "容差",
)


def test_v1_identity_and_digest():
    snapshot = v1_prompt_snapshot()
    assert snapshot.prompt_name == V1_PROMPT_NAME == "real-llm-v1"
    assert snapshot.prompt_version == V1_PROMPT_VERSION
    assert snapshot.prompt_sha256 == compute_prompt_sha256(snapshot.prompt_text)
    assert snapshot.source_file == "app/prompts/v1.py"


def test_v1_prompt_has_no_case_or_gold_leaks():
    text = v1_prompt_snapshot().prompt_text
    for token in FORBIDDEN:
        assert token not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prompts_v1.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.prompts.v1'`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/prompts/v1.py
from app.prompts.base import PromptSnapshot, make_prompt_snapshot

V1_PROMPT_NAME = "real-llm-v1"
V1_PROMPT_VERSION = "1.0.0"
V1_CHANGE_NOTE = "Initial basic business-analysis agent prompt."

V1_PROMPT_TEXT = """你是一名电商经营分析助手，基于 Synthetic E-commerce Data 回答经营分析问题。

可用工具：
- 你可以调用工具查询商品、销售、流量、投放数据，并基于已取得的工具结果计算指标。
- 需要数据时优先调用工具获取事实，不要编造工具结果。

工作方式：
- 需要数据时调用工具；取得足够事实后直接给出回答。
- 回答应包含结论和支撑结论的关键数据。
- 如果问题无法由工具数据支持，请明确说明这一限制。
"""


def v1_prompt_snapshot() -> PromptSnapshot:
    return make_prompt_snapshot(
        name=V1_PROMPT_NAME,
        version=V1_PROMPT_VERSION,
        text=V1_PROMPT_TEXT,
        change_note=V1_CHANGE_NOTE,
        source_file="app/prompts/v1.py",
    )
```

Update `app/prompts/__init__.py`:

```python
"""Frozen real-model system prompts."""

from app.prompts.v1 import V1_PROMPT_NAME, V1_PROMPT_VERSION, v1_prompt_snapshot

__all__ = [
    "V1_PROMPT_NAME",
    "V1_PROMPT_VERSION",
    "v1_prompt_snapshot",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prompts_v1.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/prompts/v1.py app/prompts/__init__.py tests/test_prompts_v1.py
git commit -m "feat(prompts): add real LLM V1 system prompt"
```

---

### Task 3: Real LLM V2 prompt

**Files:**
- Create: `app/prompts/v2.py`
- Modify: `app/prompts/__init__.py`
- Test: `tests/test_prompts_v2.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts_v2.py
from app.prompts.base import compute_prompt_sha256
from app.prompts.v2 import (
    V2_PROMPT_NAME,
    V2_PROMPT_VERSION,
    v2_prompt_snapshot,
)

FORBIDDEN = (
    "CASE_",
    "gold",
    "Gold",
    "expected",
    "score",
    "evidence_id",
    "E1",
    "tolerance",
    "容差",
)


def test_v2_identity_and_digest():
    snapshot = v2_prompt_snapshot()
    assert snapshot.prompt_name == V2_PROMPT_NAME == "real-llm-v2"
    assert snapshot.prompt_version == V2_PROMPT_VERSION
    assert snapshot.prompt_sha256 == compute_prompt_sha256(snapshot.prompt_text)
    assert snapshot.source_file == "app/prompts/v2.py"


def test_v2_prompt_contains_interventions_and_no_leaks():
    text = v2_prompt_snapshot().prompt_text
    for required in ("result_id", "结论", "关键证据", "建议或限制", "数据不足"):
        assert required in text
    for token in FORBIDDEN:
        assert token not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prompts_v2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.prompts.v2'`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/prompts/v2.py
from app.prompts.base import PromptSnapshot, make_prompt_snapshot

V2_PROMPT_NAME = "real-llm-v2"
V2_PROMPT_VERSION = "1.0.0"
V2_CHANGE_NOTE = "Initial prompt with task classification, tool boundaries, checks and evidence binding."

V2_PROMPT_TEXT = """你是一名电商经营分析助手，基于 Synthetic E-commerce Data 回答经营分析问题。

在回答前，请按以下步骤在内部完成，不需要输出这些内部步骤：
1. 判断问题属于五类经营任务中的哪一类：GMV 诊断、商品异常、转化下降、关注商品、下周重点。
2. 根据任务只选择必要的工具，避免无关调用。
3. 调用工具前检查日期窗口、对照窗口、商品范围，以及退款、缺失值相关参数。
4. 计算指标前，确认所需的来源工具结果已经齐全。
5. 每条关键结论都必须绑定到本次工具执行真实返回的 result_id。
6. 当数据不足时，说明缺失了什么、会影响什么，以及当前数据能支持的最窄结论，不要强行下确定结论。

工具使用要求：
- 需要事实时调用工具，不要编造工具结果或 result_id。
- 每次只做一个工具调用；取得对应结果后再继续下一步。

回答必须使用以下固定结构：
结论：<直接回答问题>
关键证据：<列出关键数据，并在每条关键数据后标注其来源 result_id>
建议或限制：<由证据支持的建议；若证据不足，写明限制和最窄可支持结论>

提交回答前自检：
- 所有事实是否都来自本次工具结果。
- 指标的升降方向是否与工具结果一致。
- 建议是否超出了已观察事实的范围。
"""


def v2_prompt_snapshot() -> PromptSnapshot:
    return make_prompt_snapshot(
        name=V2_PROMPT_NAME,
        version=V2_PROMPT_VERSION,
        text=V2_PROMPT_TEXT,
        change_note=V2_CHANGE_NOTE,
        source_file="app/prompts/v2.py",
    )
```

Update `app/prompts/__init__.py`:

```python
"""Frozen real-model system prompts."""

from app.prompts.v1 import V1_PROMPT_NAME, V1_PROMPT_VERSION, v1_prompt_snapshot
from app.prompts.v2 import V2_PROMPT_NAME, V2_PROMPT_VERSION, v2_prompt_snapshot

__all__ = [
    "V1_PROMPT_NAME",
    "V1_PROMPT_VERSION",
    "v1_prompt_snapshot",
    "V2_PROMPT_NAME",
    "V2_PROMPT_VERSION",
    "v2_prompt_snapshot",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_prompts_v2.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/prompts/v2.py app/prompts/__init__.py tests/test_prompts_v2.py
git commit -m "feat(prompts): add real LLM V2 system prompt"
```

---

### Task 4: Provider settings and preflight

**Files:**
- Create: `app/experiments/llm_settings.py`
- Test: `tests/test_llm_settings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_settings.py
import pytest

from app.experiments.llm_settings import (
    DEFAULT_BASE_URL,
    ProviderConfigError,
    ProviderSettings,
    base_url_host,
    load_provider_settings,
)


def env(**overrides: str) -> dict[str, str]:
    base = {
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "ep-2026xxx",
    }
    base.update(overrides)
    return base


def test_loads_required_settings_with_defaults():
    settings = load_provider_settings(env())
    assert isinstance(settings, ProviderSettings)
    assert settings.api_key == "test-key"
    assert settings.model == "ep-2026xxx"
    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.temperature == 0.0
    assert settings.timeout_seconds == 30.0


def test_base_url_host_exposes_host_only():
    assert base_url_host(DEFAULT_BASE_URL) == "ark.cn-beijing.volces.com"


@pytest.mark.parametrize(
    "overrides",
    [
        {"LLM_API_KEY": ""},
        {"LLM_MODEL": ""},
        {"LLM_TEMPERATURE": "0.1"},
        {"LLM_TEMPERATURE": "1"},
        {"LLM_TIMEOUT_SECONDS": "0"},
        {"LLM_TIMEOUT_SECONDS": "abc"},
        {"LLM_BASE_URL": "not-a-url"},
        {"LLM_BASE_URL": "file:///tmp/x"},
    ],
)
def test_rejects_invalid_configuration(overrides):
    with pytest.raises(ProviderConfigError):
        load_provider_settings(env(**overrides))


def test_api_key_is_not_in_error_repr():
    try:
        load_provider_settings(env(LLM_MODEL=""))
    except ProviderConfigError as error:
        assert "test-key" not in repr(error)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_llm_settings.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/experiments/llm_settings.py
import math
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_TIMEOUT_SECONDS = 30.0


class ProviderConfigError(RuntimeError):
    """A stable, non-secret provider configuration failure (has .code)."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ProviderSettings:
    api_key: str
    model: str
    base_url: str
    temperature: float
    timeout_seconds: float


def base_url_host(base_url: str) -> str:
    return urlsplit(base_url).hostname or ""


def _require_non_empty(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigError(code)
    return value.strip()


def _validated_base_url(value: str) -> str:
    base_url = value.rstrip("/")
    try:
        parsed = urlsplit(base_url)
    except ValueError:
        raise ProviderConfigError("invalid_base_url") from None
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ProviderConfigError("invalid_base_url")
    return base_url


def _validated_temperature(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        raise ProviderConfigError("invalid_temperature") from None
    if not math.isfinite(temperature) or temperature != 0.0:
        raise ProviderConfigError("invalid_temperature")
    return 0.0


def _validated_timeout(value: str | None) -> float:
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise ProviderConfigError("invalid_timeout") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ProviderConfigError("invalid_timeout")
    return timeout


def load_provider_settings(environ: Mapping[str, str]) -> ProviderSettings:
    api_key = _require_non_empty(environ.get("LLM_API_KEY"), "missing_api_key")
    model = _require_non_empty(environ.get("LLM_MODEL"), "missing_model")
    raw_base_url = environ.get("LLM_BASE_URL")
    base_url = (
        DEFAULT_BASE_URL if raw_base_url is None or not raw_base_url.strip()
        else _validated_base_url(raw_base_url.strip())
    )
    return ProviderSettings(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=_validated_temperature(environ.get("LLM_TEMPERATURE")),
        timeout_seconds=_validated_timeout(environ.get("LLM_TIMEOUT_SECONDS")),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_llm_settings.py -v`
Expected: PASS (all parametrized cases pass).

- [ ] **Step 5: Commit**

```bash
git add app/experiments/llm_settings.py tests/test_llm_settings.py
git commit -m "feat(experiments): add provider settings preflight"
```

---

### Task 5: Adapter system prompt injection

**Files:**
- Modify: `app/llm/openai_compatible.py`
- Test: `tests/test_llm_openai_compatible.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_openai_compatible.py`:

```python
def test_openai_adapter_injects_system_prompt_on_first_turn_only() -> None:
    transport = SequenceTransport([tool_payload(), answer_payload("完成。")])
    adapter = OpenAICompatibleAdapter(
        base_url="https://example.test/v1/",
        api_key="secret",
        model="test-model",
        system_prompt="你是经营分析助手。",
        transport=transport,
    )
    adapter.complete(sample_request())
    execution = execution_for(first_call_id := "call_abc")
    adapter.complete(sample_request(executions=[execution]))

    first_messages = transport.payloads[0]["messages"]
    assert first_messages == [
        {"role": "system", "content": "你是经营分析助手。"},
        {"role": "user", "content": "查看 P001"},
    ]
    second_roles = [message["role"] for message in transport.payloads[1]["messages"]]
    assert second_roles == ["system", "user", "assistant", "tool"]
    assert transport.payloads[1]["messages"].count(
        {"role": "system", "content": "你是经营分析助手。"}
    ) == 1


def test_openai_adapter_without_system_prompt_keeps_single_user_turn() -> None:
    transport = SequenceTransport([answer_payload()])
    adapter_with(transport).complete(sample_request())
    assert transport.payloads[0]["messages"] == [
        {"role": "user", "content": "查看 P001"}
    ]
```

(The `first_call_id :=` binding is unnecessary; if your linter flags it, write `execution = execution_for("call_abc")` instead.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_llm_openai_compatible.py -k system_prompt -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'system_prompt'`.

- [ ] **Step 3: Implement**

In `app/llm/openai_compatible.py`, change the constructor signature and state:

```python
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        system_prompt: str | None = None,
        transport: JsonTransport | None = None,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self._api_key = _validate_non_empty_string(api_key, "api_key")
        self._model = _validate_non_empty_string(model, "model")
        self._temperature = _validate_temperature(temperature)
        self._timeout_seconds = _validate_timeout(timeout_seconds)
        self._system_prompt = _validate_optional_system_prompt(system_prompt)
        self._transport = transport or UrllibJsonTransport()
        self.start_run()
```

In `start_run`, seed the flag:

```python
    def start_run(self) -> None:
        self._messages: list[dict[str, JsonValue]] = []
        self._user_input: str | None = None
        self._tools: list[dict[str, JsonValue]] | None = None
        self._pending_call_id: str | None = None
        self._pending_arguments: dict[str, JsonValue] | None = None
        self._sent_executions: list[dict[str, JsonValue]] = []
        self._sent_call_ids: set[str] = set()
        self._system_seeded = False
```

Include `_system_seeded` in the `complete()` snapshot tuple (append as the 8th element on both the snapshot build and the restore assignment):

```python
        snapshot = (
            deepcopy(self._messages),
            self._user_input,
            deepcopy(self._tools),
            self._pending_call_id,
            deepcopy(self._pending_arguments),
            deepcopy(self._sent_executions),
            set(self._sent_call_ids),
            self._system_seeded,
        )
```

```python
            (
                self._messages,
                self._user_input,
                self._tools,
                self._pending_call_id,
                self._pending_arguments,
                self._sent_executions,
                self._sent_call_ids,
                self._system_seeded,
            ) = snapshot
```

In `_advance_transcript`, prepend the system message on the first turn:

```python
        if self._user_input is None:
            if request.prior_tool_executions:
                raise OpenAIProtocolError("unknown call_id in prior tool executions")
            self._user_input = request.user_input
            self._tools = deepcopy(request.tools)
            if self._system_prompt is not None and not self._system_seeded:
                self._messages.append(
                    {"role": "system", "content": self._system_prompt}
                )
                self._system_seeded = True
            self._messages.append({"role": "user", "content": request.user_input})
            return
```

Add the module-level validators near the other `_validate_*` helpers:

```python
def _validate_optional_system_prompt(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("system_prompt must be a non-empty string when provided")
    return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_llm_openai_compatible.py -v`
Expected: PASS — the two new tests pass and all existing adapter tests remain green.

- [ ] **Step 5: Commit**

```bash
git add app/llm/openai_compatible.py tests/test_llm_openai_compatible.py
git commit -m "feat(llm): support immutable system prompt injection"
```

---

### Task 6: Artifact v2 models

**Files:**
- Create: `app/experiments/llm_artifacts.py`
- Test: `tests/test_llm_artifacts_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_artifacts_models.py
import pytest

from app.experiments.llm_artifacts import (
    RealCaseRunRecord,
    RealRunSummary,
    StoredPromptSnapshot,
)


def make_record(**overrides):
    base = dict(
        sequence=1,
        case_id="CASE_001",
        split="development",
        dataset_id="e1e81533c25e03e5",
        dataset_version="1.0.0",
        agent_version="v1",
        repeat_index=1,
        user_input="诊断最近30天 GMV 变化",
        status="completed",
        final_answer="GMV 上升",
        error_code=None,
        prior_tool_executions=(),
        decision_trace=(),
        latency_ms=120,
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
    )
    base.update(overrides)
    return RealCaseRunRecord(**base)


def test_completed_record_requires_answer_and_no_error():
    record = make_record()
    assert record.agent_version == "v1"
    with pytest.raises(Exception):
        make_record(final_answer=None)
    with pytest.raises(Exception):
        make_record(error_code="adapter_error")


def test_failed_record_requires_error_code_and_null_usage_allowed():
    record = make_record(
        status="failed",
        final_answer=None,
        error_code="adapter_error",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )
    assert record.status == "failed"
    assert record.total_tokens is None


def test_prompt_snapshot_persists_text():
    prompt = StoredPromptSnapshot(
        prompt_name="real-llm-v1",
        prompt_version="1.0.0",
        prompt_text="系统",
        prompt_sha256="a" * 64,
        change_note="initial",
        source_file="app/prompts/v1.py",
    )
    assert prompt.prompt_text == "系统"


def test_summary_accepts_unavailable_tokens_and_unavailable_cost():
    summary = RealRunSummary(
        total=1,
        completed=1,
        failed=0,
        stopped=0,
        split_counts={"development": 1, "public_validation": 0},
        split_status_counts={
            "development": {"completed": 1, "failed": 0},
            "public_validation": {"completed": 0, "failed": 0},
        },
        first_case_id="CASE_001",
        last_case_id="CASE_001",
        latency={"p50": 120, "p95": 120, "sample_size": 1},
        tokens={
            "available": True,
            "prompt_total": 10,
            "completion_total": 4,
            "total_total": 14,
            "p50": 14,
            "p95": 14,
            "sample_size": 1,
        },
        cost="unavailable",
        evaluation_status="pending_not_run",
    )
    assert summary.cost == "unavailable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_llm_artifacts_models.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/experiments/llm_artifacts.py
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import SplitCounts, SplitRunStatusCounts

ARTIFACT_SCHEMA_VERSION = "2.0.0"
REAL_OUTPUT_FILE_NAMES = {
    "case_runs.jsonl",
    "summary.json",
    "prompt_snapshot.json",
}
AgentVersion = Literal["v1", "v2"]
ExperimentMode = Literal["pilot", "formal"]


class LlmArtifactModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )


class RealCaseRunRecord(LlmArtifactModel):
    sequence: int = Field(ge=1)
    case_id: str = Field(pattern=r"^CASE_[0-9]{3}$")
    split: Literal["development", "public_validation"]
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    dataset_version: str = Field(min_length=1)
    agent_version: AgentVersion
    repeat_index: Literal[1]
    user_input: str = Field(min_length=1)
    status: Literal["completed", "failed"]
    final_answer: str | None
    error_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    prior_tool_executions: tuple
    decision_trace: tuple
    latency_ms: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_status_fields(self) -> "RealCaseRunRecord":
        if self.status == "completed":
            if self.final_answer is None:
                raise ValueError("completed records require final_answer")
            if self.error_code is not None:
                raise ValueError("completed records cannot contain error_code")
        elif self.final_answer is not None or self.error_code is None:
            raise ValueError("failed records require error_code and no final_answer")
        token_fields = (
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        )
        if any(value is None for value in token_fields) and not all(
            value is None for value in token_fields
        ):
            raise ValueError("usage token fields must all be present or all be null")
        return self


class StoredPromptSnapshot(LlmArtifactModel):
    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    change_note: str = Field(min_length=1)
    source_file: str = Field(min_length=1)


class LatencyStats(LlmArtifactModel):
    p50: int = Field(ge=0)
    p95: int = Field(ge=0)
    sample_size: int = Field(ge=0)


class TokenStats(LlmArtifactModel):
    available: bool
    prompt_total: int | None = Field(default=None, ge=0)
    completion_total: int | None = Field(default=None, ge=0)
    total_total: int | None = Field(default=None, ge=0)
    p50: int | None = Field(default=None, ge=0)
    p95: int | None = Field(default=None, ge=0)
    sample_size: int = Field(ge=0)


class RealRunSummary(LlmArtifactModel):
    total: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    stopped: int = Field(ge=0)
    split_counts: SplitCounts
    split_status_counts: SplitRunStatusCounts
    first_case_id: str | None = Field(default=None, pattern=r"^CASE_[0-9]{3}$")
    last_case_id: str | None = Field(default=None, pattern=r"^CASE_[0-9]{3}$")
    latency: LatencyStats
    tokens: TokenStats
    cost: Literal["unavailable"]
    evaluation_status: Literal["pending_not_run"]

    @model_validator(mode="after")
    def validate_counts(self) -> "RealRunSummary":
        if self.completed + self.failed + self.stopped != self.total:
            raise ValueError("status counts must equal total")
        if (
            self.split_counts.development + self.split_counts.public_validation
            != self.total
        ):
            raise ValueError("split counts must equal total")
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_llm_artifacts_models.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add app/experiments/llm_artifacts.py tests/test_llm_artifacts_models.py
git commit -m "feat(experiments): add artifact v2 data models"
```

---

<!-- BATCH_A_CONTINUED -->

### Task 7: Artifact v2 manifest, canonical bytes and staging writer

**Files:**
- Modify: `app/experiments/llm_artifacts.py`
- Test: `tests/test_llm_artifacts_writer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_artifacts_writer.py
import json
from pathlib import Path

import pytest

from app.experiments.artifacts import RuntimeVersions
from app.experiments.llm_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    RealArtifactWriter,
    RealCaseRunRecord,
    RealRunArtifactBundle,
    RealRunManifest,
    RealRunSummary,
    StagedPromptSnapshot,
    canonical_real_jsonl_bytes,
    verify_published_real_run,
)

DATASET_ID = "e1e81533c25e03e5"
SHA = "a" * 64


def make_record(sequence: int = 1, case_id: str = "CASE_001") -> RealCaseRunRecord:
    return RealCaseRunRecord(
        sequence=sequence,
        case_id=case_id,
        split="development",
        dataset_id=DATASET_ID,
        dataset_version="1.0.0",
        agent_version="v1",
        repeat_index=1,
        user_input="诊断最近30天 GMV 变化",
        status="completed",
        final_answer="GMV 上升",
        error_code=None,
        prior_tool_executions=(),
        decision_trace=(),
        latency_ms=120,
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
    )


def prompt() -> StoredPromptSnapshot:
    return StoredPromptSnapshot(
        prompt_name="real-llm-v1",
        prompt_version="1.0.0",
        prompt_text="系统",
        prompt_sha256=SHA,
        change_note="initial",
        source_file="app/prompts/v1.py",
    )


def summary() -> RealRunSummary:
    return RealRunSummary(
        total=1,
        completed=1,
        failed=0,
        stopped=0,
        split_counts={"development": 1, "public_validation": 0},
        split_status_counts={
            "development": {"completed": 1, "failed": 0},
            "public_validation": {"completed": 0, "failed": 0},
        },
        first_case_id="CASE_001",
        last_case_id="CASE_001",
        latency={"p50": 120, "p95": 120, "sample_size": 1},
        tokens={
            "available": True,
            "prompt_total": 10,
            "completion_total": 4,
            "total_total": 14,
            "p50": 14,
            "p95": 14,
            "sample_size": 1,
        },
        cost="unavailable",
        evaluation_status="pending_not_run",
    )


def manifest(*, run_id: str = "real-v1__20261009T000000Z__abcd1234") -> RealRunManifest:
    return RealRunManifest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        run_id=run_id,
        comparison_group_id="formal__20261009T000000Z__abcd1234",
        experiment_mode="formal",
        agent_version="v1",
        started_at_utc="2026-10-09T00:00:00.000000Z",
        completed_at_utc="2026-10-09T00:01:00.000000Z",
        git_commit="0" * 40,
        git_dirty=False,
        runtime_versions=RuntimeVersions(
            python="3.11.0",
            pydantic="2.9.0",
            duckdb="1.1.0",
            pandas="2.2.0",
            pyarrow="17.0.0",
        ),
        case_set_id="1" * 16,
        case_jsonl_sha256=SHA,
        case_manifest_sha256=SHA,
        case_subset_id="all-cases-v1",
        case_subset_sha256=SHA,
        datasets=(),
        tool_contract_version="1.0",
        tool_contract_sha256=SHA,
        adapter_name="openai-compatible",
        model="ep-test",
        base_url_host="ark.cn-beijing.volces.com",
        temperature=0.0,
        timeout_seconds=30.0,
        max_steps=8,
        concurrency=1,
        repeats=1,
        prompt={
            "prompt_name": "real-llm-v1",
            "prompt_version": "1.0.0",
            "source_file": "app/prompts/v1.py",
            "prompt_sha256": SHA,
        },
        output_files=None,
        evaluation_status="pending_not_run",
        usage_status="available",
        cost_status="unavailable",
    )


def bundle() -> RealRunArtifactBundle:
    return RealRunArtifactBundle(
        records=(make_record(),),
        expected_case_ids=("CASE_001",),
        summary=summary(),
        prompt_snapshot=prompt(),
        manifest=manifest(),
    )


def test_canonical_real_jsonl_roundtrip():
    payload = canonical_real_jsonl_bytes((make_record(),))
    assert payload.endswith(b"\n")
    parsed = json.loads(payload)
    assert parsed["case_id"] == "CASE_001"
    assert parsed["latency_ms"] == 120
    assert parsed["total_tokens"] == 14


def test_stage_does_not_publish_until_commit(tmp_path: Path):
    writer = RealArtifactWriter(tmp_path)
    staged = writer.stage(bundle())
    assert not (tmp_path / "real-v1__20261009T000000Z__abcd1234").exists()
    final = writer.commit(staged)
    assert final.is_dir()
    published = verify_published_real_run(final)
    assert published.manifest.run_id == "real-v1__20261009T000000Z__abcd1234"


def test_publish_rejects_overwrite(tmp_path: Path):
    RealArtifactWriter(tmp_path).publish(bundle())
    with pytest.raises(FileExistsError):
        RealArtifactWriter(tmp_path).publish(bundle())


def test_publish_rejects_prompt_hash_mismatch(tmp_path: Path):
    bad_prompt = prompt().model_copy(update={"prompt_text": "different"})
    bad_bundle = bundle().model_copy(update={"prompt_snapshot": bad_prompt})
    with pytest.raises(ValueError, match="prompt"):
        RealArtifactWriter(tmp_path).publish(bad_bundle)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_llm_artifacts_writer.py -v`
Expected: FAIL with `ImportError` for the new writer/functions.

- [ ] **Step 3: Implement — append to `app/experiments/llm_artifacts.py`**

```python
import json
import os
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from filelock import FileLock
from pydantic import Field, RootModel, model_validator

from .artifacts import (
    DatasetArtifactIdentity,
    RuntimeVersions,
    canonical_json_bytes,
)

_REAL_PUBLISH_LOCK_NAME = ".real-artifact-publish.lock"


class PromptIdentity(LlmArtifactModel):
    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RealOutputFileHashes(RootModel[dict[str, str]]):
    model_config = ConfigDict(
        revalidate_instances="always",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )

    @model_validator(mode="after")
    def validate_file_hashes(self) -> "RealOutputFileHashes":
        if set(self.root) != REAL_OUTPUT_FILE_NAMES:
            raise ValueError("real output hashes must cover the three non-manifest files")
        if any(
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.root.values()
        ):
            raise ValueError("output hashes must be lowercase SHA-256 values")
        return self


class RealRunManifest(LlmArtifactModel):
    artifact_schema_version: Literal["2.0.0"]
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    comparison_group_id: str = Field(min_length=1)
    experiment_mode: ExperimentMode
    agent_version: AgentVersion
    started_at_utc: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
    )
    completed_at_utc: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
    )
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    runtime_versions: RuntimeVersions
    case_set_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    case_jsonl_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_subset_id: str = Field(min_length=1)
    case_subset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    datasets: tuple[DatasetArtifactIdentity, ...]
    tool_contract_version: str = Field(min_length=1)
    tool_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url_host: str = Field(min_length=1)
    temperature: float
    timeout_seconds: float
    max_steps: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    repeats: int = Field(ge=1)
    prompt: PromptIdentity
    output_files: RealOutputFileHashes | None = None
    evaluation_status: Literal["pending_not_run"]
    usage_status: Literal["available", "unavailable"]
    cost_status: Literal["unavailable"]


def _record_json_payload(record: RealCaseRunRecord) -> dict[str, object]:
    return {
        "sequence": record.sequence,
        "case_id": record.case_id,
        "split": record.split,
        "dataset_id": record.dataset_id,
        "dataset_version": record.dataset_version,
        "agent_version": record.agent_version,
        "repeat_index": record.repeat_index,
        "user_input": record.user_input,
        "status": record.status,
        "final_answer": record.final_answer,
        "error_code": record.error_code,
        "prior_tool_executions": record.prior_tool_executions,
        "decision_trace": record.decision_trace,
        "latency_ms": record.latency_ms,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "total_tokens": record.total_tokens,
    }


def canonical_real_jsonl_bytes(records: Sequence[RealCaseRunRecord]) -> bytes:
    return b"".join(
        canonical_json_bytes(_record_json_payload(record)) for record in records
    )


class RealRunArtifactBundle(LlmArtifactModel):
    records: tuple[RealCaseRunRecord, ...]
    expected_case_ids: tuple[str, ...]
    summary: RealRunSummary
    prompt_snapshot: StoredPromptSnapshot
    manifest: RealRunManifest

    @model_validator(mode="after")
    def validate_bundle(self) -> "RealRunArtifactBundle":
        actual_ids = tuple(record.case_id for record in self.records)
        if actual_ids != self.expected_case_ids:
            raise ValueError("case record order does not match expected case order")
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError("case IDs must be unique")
        if tuple(record.sequence for record in self.records) != tuple(
            range(1, len(self.records) + 1)
        ):
            raise ValueError("case record sequence must be contiguous and one-based")
        statuses = Counter(record.status for record in self.records)
        if self.summary.completed != statuses["completed"] or self.summary.failed != (
            statuses["failed"]
        ):
            raise ValueError("summary status counts do not match records")
        if self.manifest.output_files is not None:
            raise ValueError("unpublished bundle manifest cannot contain output hashes")
        return self


class PublishedRealRun(LlmArtifactModel):
    records: tuple[RealCaseRunRecord, ...]
    summary: RealRunSummary
    prompt_snapshot: StoredPromptSnapshot
    manifest: RealRunManifest


def _read_real_directory(path: Path) -> dict[str, bytes]:
    if not path.is_dir() or path.is_symlink():
        raise ValueError("published run must be a non-symlink directory")
    expected = REAL_OUTPUT_FILE_NAMES | {"run_manifest.json"}
    contents: dict[str, bytes] = {}
    for name in expected:
        file_path = path / name
        if not file_path.is_file() or file_path.is_symlink():
            raise ValueError(f"{name} must be a regular non-symlink file")
        contents[name] = file_path.read_bytes()
    return contents


def verify_published_real_run(path: Path) -> PublishedRealRun:
    files = _read_real_directory(path)
    manifest = RealRunManifest.model_validate_json(
        files["run_manifest.json"], strict=True
    )
    summary = RealRunSummary.model_validate_json(files["summary.json"], strict=True)
    prompt_snapshot = StoredPromptSnapshot.model_validate_json(
        files["prompt_snapshot.json"], strict=True
    )
    records = tuple(
        RealCaseRunRecord.model_validate_json(line, strict=True)
        for line in files["case_runs.jsonl"].splitlines()
        if line
    )
    if manifest.output_files is None:
        raise ValueError("published manifest must contain output hashes")
    builders = {
        "case_runs.jsonl": canonical_real_jsonl_bytes(records),
        "summary.json": canonical_json_bytes(summary),
        "prompt_snapshot.json": canonical_json_bytes(prompt_snapshot),
    }
    for name, expected_bytes in builders.items():
        if sha256(files[name]).hexdigest() != manifest.output_files.root[name]:
            raise ValueError(f"{name} SHA-256 does not match manifest")
        if files[name] != expected_bytes:
            raise ValueError(f"{name} is not canonical")
    if prompt_snapshot.prompt_sha256 != manifest.prompt.prompt_sha256:
        raise ValueError("prompt SHA-256 does not match manifest identity")
    return PublishedRealRun(
        records=records,
        summary=summary,
        prompt_snapshot=prompt_snapshot,
        manifest=manifest,
    )


@dataclass(frozen=True)
class StagedRealRun:
    temp_dir: Path
    final_dir: Path


class RealArtifactWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def stage(self, bundle: RealRunArtifactBundle) -> StagedRealRun:
        validated = RealRunArtifactBundle.model_validate(
            bundle.model_dump(mode="python"), strict=True
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        final_dir = self.output_root / validated.manifest.run_id
        if final_dir.exists() or final_dir.is_symlink():
            raise FileExistsError(
                f"run already exists: {validated.manifest.run_id}"
            )
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{validated.manifest.run_id}.tmp-",
                dir=self.output_root,
            )
        )
        files = {
            "case_runs.jsonl": canonical_real_jsonl_bytes(validated.records),
            "summary.json": canonical_json_bytes(validated.summary),
            "prompt_snapshot.json": canonical_json_bytes(validated.prompt_snapshot),
        }
        prompt_hash = sha256(files["prompt_snapshot.json"]).hexdigest()
        if (
            validated.prompt_snapshot.prompt_sha256
            != validated.manifest.prompt.prompt_sha256
            or prompt_hash != validated.manifest.prompt.prompt_sha256
        ):
            shutil.rmtree(temp_dir)
            raise ValueError("prompt snapshot SHA-256 does not match manifest")
        try:
            for name, contents in files.items():
                (temp_dir / name).write_bytes(contents)
            hashes = {
                name: sha256(contents).hexdigest()
                for name, contents in files.items()
            }
            published_manifest = validated.manifest.model_copy(
                update={"output_files": RealOutputFileHashes(hashes)}
            )
            (temp_dir / "run_manifest.json").write_bytes(
                canonical_json_bytes(published_manifest)
            )
            verify_published_real_run(temp_dir)
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return StagedRealRun(temp_dir=temp_dir, final_dir=final_dir)

    def commit(self, staged: StagedRealRun) -> Path:
        with FileLock(self.output_root / _REAL_PUBLISH_LOCK_NAME):
            if staged.final_dir.exists() or staged.final_dir.is_symlink():
                shutil.rmtree(staged.temp_dir, ignore_errors=True)
                raise FileExistsError(f"run already exists: {staged.final_dir.name}")
            staged.temp_dir.replace(staged.final_dir)
        return staged.final_dir

    def publish(self, bundle: RealRunArtifactBundle) -> Path:
        staged = self.stage(bundle)
        return self.commit(staged)
```

Because the top of the file already imports `Literal` and `BaseModel, ConfigDict` etc., merge duplicate imports (keep one combined import block); do not leave two `from pydantic import ...` lines. `os` and `stat` may be unused after the merge — only import the modules the appended code actually uses (`json`, `shutil`, `tempfile`, `Counter`, `Mapping`, `Sequence`, `dataclass`, `sha256`, `Path`, `FileLock`, `Field`, `RootModel`, `model_validator`, plus the `.artifacts` names).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_llm_artifacts_writer.py tests/test_llm_artifacts_models.py -v`
Expected: PASS for both files.

- [ ] **Step 5: Lint and commit**

Run: `python3 -m ruff check app/experiments/llm_artifacts.py tests/test_llm_artifacts_writer.py`
Expected: `All checks passed!` Fix import ordering if ruff reports it.

```bash
git add app/experiments/llm_artifacts.py tests/test_llm_artifacts_writer.py
git commit -m "feat(experiments): add artifact v2 staging writer and verification"
```

---

### Task 8: Frozen 20-Case Pilot set and selection

**Files:**
- Create: `app/experiments/pilot_cases.py`
- Test: `tests/test_pilot_cases.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pilot_cases.py
import json
from pathlib import Path

import pytest

from app.experiments.cases import (
    CaseBundle,
    DatasetIdentity,
    EvaluationCaseManifest,
    RunnableCase,
)
from app.experiments.pilot_cases import (
    PILOT_CASE_IDS,
    PILOT_CASE_SET_VERSION,
    PilotCaseSetError,
    pilot_case_set_sha256,
    select_pilot_cases,
)
from tests.test_baseline_v1_batch import dataset_dirs  # noqa: F401

SHA = "a" * 64


def _bundle(
    dataset_dirs: dict[str, Path], case_ids: tuple[str, ...]
) -> CaseBundle:
    development = json.loads(
        (dataset_dirs["development"] / "manifest.json").read_text()
    )
    datasets = {
        "development": DatasetIdentity(
            dataset_id=development["dataset_id"],
            dataset_version=development["dataset_version"],
            generator_config_hash=development["config_sha256"],
        ),
    }
    cases = tuple(
        RunnableCase(
            case_id=case_id,
            user_input="诊断最近30天 GMV 变化",
            split="development",
            dataset_id=datasets["development"].dataset_id,
            dataset_version=datasets["development"].dataset_version,
        )
        for case_id in case_ids
    )
    manifest = EvaluationCaseManifest(
        business_task_counts={"gmv_diagnosis": len(cases)},
        capability_counts={"basic_query": len(cases)},
        case_count=len(cases),
        case_schema_version="1.0",
        case_set_id="1" * 16,
        datasets={
            **datasets,
            "public_validation": DatasetIdentity(
                dataset_id="2" * 16,
                dataset_version="1.0.0",
                generator_config_hash=SHA,
            ),
        },
        difficulty_counts={"medium": len(cases)},
        jsonl_sha256=SHA,
        source_label="test cases",
        split_counts={"development": len(cases), "public_validation": 0},
        split_strategy={"kind": "test"},
        tool_contract={"version": "1.0", "sha256": SHA},
    )
    return CaseBundle(manifest=manifest, cases=cases, manifest_sha256=SHA)


def test_pilot_set_is_frozen_20_unique_and_stable_hash():
    assert PILOT_CASE_SET_VERSION == "1.0.0"
    assert len(PILOT_CASE_IDS) == 20
    assert len(set(PILOT_CASE_IDS)) == 20
    assert pilot_case_set_sha256() == pilot_case_set_sha256()


def test_select_pilot_cases_returns_frozen_order(
    dataset_dirs: dict[str, Path],
) -> None:
    bundle = _bundle(dataset_dirs, tuple(PILOT_CASE_IDS))
    selected = select_pilot_cases(bundle)
    assert tuple(case.case_id for case in selected) == PILOT_CASE_IDS


def test_select_pilot_cases_raises_when_id_missing(dataset_dirs):
    bundle = _bundle(dataset_dirs, tuple(PILOT_CASE_IDS[:-1]))
    with pytest.raises(PilotCaseSetError):
        select_pilot_cases(bundle)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_pilot_cases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.experiments.pilot_cases'`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/experiments/pilot_cases.py
import json
from hashlib import sha256

from .cases import CaseBundle, RunnableCase

PILOT_CASE_SET_VERSION = "1.0.0"
PILOT_CASE_IDS = (
    "CASE_003",
    "CASE_005",
    "CASE_013",
    "CASE_017",
    "CASE_022",
    "CASE_027",
    "CASE_029",
    "CASE_032",
    "CASE_041",
    "CASE_053",
    "CASE_055",
    "CASE_059",
    "CASE_063",
    "CASE_065",
    "CASE_076",
    "CASE_079",
    "CASE_087",
    "CASE_089",
    "CASE_091",
    "CASE_097",
)


class PilotCaseSetError(RuntimeError):
    pass


def pilot_case_set_sha256() -> str:
    payload = json.dumps(
        {"version": PILOT_CASE_SET_VERSION, "case_ids": list(PILOT_CASE_IDS)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def select_pilot_cases(bundle: CaseBundle) -> tuple[RunnableCase, ...]:
    by_id = {case.case_id: case for case in bundle.cases}
    missing = [case_id for case_id in PILOT_CASE_IDS if case_id not in by_id]
    if missing:
        raise PilotCaseSetError("pilot case set is incomplete")
    selected = tuple(by_id[case_id] for case_id in PILOT_CASE_IDS)
    if any(case.split != "development" for case in selected):
        raise PilotCaseSetError("pilot cases must all be development cases")
    return selected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_pilot_cases.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/experiments/pilot_cases.py tests/test_pilot_cases.py
git commit -m "feat(experiments): add frozen 20-case pilot selection"
```

---

<!-- BATCH_A_RUNNER -->

### Task 9: PairedRunCoordinator — isolated real-model paired execution

Create the coordinator that runs both prompt versions against the same Cases, with a fresh catalog/registry/adapter per version, per-Case failure isolation, formal alternation, drift checks, and paired publish. Records carry JSON-compatible payloads (never pydantic objects).

**Files:**
- Create: `app/experiments/llm_run.py`
- Test: `tests/test_real_llm_run.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_real_llm_run.py
import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

import pytest

from app.data.generator import build_snapshot
from app.experiments.cases import (
    CaseBundle,
    DatasetIdentity,
    EvaluationCaseManifest,
    RunnableCase,
)
from app.experiments.llm_run import PairedRunCoordinator, RunSettings
from app.experiments.llm_settings import ProviderSettings

from tests.test_baseline_v1_batch import _case_bundle, _dataset_mapping
from tests.test_llm_openai_compatible import answer_payload

ROOT = Path(__file__).parents[1]
DEVELOPMENT_CONFIG = ROOT / "configs/data/synthetic_v1.yaml"
PUBLIC_VALIDATION_CONFIG = (
    ROOT / "configs/data/synthetic_public_validation_v1.yaml"
)
TOOL_CONTRACT = ROOT / "configs/evaluation/tool_contract_v1.yaml"
TOOL_CONTRACT_SHA256 = sha256(TOOL_CONTRACT.read_bytes()).hexdigest()
SHA = "a" * 64


@pytest.fixture(scope="module")
def dataset_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("real-llm-run")
    paths = {
        "development": root / "development",
        "public_validation": root / "public-validation",
    }
    build_snapshot(DEVELOPMENT_CONFIG, paths["development"])
    build_snapshot(PUBLIC_VALIDATION_CONFIG, paths["public_validation"])
    return paths


def _pilot_bundle(dataset_dirs: dict[str, Path]) -> CaseBundle:
    metadata = json.loads(
        (dataset_dirs["development"] / "manifest.json").read_text(encoding="utf-8")
    )
    development = DatasetIdentity(
        dataset_id=metadata["dataset_id"],
        dataset_version=metadata["dataset_version"],
        generator_config_hash=metadata["config_sha256"],
    )
    cases = tuple(
        RunnableCase(
            case_id=case_id,
            user_input="诊断最近30天 GMV 变化",
            split="development",
            dataset_id=development.dataset_id,
            dataset_version=development.dataset_version,
        )
        for case_id in PILOT_IDS
    )
    return CaseBundle(
        manifest=EvaluationCaseManifest(
            business_task_counts={"gmv_change": 20},
            capability_counts={"diagnosis": 20},
            case_count=20,
            case_schema_version="1.0",
            case_set_id="1" * 16,
            datasets={"development": development},
            difficulty_counts={"medium": 20},
            jsonl_sha256=SHA,
            source_label="test cases",
            split_counts={"development": 20},
            split_strategy={"kind": "test"},
            tool_contract={
                "version": "1.0",
                "sha256": TOOL_CONTRACT_SHA256,
            },
        ),
        cases=cases,
        manifest_sha256=SHA,
    )


PILOT_IDS = (
    "CASE_003", "CASE_005", "CASE_013", "CASE_017", "CASE_022",
    "CASE_027", "CASE_029", "CASE_032", "CASE_041", "CASE_053",
    "CASE_055", "CASE_059", "CASE_063", "CASE_065", "CASE_076",
    "CASE_079", "CASE_087", "CASE_089", "CASE_091", "CASE_097",
)


class _ImmediateAnswer:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, url, *, headers, payload, timeout_seconds):  # noqa: ANN001
        self.calls += 1
        return answer_payload("完成。")


class _FailOnce:
    def __init__(self, fail_on: int) -> None:
        self.calls = 0
        self._fail_on = fail_on

    def post(self, url, *, headers, payload, timeout_seconds):  # noqa: ANN001
        self.calls += 1
        if self.calls == self._fail_on:
            from app.llm.openai_compatible import OpenAITransportError

            raise OpenAITransportError("provider request failed")
        return answer_payload("完成。")


def _provider(transport) -> ProviderSettings:
    return ProviderSettings(
        api_key="test-key",
        model="test-model",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        temperature=0.0,
        timeout_seconds=30.0,
    )


def test_pilot_runs_both_versions_in_frozen_order(tmp_path: Path, dataset_dirs):
    cases = _pilot_bundle(dataset_dirs)
    mapping = {
        cases.manifest.datasets["development"].dataset_id: dataset_dirs["development"]
    }
    coordinator = PairedRunCoordinator(
        provider=_provider(None),
        run_settings=RunSettings(),
        transport=_ImmediateAnswer(),
    )

    result = coordinator.run(
        mode="pilot",
        cases=cases,
        dataset_dirs=mapping,
        run_output_root=tmp_path,
    )

    assert result.comparison_group_id.startswith("pilot__")
    assert result.case_ids == PILOT_IDS
    assert [r.case_id for r in result.v1.records] == list(PILOT_IDS)
    assert [r.case_id for r in result.v2.records] == list(PILOT_IDS)
    assert {r.agent_version for r in result.v1.records} == {"v1"}
    assert {r.agent_version for r in result.v2.records} == {"v2"}
    assert result.v1.summary.total == 20
    assert result.v2.summary.total == 20
    assert all(order == ("v1", "v2") for order in result.version_order.values())
    assert all(
        r.prior_tool_executions == () and isinstance(r.decision_trace, tuple)
        for r in result.v1.records
    )


def test_formal_alternates_version_order(tmp_path: Path, dataset_dirs):
    cases = _case_bundle()
    mapping: Mapping[str, Path] = _dataset_mapping(cases, dataset_dirs)
    coordinator = PairedRunCoordinator(
        provider=_provider(None),
        run_settings=RunSettings(),
        transport=_ImmediateAnswer(),
    )

    result = coordinator.run(
        mode="formal",
        cases=cases,
        dataset_dirs=mapping,
        run_output_root=tmp_path,
    )

    ids = [c.case_id for c in cases.cases]
    expected = {
        case_id: (("v1", "v2") if index % 2 == 1 else ("v2", "v1"))
        for index, case_id in enumerate(ids, start=1)
    }
    assert result.version_order == expected
    assert [r.case_id for r in result.v1.records] == ids
    assert [r.case_id for r in result.v2.records] == ids
    assert result.comparison_group_id.startswith("formal__")


def test_single_case_failure_is_isolated(tmp_path: Path, dataset_dirs):
    cases = _pilot_bundle(dataset_dirs)
    mapping = {
        cases.manifest.datasets["development"].dataset_id: dataset_dirs["development"]
    }
    # call 1 = CASE_003/v1, call 2 = CASE_003/v2, call 3 = CASE_005/v1.
    coordinator = PairedRunCoordinator(
        provider=_provider(None),
        run_settings=RunSettings(),
        transport=_FailOnce(fail_on=3),
    )

    result = coordinator.run(
        mode="pilot",
        cases=cases,
        dataset_dirs=mapping,
        run_output_root=tmp_path,
    )

    assert result.v1.summary.failed == 1
    assert result.v1.summary.completed == 19
    assert result.v2.summary.completed == 20
    failed = [
        r for r in result.v1.records
        if r.status == "failed" and r.case_id == "CASE_005"
    ]
    assert len(failed) == 1
    assert failed[0].error_code == "adapter_error"
    assert failed[0].final_answer is None
    assert all(
        r.total_tokens is None for r in result.v1.records
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_real_llm_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.experiments.llm_run'`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/experiments/llm_run.py
import json
import math
import secrets
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from app.agents import AgentRunner, RunRequest
from app.agents.schemas import TraceEventType
from app.data.database import open_dataset
from app.data.manifest import file_sha256
from app.llm.openai_compatible import JsonTransport, OpenAICompatibleAdapter
from app.prompts.v1 import v1_prompt_snapshot
from app.prompts.v2 import v2_prompt_snapshot
from app.tools import build_default_registry
from app.tools.schemas import canonical_tool_result_payload

from .artifacts import DatasetArtifactIdentity, RuntimeVersions
from .baseline_v1 import (
    RunLevelError,
    dataset_source_identity,
    tool_contract_identity,
    validate_registry_contract,
)
from .cases import CaseBundle, RunnableCase
from .llm_artifacts import (
    LatencyStats,
    PublishedRealRun,
    RealArtifactWriter,
    RealCaseRunRecord,
    RealRunArtifactBundle,
    RealRunManifest,
    RealRunSummary,
    StoredPromptSnapshot,
    TokenStats,
    verify_published_real_run,
)
from .llm_settings import ProviderSettings, base_url_host
from .pilot_cases import pilot_case_set_sha256, select_pilot_cases

_SAFE_ERROR_CODES = frozenset(
    {
        "catalog_not_found",
        "unknown_tool",
        "invalid_arguments",
        "execution_failed",
        "max_steps_exceeded",
        "adapter_error",
        "adapter_http_error",
        "rate_limited",
        "response_format_error",
        "json_repair_failed",
    }
)


@dataclass(frozen=True)
class RunSettings:
    max_steps: int = 8
    concurrency: int = 1
    repeats: int = 1


@dataclass(frozen=True)
class PairedRunResult:
    comparison_group_id: str
    mode: str
    v1: PublishedRealRun
    v2: PublishedRealRun
    v1_path: Path
    v2_path: Path
    version_order: dict[str, tuple[str, ...]]
    case_ids: tuple[str, ...]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _timestamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%SZ")


def _group_id(mode: str, started_at: datetime) -> str:
    return f"{mode}__{_timestamp(started_at)}__{secrets.token_hex(4)}"


def _run_id(version: str, mode: str, started_at: datetime) -> str:
    if mode == "pilot":
        prefix = "real-v1-pilot" if version == "v1" else "real-v2-pilot"
    else:
        prefix = "real-v1" if version == "v1" else "real-v2"
    return f"{prefix}__{_timestamp(started_at)}__{secrets.token_hex(4)}"


def _subset_sha256(ids: tuple[str, ...]) -> str:
    payload = json.dumps(
        {"case_ids": list(ids)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _git_identity() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _runtime_versions() -> RuntimeVersions:
    from importlib.metadata import version

    return RuntimeVersions(
        python=sys.version.split()[0],
        pydantic=version("pydantic"),
        duckdb=version("duckdb"),
        pandas=version("pandas"),
        pyarrow=version("pyarrow"),
    )


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _error_code(result) -> str:
    for event in reversed(result.decision_trace):
        if event.event_type is TraceEventType.ERROR:
            code = event.payload.get("code")
            if isinstance(code, str) and code in _SAFE_ERROR_CODES:
                return code
    return "case_execution_error"


def _record_from_result(
    *,
    sequence: int,
    case: RunnableCase,
    agent_version: str,
    result,
    latency_ms: int,
) -> RealCaseRunRecord:
    if result.status == "completed":
        status = "completed"
        final_answer = result.final_answer
        error_code = None
    else:
        status = "failed"
        final_answer = None
        error_code = _error_code(result)
    prior_tool_executions = tuple(
        {
            "call_id": execution.call_id,
            "arguments": execution.model_dump(mode="json")["arguments"],
            "result": canonical_tool_result_payload(execution.result),
        }
        for execution in result.prior_tool_executions
    )
    decision_trace = tuple(
        event.model_dump(mode="json")
        for event in result.decision_trace
        if status == "completed" or event.event_type is not TraceEventType.ERROR
    )
    usage = result.usage
    return RealCaseRunRecord(
        sequence=sequence,
        case_id=case.case_id,
        split=case.split,
        dataset_id=case.dataset_id,
        dataset_version=case.dataset_version,
        agent_version=agent_version,
        repeat_index=1,
        user_input=case.user_input,
        status=status,
        final_answer=final_answer,
        error_code=error_code,
        prior_tool_executions=prior_tool_executions,
        decision_trace=decision_trace,
        latency_ms=latency_ms,
        prompt_tokens=None if usage is None else usage.prompt_tokens,
        completion_tokens=None if usage is None else usage.completion_tokens,
        total_tokens=None if usage is None else usage.total_tokens,
    )


def _failed_record(
    *,
    sequence: int,
    case: RunnableCase,
    agent_version: str,
    latency_ms: int,
) -> RealCaseRunRecord:
    return RealCaseRunRecord(
        sequence=sequence,
        case_id=case.case_id,
        split=case.split,
        dataset_id=case.dataset_id,
        dataset_version=case.dataset_version,
        agent_version=agent_version,
        repeat_index=1,
        user_input=case.user_input,
        status="failed",
        final_answer=None,
        error_code="case_execution_error",
        prior_tool_executions=(),
        decision_trace=(),
        latency_ms=latency_ms,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )


def _build_summary(records: tuple[RealCaseRunRecord, ...]) -> RealRunSummary:
    statuses = Counter(record.status for record in records)
    splits = Counter(record.split for record in records)
    split_status = Counter((record.split, record.status) for record in records)
    latencies = [record.latency_ms for record in records]
    with_usage = [record for record in records if record.total_tokens is not None]
    tokens_available = bool(with_usage)
    tokens = TokenStats(
        available=tokens_available,
        prompt_total=(
            sum(record.prompt_tokens for record in with_usage)
            if tokens_available
            else None
        ),
        completion_total=(
            sum(record.completion_tokens for record in with_usage)
            if tokens_available
            else None
        ),
        total_total=(
            sum(record.total_tokens for record in with_usage)
            if tokens_available
            else None
        ),
        p50=(
            _percentile([record.total_tokens for record in with_usage], 0.5)
            if tokens_available
            else None
        ),
        p95=(
            _percentile([record.total_tokens for record in with_usage], 0.95)
            if tokens_available
            else None
        ),
        sample_size=len(with_usage),
    )
    return RealRunSummary(
        total=len(records),
        completed=statuses["completed"],
        failed=statuses["failed"],
        stopped=0,
        split_counts={
            "development": splits["development"],
            "public_validation": splits["public_validation"],
        },
        split_status_counts={
            "development": {
                "completed": split_status[("development", "completed")],
                "failed": split_status[("development", "failed")],
            },
            "public_validation": {
                "completed": split_status[("public_validation", "completed")],
                "failed": split_status[("public_validation", "failed")],
            },
        },
        first_case_id=records[0].case_id if records else None,
        last_case_id=records[-1].case_id if records else None,
        latency=LatencyStats(
            p50=_percentile(latencies, 0.5),
            p95=_percentile(latencies, 0.95),
            sample_size=len(records),
        ),
        tokens=tokens,
        cost="unavailable",
        evaluation_status="pending_not_run",
    )


def _dataset_identities(
    selected: tuple[RunnableCase, ...],
    dataset_dirs: Mapping[str, Path],
) -> tuple[DatasetArtifactIdentity, ...]:
    identities = []
    seen = set()
    for case in selected:
        if case.dataset_id in seen:
            continue
        seen.add(case.dataset_id)
        directory = dataset_dirs[case.dataset_id]
        identities.append(
            DatasetArtifactIdentity(
                split=case.split,
                dataset_id=case.dataset_id,
                dataset_version=case.dataset_version,
                manifest_sha256=file_sha256(directory / "manifest.json"),
            )
        )
    return tuple(identities)


class PairedRunCoordinator:
    def __init__(
        self,
        *,
        provider: ProviderSettings,
        run_settings: RunSettings = RunSettings(),
        transport: JsonTransport | None = None,
        tool_contract_path: Path = Path("configs/evaluation/tool_contract_v1.yaml"),
    ) -> None:
        self._provider = provider
        self._run_settings = run_settings
        self._transport = transport
        self._tool_contract_path = tool_contract_path

    def _execute(
        self,
        *,
        prompt,
        case: RunnableCase,
        sequence: int,
        agent_version: str,
        dataset_dir: Path,
    ) -> RealCaseRunRecord:
        started_ns = time.perf_counter_ns()
        try:
            with open_dataset(dataset_dir) as catalog:
                adapter = OpenAICompatibleAdapter(
                    base_url=self._provider.base_url,
                    api_key=self._provider.api_key,
                    model=self._provider.model,
                    temperature=self._provider.temperature,
                    timeout_seconds=self._provider.timeout_seconds,
                    system_prompt=prompt.prompt_text,
                    transport=self._transport,
                )
                runner = AgentRunner(build_default_registry(), adapter)
                result = runner.run(
                    RunRequest(
                        user_input=case.user_input,
                        max_steps=self._run_settings.max_steps,
                    ),
                    catalog,
                )
        except Exception:
            latency_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
            return _failed_record(
                sequence=sequence,
                case=case,
                agent_version=agent_version,
                latency_ms=latency_ms,
            )
        latency_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
        return _record_from_result(
            sequence=sequence,
            case=case,
            agent_version=agent_version,
            result=result,
            latency_ms=latency_ms,
        )

    def run(
        self,
        *,
        mode: str,
        cases: CaseBundle,
        dataset_dirs: Mapping[str, Path],
        run_output_root: Path,
    ) -> PairedRunResult:
        expected_datasets = cases.manifest.datasets
        for case in cases.cases:
            expected = expected_datasets[case.split]
            if (
                case.dataset_id != expected.dataset_id
                or case.dataset_version != expected.dataset_version
            ):
                raise RunLevelError("dataset_mapping_error")
        mapped_ids = set(dataset_dirs)
        required_ids = {
            expected_datasets[split].dataset_id for split in expected_datasets
        }
        if mapped_ids != required_ids:
            raise RunLevelError("dataset_mapping_error")

        contract = tool_contract_identity(self._tool_contract_path)
        if (
            cases.manifest.tool_contract.version != contract.version
            or cases.manifest.tool_contract.sha256 != contract.sha256
        ):
            raise RunLevelError("tool_contract_identity_mismatch")
        validate_registry_contract(contract)

        started_at = _utc_now()
        initial_dataset_identity = dataset_source_identity(dataset_dirs)

        if mode == "pilot":
            selected = select_pilot_cases(cases)
            subset_id = "pilot-20-v1"
            subset_sha = pilot_case_set_sha256()
        else:
            selected = cases.cases
            subset_id = "all-100-v1"
            subset_sha = _subset_sha256(tuple(case.case_id for case in selected))

        prompts = {
            "v1": v1_prompt_snapshot(),
            "v2": v2_prompt_snapshot(),
        }
        original_index = {
            case.case_id: index
            for index, case in enumerate(cases.cases, start=1)
        }
        v1_records: list[RealCaseRunRecord] = []
        v2_records: list[RealCaseRunRecord] = []
        version_order: dict[str, tuple[str, ...]] = {}
        for sequence, case in enumerate(selected, start=1):
            if mode == "formal":
                order = (
                    ("v1", "v2")
                    if original_index[case.case_id] % 2 == 1
                    else ("v2", "v1")
                )
            else:
                order = ("v1", "v2")
            version_order[case.case_id] = order
            for agent_version in order:
                record = self._execute(
                    prompt=prompts[agent_version],
                    case=case,
                    sequence=sequence,
                    agent_version=agent_version,
                    dataset_dir=dataset_dirs[case.dataset_id],
                )
                if agent_version == "v1":
                    v1_records.append(record)
                else:
                    v2_records.append(record)

        if dataset_source_identity(dataset_dirs) != initial_dataset_identity:
            raise RunLevelError("dataset_identity_drift")
        if tool_contract_identity(self._tool_contract_path) != contract:
            raise RunLevelError("tool_contract_identity_drift")
        completed_at = _utc_now()

        git_commit, git_dirty = _git_identity()
        runtime = _runtime_versions()
        case_ids = tuple(case.case_id for case in selected)
        dataset_identities = _dataset_identities(selected, dataset_dirs)
        group_id = _group_id(mode, started_at)

        def build_bundle(
            agent_version: str,
            records: list[RealCaseRunRecord],
        ) -> RealRunArtifactBundle:
            run_id = _run_id(agent_version, mode, started_at)
            prompt = prompts[agent_version]
            usage_available = any(record.total_tokens is not None for record in records)
            manifest = RealRunManifest(
                artifact_schema_version="2.0.0",
                run_id=run_id,
                comparison_group_id=group_id,
                experiment_mode=mode,
                agent_version=agent_version,
                started_at_utc=_utc_text(started_at),
                completed_at_utc=_utc_text(completed_at),
                git_commit=git_commit,
                git_dirty=git_dirty,
                runtime_versions=runtime,
                case_set_id=cases.manifest.case_set_id,
                case_jsonl_sha256=cases.manifest.jsonl_sha256,
                case_manifest_sha256=cases.manifest_sha256,
                case_subset_id=subset_id,
                case_subset_sha256=subset_sha,
                datasets=dataset_identities,
                tool_contract_version=contract.version,
                tool_contract_sha256=contract.sha256,
                adapter_name="openai-compatible",
                model=self._provider.model,
                base_url_host=base_url_host(self._provider.base_url),
                temperature=self._provider.temperature,
                timeout_seconds=self._provider.timeout_seconds,
                max_steps=self._run_settings.max_steps,
                concurrency=self._run_settings.concurrency,
                repeats=self._run_settings.repeats,
                prompt={
                    "prompt_name": prompt.prompt_name,
                    "prompt_version": prompt.prompt_version,
                    "source_file": prompt.source_file,
                    "prompt_sha256": prompt.prompt_sha256,
                },
                output_files=None,
                evaluation_status="pending_not_run",
                usage_status="available" if usage_available else "unavailable",
                cost_status="unavailable",
            )
            stored_prompt = StoredPromptSnapshot(
                prompt_name=prompt.prompt_name,
                prompt_version=prompt.prompt_version,
                prompt_text=prompt.prompt_text,
                prompt_sha256=prompt.prompt_sha256,
                change_note=prompt.change_note,
                source_file=prompt.source_file,
            )
            return RealRunArtifactBundle(
                records=tuple(records),
                expected_case_ids=case_ids,
                summary=_build_summary(tuple(records)),
                prompt_snapshot=stored_prompt,
                manifest=manifest,
            )

        writer = RealArtifactWriter(run_output_root)
        staged_v1 = None
        staged_v2 = None
        try:
            staged_v1 = writer.stage(build_bundle("v1", v1_records))
            staged_v2 = writer.stage(build_bundle("v2", v2_records))
            v1_path = writer.commit(staged_v1)
            v2_path = writer.commit(staged_v2)
        except BaseException:
            if staged_v1 is not None:
                shutil.rmtree(staged_v1.temp_dir, ignore_errors=True)
            if staged_v2 is not None:
                shutil.rmtree(staged_v2.temp_dir, ignore_errors=True)
            raise

        return PairedRunResult(
            comparison_group_id=group_id,
            mode=mode,
            v1=verify_published_real_run(v1_path),
            v2=verify_published_real_run(v2_path),
            v1_path=v1_path,
            v2_path=v2_path,
            version_order=version_order,
            case_ids=case_ids,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_real_llm_run.py -v`
Expected: PASS (3 passed). Then run `python3 -m ruff check app/experiments/llm_run.py tests/test_real_llm_run.py` with no errors.

- [ ] **Step 5: Commit**

```bash
git add app/experiments/llm_run.py tests/test_real_llm_run.py
git commit -m "feat(experiments): add paired real-llm run coordinator"
```

---

### Task 10: Comparison manifest and paired fairness validation

Create the comparison artifact that binds a V1/V2 pair, records each run's record-file hash, and enforces fairness (same model/host/params/subset; distinct prompts; valid version order). A comparison is written only after re-verifying both published runs; any failed check raises and nothing is published.

**Files:**
- Create: `app/experiments/comparisons.py`
- Test: `tests/test_comparisons.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_comparison.py
from pathlib import Path

import pytest

from app.experiments.comparisons import (
    ComparisonError,
    verify_comparison,
    write_comparison,
)

from tests.test_real_llm_run import (
    _ImmediateAnswer,
    _pilot_bundle,
    _provider,
)
from app.experiments.llm_run import PairedRunCoordinator, RunSettings


def _paired(tmp_path: Path, dataset_dirs):
    cases = _pilot_bundle(dataset_dirs)
    mapping = {
        cases.manifest.datasets["development"].dataset_id: dataset_dirs["development"]
    }
    coordinator = PairedRunCoordinator(
        provider=_provider(None),
        run_settings=RunSettings(),
        transport=_ImmediateAnswer(),
    )
    runs_root = tmp_path / "runs"
    comparisons_root = tmp_path / "comparisons"
    result = coordinator.run(
        mode="pilot",
        cases=cases,
        dataset_dirs=mapping,
        run_output_root=runs_root,
    )
    return result, comparisons_root


def test_comparison_published_when_fair(tmp_path: Path, dataset_dirs):
    result, comparisons_root = _paired(tmp_path, dataset_dirs)

    path = write_comparison(result, comparisons_root=comparisons_root)

    published = verify_comparison(path)
    assert published.comparison_group_id == result.comparison_group_id
    assert published.v1_run_id == result.v1.manifest.run_id
    assert published.v2_run_id == result.v2.manifest.run_id
    assert published.case_ids == result.case_ids
    assert published.fairness.passed is True
    assert published.fairness.failures == ()


def test_comparison_rejects_tampered_version_order(tmp_path: Path, dataset_dirs):
    result, comparisons_root = _paired(tmp_path, dataset_dirs)
    # Mutate the in-memory result's order to violate the pilot rule (must be v1,v2).
    first_id = result.case_ids[0]
    result.version_order[first_id] = ("v2", "v1")

    with pytest.raises(ComparisonError) as excinfo:
        write_comparison(result, comparisons_root=comparisons_root)

    assert "version_order_valid" in str(excinfo.value)
    assert not comparisons_root.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_comparisons.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.experiments.comparisons'`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/experiments/comparisons.py
import os
import shutil
from pathlib import Path

from filelock import FileLock

from app.data.manifest import file_sha256

from .artifacts import canonical_json_bytes
from .llm_artifacts import PublishedRealRun, verify_published_real_run
from .llm_run import PairedRunResult


class ComparisonError(RuntimeError):
    pass


class FairnessReport:
    def __init__(
        self,
        checks: dict[str, bool],
    ) -> None:
        self.checks = checks
        self.passed = all(checks.values())
        self.failures = tuple(name for name, ok in checks.items() if not ok)

    def payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "failures": list(self.failures),
        }


def _version_order_valid(mode: str, version_order: dict[str, tuple[str, ...]]) -> bool:
    if mode == "pilot":
        return all(order == ("v1", "v2") for order in version_order.values())
    orders = list(version_order.values())
    if any(set(order) != {"v1", "v2"} for order in orders):
        return False
    alternates = [order == ("v1", "v2") for order in orders]
    return all(value == (index % 2 == 0) for index, value in enumerate(alternates))


def _build_fairness(
    v1: PublishedRealRun,
    v2: PublishedRealRun,
    *,
    case_ids: tuple[str, ...],
    version_order: dict[str, tuple[str, ...]],
) -> FairnessReport:
    m1 = v1.manifest
    m2 = v2.manifest
    v1_ids = tuple(record.case_id for record in v1.records)
    v2_ids = tuple(record.case_id for record in v2.records)
    checks = {
        "group_ids_match": m1.comparison_group_id == m2.comparison_group_id,
        "mode_match": m1.experiment_mode == m2.experiment_mode,
        "model_match": m1.model == m2.model,
        "base_url_host_match": m1.base_url_host == m2.base_url_host,
        "temperature_match": m1.temperature == m2.temperature,
        "timeout_match": m1.timeout_seconds == m2.timeout_seconds,
        "max_steps_match": m1.max_steps == m2.max_steps,
        "case_subset_id_match": m1.case_subset_id == m2.case_subset_id,
        "case_subset_sha_match": m1.case_subset_sha256 == m2.case_subset_sha256,
        "record_count_match": len(v1.records) == len(v2.records),
        "case_ids_match": v1_ids == case_ids and v2_ids == case_ids,
        "version_order_valid": _version_order_valid(m1.experiment_mode, version_order),
        "prompts_distinct": (
            m1.prompt["prompt_sha256"] != m2.prompt["prompt_sha256"]
        ),
    }
    return FairnessReport(checks)


def _comparison_payload(
    result: PairedRunResult,
    v1: PublishedRealRun,
    v2: PublishedRealRun,
    fairness: FairnessReport,
) -> dict[str, object]:
    return {
        "artifact_schema_version": "1.0.0",
        "comparison_group_id": result.comparison_group_id,
        "experiment_mode": result.mode,
        "v1_run_id": v1.manifest.run_id,
        "v2_run_id": v2.manifest.run_id,
        "case_subset_id": v1.manifest.case_subset_id,
        "case_subset_sha256": v1.manifest.case_subset_sha256,
        "case_ids": list(result.case_ids),
        "v1_case_runs_sha256": file_sha256(result.v1_path / "case_runs.jsonl"),
        "v2_case_runs_sha256": file_sha256(result.v2_path / "case_runs.jsonl"),
        "version_order": {
            case_id: list(order) for case_id, order in result.version_order.items()
        },
        "fairness": fairness.payload(),
    }


def write_comparison(
    result: PairedRunResult,
    *,
    comparisons_root: Path,
) -> Path:
    v1 = verify_published_real_run(result.v1_path)
    v2 = verify_published_real_run(result.v2_path)
    fairness = _build_fairness(
        v1,
        v2,
        case_ids=result.case_ids,
        version_order=result.version_order,
    )
    if not fairness.passed:
        raise ComparisonError(f"fairness checks failed: {', '.join(fairness.failures)}")

    comparisons_root.mkdir(parents=True, exist_ok=True)
    target = comparisons_root / result.comparison_group_id
    lock = FileLock(comparisons_root / ".comparison-publish.lock")
    with lock:
        if target.exists():
            raise ComparisonError("comparison already published")
        staging = comparisons_root / f".{result.comparison_group_id}.staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            payload = _comparison_payload(result, v1, v2, fairness)
            (staging / "comparison.json").write_bytes(
                canonical_json_bytes(payload)
            )
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return target


def verify_comparison(path: Path) -> object:
    file_path = path / "comparison.json"
    if not file_path.is_file():
        raise ComparisonError("comparison.json missing")
    import json

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not payload.get("fairness", {}).get("passed", False):
        raise ComparisonError("published comparison did not pass fairness")
    from types import SimpleNamespace

    return SimpleNamespace(**payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_comparisons.py -v`
Expected: PASS (2 passed). Then `python3 -m ruff check app/experiments/comparisons.py tests/test_comparisons.py` with no errors.

- [ ] **Step 5: Commit**

```bash
git add app/experiments/comparisons.py tests/test_comparisons.py
git commit -m "feat(experiments): add paired comparison manifest and fairness checks"
```

---

### Task 11: Pilot/formal CLI, pilot technical gate, console scripts and Makefile targets

Add a thin CLI that loads provider settings from the environment, runs the paired coordinator, writes the comparison, and (pilot only) reports the technical gate (per-version failure rate ≤ 10%). Wire two console scripts and Makefile targets. The analysis console scripts (`evaluate-runs`, `analyze-errors`, `aggregate-statistics`) are added in the batches that create their target modules so no entry point is ever dangling.

**Files:**
- Create: `app/experiments/llm_cli.py`
- Test: `tests/test_llm_cli.py`
- Modify: `pyproject.toml`
- Modify: `Makefile`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_cli.py
from types import SimpleNamespace

import pytest

from app.experiments.llm_cli import formal_main, pilot_gate, pilot_main


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_TEMPERATURE",
        "LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_pilot_main_missing_provider_config_returns_1(capsys: pytest.CaptureFixture) -> None:
    assert pilot_main([]) == 1
    assert "provider_config_error" in capsys.readouterr().err


def test_formal_main_missing_provider_config_returns_1(capsys: pytest.CaptureFixture) -> None:
    assert formal_main([]) == 1
    assert "provider_config_error" in capsys.readouterr().err


def _records(failed: int, total: int = 20):
    return (
        [SimpleNamespace(status="failed")] * failed
        + [SimpleNamespace(status="completed")] * (total - failed)
    )


def test_pilot_gate_allows_at_most_two_failures() -> None:
    ok, rate = pilot_gate(_records(2))
    assert ok is True
    assert rate == pytest.approx(0.10)

    ok, rate = pilot_gate(_records(3))
    assert ok is False
    assert rate == pytest.approx(0.15)


def test_pilot_gate_rejects_empty_record_set() -> None:
    ok, _ = pilot_gate([])
    assert ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_llm_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.experiments.llm_cli'`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/experiments/llm_cli.py
import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .artifacts import canonical_json_bytes
from .baseline_v1 import RunLevelError
from .cases import load_runnable_cases
from .comparisons import ComparisonError, write_comparison
from .llm_run import PairedRunCoordinator, RunSettings
from .llm_settings import ProviderConfigError, load_provider_settings

DEFAULT_CASE_DIR = Path("data/evaluation_cases/v1")
DEFAULT_DEVELOPMENT_DATASET = Path("data/synthetic/v1")
DEFAULT_PUBLIC_VALIDATION_DATASET = Path("data/synthetic/public-validation-v1")
DEFAULT_RUN_OUTPUT_ROOT = Path("outputs/real_runs")
DEFAULT_COMPARISONS_ROOT = Path("outputs/paired_comparisons")
PILOT_FAILURE_RATE_LIMIT = 0.10


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real-llm V1/V2 paired experiment"
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument(
        "--development-dataset",
        type=Path,
        default=DEFAULT_DEVELOPMENT_DATASET,
    )
    parser.add_argument(
        "--public-validation-dataset",
        type=Path,
        default=DEFAULT_PUBLIC_VALIDATION_DATASET,
    )
    parser.add_argument(
        "--run-output-root", type=Path, default=DEFAULT_RUN_OUTPUT_ROOT
    )
    parser.add_argument(
        "--comparisons-root", type=Path, default=DEFAULT_COMPARISONS_ROOT
    )
    return parser


def pilot_gate(records: Sequence[object]) -> tuple[bool, float]:
    total = len(records)
    if total == 0:
        return False, 1.0
    failed = sum(1 for record in records if record.status != "completed")
    rate = failed / total
    return rate <= PILOT_FAILURE_RATE_LIMIT, rate


def _dataset_dirs(cases, args) -> dict[str, Path]:
    return {
        cases.manifest.datasets["development"].dataset_id: args.development_dataset,
        cases.manifest.datasets["public_validation"].dataset_id: (
            args.public_validation_dataset
        ),
    }


def _run(mode: str, argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    try:
        provider = load_provider_settings(os.environ)
        cases = load_runnable_cases(args.case_dir)
        coordinator = PairedRunCoordinator(
            provider=provider,
            run_settings=RunSettings(),
        )
        result = coordinator.run(
            mode=mode,
            cases=cases,
            dataset_dirs=_dataset_dirs(cases, args),
            run_output_root=args.run_output_root,
        )
        comparison_path = write_comparison(
            result,
            comparisons_root=args.comparisons_root,
        )
    except ProviderConfigError as exc:
        print(f"provider_config_error: {exc.code}", file=sys.stderr)
        return 1
    except RunLevelError as exc:
        print(getattr(exc, "code", "run_level_error"), file=sys.stderr)
        return 1
    except ComparisonError as exc:
        print(f"comparison_error: {exc}", file=sys.stderr)
        return 1

    payload = {
        "comparison_group_id": result.comparison_group_id,
        "mode": mode,
        "v1_run_id": result.v1.manifest.run_id,
        "v2_run_id": result.v2.manifest.run_id,
        "comparison_path": str(comparison_path),
    }
    if mode == "pilot":
        v1_ok, v1_rate = pilot_gate(result.v1.records)
        v2_ok, v2_rate = pilot_gate(result.v2.records)
        payload["pilot_gate"] = {
            "failure_rate_limit": PILOT_FAILURE_RATE_LIMIT,
            "v1_failure_rate": v1_rate,
            "v2_failure_rate": v2_rate,
            "passed": v1_ok and v2_ok,
        }
        sys.stdout.write(canonical_json_bytes(payload).decode("utf-8"))
        return 0 if (v1_ok and v2_ok) else 2
    sys.stdout.write(canonical_json_bytes(payload).decode("utf-8"))
    return 0


def pilot_main(argv: Sequence[str] | None = None) -> int:
    return _run("pilot", argv)


def formal_main(argv: Sequence[str] | None = None) -> int:
    return _run("formal", argv)


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    main = formal_main if command == "formal" else pilot_main
    argv = sys.argv[2:] if command in {"pilot", "formal"} else sys.argv[1:]
    raise SystemExit(main(argv))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_llm_cli.py -v`
Expected: PASS (5 passed). Then `python3 -m ruff check app/experiments/llm_cli.py tests/test_llm_cli.py` with no errors.

- [ ] **Step 5: Register console scripts**

In `pyproject.toml`, extend `[project.scripts]` so it reads:

```toml
[project.scripts]
build-phase1 = "app.data.phase1:main"
phase2-smoke = "app.experiments.phase2_smoke:main"
phase3-baseline = "app.experiments.baseline_v1:main"
real-llm-pilot = "app.experiments.llm_cli:pilot_main"
real-llm-formal = "app.experiments.llm_cli:formal_main"
```

Also append these two lines to `.gitignore` so real-run artifacts are never committed:

```gitignore
outputs/real_runs/
outputs/paired_comparisons/
```

- [ ] **Step 6: Add Makefile targets**

Add these variables near the Phase 3 variables in `Makefile`:

```make
REAL_RUN_CASE_DIR ?= data/evaluation_cases/v1
REAL_RUN_DEVELOPMENT_DATASET ?= data/synthetic/v1
REAL_RUN_PUBLIC_VALIDATION_DATASET ?= data/synthetic/public-validation-v1
REAL_RUN_OUTPUT_ROOT ?= outputs/real_runs
REAL_RUN_COMPARISONS_ROOT ?= outputs/paired_comparisons
```

Append these targets at the end of `Makefile`:

```make
real-llm-pilot: check-python
	@"$(PYTHON)" -m app.experiments.llm_cli --case-dir "$(REAL_RUN_CASE_DIR)" \
		--development-dataset "$(REAL_RUN_DEVELOPMENT_DATASET)" \
		--public-validation-dataset "$(REAL_RUN_PUBLIC_VALIDATION_DATASET)" \
		--run-output-root "$(REAL_RUN_OUTPUT_ROOT)" \
		--comparisons-root "$(REAL_RUN_COMPARISONS_ROOT)" ; \
	code=$$?; if [ $$code -eq 2 ]; then echo "pilot technical gate failed; do not run formal" >&2; fi; exit $$code

real-llm-formal: check-python
	@"$(PYTHON)" -m app.experiments.llm_cli --case-dir "$(REAL_RUN_CASE_DIR)" \
		--development-dataset "$(REAL_RUN_DEVELOPMENT_DATASET)" \
		--public-validation-dataset "$(REAL_RUN_PUBLIC_VALIDATION_DATASET)" \
	--run-output-root "$(REAL_RUN_OUTPUT_ROOT)" \
	--comparisons-root "$(REAL_RUN_COMPARISONS_ROOT)"
```

Add `real-llm-pilot real-llm-formal` to the single `.PHONY:` declaration.

- [ ] **Step 7: Reinstall and verify wiring**

Run: `python3 -m pip install --no-deps --editable .`
Then verify help works offline: `real-llm-pilot --help` and `real-llm-formal --help` (exit 0).
Run the full suite: `python3 -m pytest` (no regressions).

- [ ] **Step 8: Commit**

```bash
git add app/experiments/llm_cli.py tests/test_llm_cli.py pyproject.toml Makefile .gitignore
git commit -m "feat(experiments): add real-llm paired CLI, pilot gate and targets"
```

---

### Task 12: Execute the real-model Pilot (human-supervised gate)

Run the 20-Case pilot against the real Volcano Ark model. This is a decision gate, not a code task: it requires real credentials supplied only in the shell, produces git-ignored artifacts, and pauses for human review before any Formal run.

**Files:**
- No source changes (artifacts written under git-ignored `outputs/real_runs/` and `outputs/paired_comparisons/`).

- [ ] **Step 1: Confirm preconditions**

Run: `make test` and `make lint` (both clean).
Confirm the frozen assets exist:
- cases: `data/evaluation_cases/v1/cases.jsonl` and `manifest.json`
- datasets: `data/synthetic/v1/` and `data/synthetic/public-validation-v1/` (run `make phase1` if missing)
Confirm the working tree is clean and HEAD is the Batch A commit: `git status --porcelain` empty.

- [ ] **Step 2: Inject credentials without exposing them**

In the shell that will run the pilot, provide the key via hidden input. Never paste the key into the agent chat, into code, or into artifacts.

```bash
read -rs LLM_API_KEY; export LLM_API_KEY
export LLM_MODEL="<your-ark-endpoint-id>"   # non-secret endpoint id
# optional: export LLM_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
```

Verify the configuration parses without printing the secret (the key is never echoed):

```bash
python3 -c "import os; from app.experiments.llm_settings import load_provider_settings; s=load_provider_settings(os.environ); print(s.model, s.temperature)"
```

Expected: prints `<endpoint-id> 0.0`.

- [ ] **Step 3: Run the pilot**

Run: `real-llm-pilot | tee .tmp/pilot-result.json`
Expected exit code: `0` (gate passed). Exit code `2` means the technical gate failed — stop and investigate; do not run formal.

- [ ] **Step 4: Verify the paired artifacts**

Confirm the printed JSON contains:
- `comparison_group_id` starting with `pilot__`
- distinct `v1_run_id` / `v2_run_id`
- `pilot_gate.passed = true`, with `v1_failure_rate` and `v2_failure_rate` each `≤ 0.10`

Independently verify on disk (both runs re-verify and the comparison passes fairness):

```bash
ls outputs/real_runs
ls outputs/paired_comparisons
wc -l outputs/real_runs/*/case_runs.jsonl   # 20 lines each
```

- [ ] **Step 5: Human review checkpoint — pause**

Inspect a small sample of V1/V2 answers (including any failed Cases) and confirm:
- V2 answers that cite evidence reference a runtime `result_id` present in that Case's own `prior_tool_executions` (never a Gold id).
- Failures are understood and acceptable at the ≤ 10% gate.

**Do not proceed to Batch B/Formal until the human explicitly approves.** Record the approved `comparison_group_id` in the handoff.

---

## Batch B — Scoring and analysis protocol freeze

### Task 13: Freeze rubric, error taxonomy and statistics configs with SHA loader

Create three versioned YAML configs and a loader that returns each config's version and SHA-256 plus a freeze manifest. These are frozen before the Formal run and never adjusted against Formal results.

**Files:**
- Create: `configs/evaluation/rubric_v1.yaml`
- Create: `configs/evaluation/error_taxonomy_v1.yaml`
- Create: `configs/evaluation/statistics_v1.yaml`
- Create: `app/evaluation/__init__.py`
- Create: `app/evaluation/freeze.py`
- Test: `tests/test_evaluation_freeze.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation_freeze.py
from pathlib import Path

import pytest

from app.evaluation.freeze import (
    FreezeError,
    freeze_manifest,
    load_frozen_configs,
)

ROOT = Path(__file__).parents[1]


def test_all_three_configs_load_with_version_and_stable_sha() -> None:
    configs = {config.name: config for config in load_frozen_configs()}
    assert set(configs) == {"rubric", "error_taxonomy", "statistics"}
    for name, config in configs.items():
        assert config.version == "1.0.0"
        assert len(config.sha256) == 64

    again = {config.name: config for config in load_frozen_configs()}
    for name in configs:
        assert again[name].sha256 == configs[name].sha256


def test_freeze_manifest_lists_every_asset() -> None:
    manifest = freeze_manifest()
    assert manifest["schema_version"] == "1.0.0"
    assets = {asset["name"] for asset in manifest["assets"]}
    assert {"rubric", "error_taxonomy", "statistics"} <= assets
    for asset in manifest["assets"]:
        assert len(asset["sha256"]) == 64


def test_loader_rejects_missing_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "rubric_v1.yaml"
    bad.write_text("scoring_dimensions: []\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.evaluation.freeze.FROZEN_CONFIG_PATHS",
        {"rubric": bad},
    )
    with pytest.raises(FreezeError):
        load_frozen_configs()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evaluation_freeze.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation'`.

- [ ] **Step 3: Create the configs**

```yaml
# configs/evaluation/rubric_v1.yaml
version: "1.0.0"

metric_aliases:
  current_gmv: ["current_gmv", "本期销售额", "本期gmv", "本期成交"]
  previous_gmv: ["previous_gmv", "上期销售额", "上期gmv"]
  gmv_change_rate: ["gmv_change_rate", "销售额变化率", "gmv变化率", "营收变化率"]
  current_orders: ["current_orders", "本期订单", "本期订单数"]
  previous_orders: ["previous_orders", "上期订单", "上期订单数"]
  current_aov: ["current_aov", "本期客单价"]
  previous_aov: ["previous_aov", "上期客单价"]
  aov_change_rate: ["aov_change_rate", "客单价变化率"]
  current_cvr: ["current_cvr", "本期转化率", "本期cvr"]
  previous_cvr: ["previous_cvr", "上期转化率", "上期cvr"]
  cvr_change_rate: ["cvr_change_rate", "转化率变化率", "cvr变化率"]
  traffic_change_rate: ["traffic_change_rate", "流量变化率"]
  refund_rate: ["refund_rate", "退款率"]
  evidence_value: ["evidence_value", "证据强度", "优先级分值"]

aov_direction_aliases: ["aov_direction", "客单价方向"]

core_facts:
  gmv_diagnosis:
    - current_gmv
    - previous_gmv
    - gmv_change_rate
    - current_orders
    - aov_direction
  product_anomaly:
    - target_product
    - gmv_change_rate
    - traffic_change_rate
    - current_cvr
    - refund_rate
  conversion_drop:
    - target_product
    - current_cvr
    - previous_cvr
    - cvr_change_rate
    - observed_days
  products_to_watch:
    - target_products
    - gmv_signal
    - cvr_signal
    - refund_signal
    - evidence_order
  next_week_priorities:
    - target_products
    - evidence_value
    - action_recommendation
    - limitation
  data_insufficient:
    - observed_days
    - missing_scope
    - narrowest_supported_conclusion

scoring_dimensions:
  - execution_status
  - tool_selection
  - parameter_accuracy
  - factual_accuracy
  - required_output_completeness
  - evidence_binding
  - unsupported_claim
  - instruction_following
  - data_insufficiency_handling
  - recommendation_support

task_success_gate:
  - execution_status
  - correct_tool
  - valid_parameters
  - correct_core_facts
  - no_critical_unsupported_claim
  - required_output_complete

success_criteria_map:
  correct_tool: tool_selection
  valid_parameters: parameter_accuracy
  correct_core_facts: factual_accuracy
  no_critical_unsupported_claim: unsupported_claim
  required_output_complete: required_output_completeness

evidence_binding:
  applies_to_versions: ["v2"]
  other_versions_state: not_applicable
```

```yaml
# configs/evaluation/error_taxonomy_v1.yaml
version: "1.0.0"

errors:
  E1: {name: "Tool Selection Error"}
  E2: {name: "Parameter Error"}
  E3: {name: "Data Retrieval Error"}
  E4: {name: "Calculation Error"}
  E5: {name: "Reasoning Error"}
  E6: {name: "Hallucination"}
  E7: {name: "Instruction Following Error"}
  E8: {name: "Incomplete Answer"}
  E9: {name: "Unsupported Recommendation"}
  E10: {name: "Data Insufficiency Handling Error"}

# Maps a failed scoring dimension to its candidate error code. The earliest causal
# failure along the decision trace wins; this table is the dimension -> code fallback.
dimension_error_map:
  tool_selection: E1
  parameter_accuracy: E2
  factual_accuracy: E4
  required_output_completeness: E8
  evidence_binding: E6
  unsupported_claim: E6
  instruction_following: E7
  data_insufficiency_handling: E10
  recommendation_support: E9

# Causal precedence when two causes both explain the failure (trace order first).
causal_precedence: [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10]
```

```yaml
# configs/evaluation/statistics_v1.yaml
version: "1.0.0"

cluster_field: statistical_cluster_id
resample_unit: statistical_cluster_id
bootstrap:
  iterations: 10000
  seed: 20261009
  confidence: 0.95
delta:
  direction: v2_minus_v1
mcnemar:
  method: exact
  cluster_success_rule: both_cases_pass
  zero_discordant_state: not_applicable
descriptive_only_below_clusters: 10
```

- [ ] **Step 4: Write the loader**

```python
# app/evaluation/__init__.py
```

```python
# app/evaluation/freeze.py
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).parents[2]

FROZEN_CONFIG_PATHS: dict[str, Path] = {
    "rubric": _PROJECT_ROOT / "configs/evaluation/rubric_v1.yaml",
    "error_taxonomy": _PROJECT_ROOT / "configs/evaluation/error_taxonomy_v1.yaml",
    "statistics": _PROJECT_ROOT / "configs/evaluation/statistics_v1.yaml",
}


class FreezeError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenConfig:
    name: str
    path: Path
    version: str
    sha256: str


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise FreezeError(f"cannot read frozen config {path}") from exc
    if not isinstance(data, dict):
        raise FreezeError(f"frozen config {path} must be a mapping")
    version = data.get("version")
    if not isinstance(version, str) or not version:
        raise FreezeError(f"frozen config {path} missing string version")
    return data


def load_frozen_configs(
    paths: dict[str, Path] | None = None,
) -> tuple[FrozenConfig, ...]:
    config_paths = FROZEN_CONFIG_PATHS if paths is None else paths
    configs = []
    for name, path in config_paths.items():
        _load_yaml(path)
        configs.append(
            FrozenConfig(
                name=name,
                path=path,
                version=str(_load_yaml(path)["version"]),
                sha256=sha256(path.read_bytes()).hexdigest(),
            )
        )
    return tuple(configs)


def freeze_manifest() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "assets": [
            {
                "name": config.name,
                "path": str(config.path.relative_to(_PROJECT_ROOT)),
                "version": config.version,
                "sha256": config.sha256,
            }
            for config in load_frozen_configs()
        ],
    }
```

- [ ] **Step 5: Run tests and lint**

Run: `python3 -m pytest tests/test_evaluation_freeze.py -v`
Expected: PASS (3 passed). Then `python3 -m ruff check app/evaluation/freeze.py tests/test_evaluation_freeze.py` with no errors.

- [ ] **Step 6: Commit**

```bash
git add configs/evaluation/rubric_v1.yaml configs/evaluation/error_taxonomy_v1.yaml \
  configs/evaluation/statistics_v1.yaml app/evaluation/__init__.py app/evaluation/freeze.py \
  tests/test_evaluation_freeze.py
git commit -m "feat(evaluation): freeze rubric, error taxonomy and statistics configs"
```

---

### Task 14: Full Gold-bearing EvaluationCase model and integrity-checked loader

The scorer needs the complete Case (Gold, expected tools/params, success criteria, cluster). Add a strict full-case model and a loader that parses the same frozen `cases.jsonl` and proves its runnable projection equals exactly what the Run process received.

**Files:**
- Create: `app/evaluation/models.py`
- Create: `app/evaluation/loader.py`
- Test: `tests/test_evaluation_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation_loader.py
from pathlib import Path

from app.evaluation.loader import EvaluationCaseBundle, load_evaluation_cases
from app.experiments.cases import load_runnable_cases

CASE_DIR = Path("data/evaluation_cases/v1")


def test_full_loader_recovers_all_cases_with_gold() -> None:
    bundle = load_evaluation_cases(CASE_DIR)
    runnable = load_runnable_cases(CASE_DIR)

    assert isinstance(bundle, EvaluationCaseBundle)
    assert len(bundle.cases) == bundle.manifest.case_count == len(runnable.cases)
    case = bundle.cases[0]
    assert case.gold_metrics
    assert case.gold_evidence[0].evidence_id
    assert case.expected_tool_calls[0].name
    assert case.success_criteria
    assert case.statistical_cluster_id


def test_full_and_runnable_projections_match() -> None:
    bundle = load_evaluation_cases(CASE_DIR)
    runnable = load_runnable_cases(CASE_DIR)

    projected = {
        case.case_id: case.to_runnable()
        for case in bundle.cases
    }
    assert set(projected) == {case.case_id for case in runnable.cases}
    for case in runnable.cases:
        assert projected[case.case_id] == case


def test_gold_fields_are_never_on_runnable_view() -> None:
    bundle = load_evaluation_cases(CASE_DIR)
    projected = bundle.cases[0].to_runnable()
    assert not hasattr(projected, "gold_metrics")
    assert not hasattr(projected, "success_criteria")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evaluation_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation.loader'`.

- [ ] **Step 3: Write the models**

```python
# app/evaluation/models.py
from pydantic import BaseModel, ConfigDict, Field

from app.experiments.cases import CaseSplit, RunnableCase


class EvaluationModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
    )


class ExpectedToolCall(EvaluationModel):
    name: str = Field(min_length=1)
    parameters: dict[str, object]


class GoldEvidence(EvaluationModel):
    evidence_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    dimensions: dict[str, object]
    metrics: dict[str, float]


class CaseMetadata(EvaluationModel):
    complexity_score: float
    semantic_family_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    top_k: int | None
    validation_role: str = Field(min_length=1)
    variant: int = Field(ge=1)


class EvaluationCase(EvaluationModel):
    case_id: str = Field(pattern=r"^CASE_[0-9]{3}$")
    case_version: str = Field(min_length=1)
    user_input: str = Field(min_length=5)
    split: CaseSplit
    dataset_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    dataset_version: str = Field(min_length=1)
    generator_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    business_task: str = Field(min_length=1)
    primary_capability: str = Field(min_length=1)
    capability_tags: tuple[str, ...]
    difficulty: str = Field(min_length=1)
    statistical_cluster_id: str = Field(min_length=1)
    expected_tool_calls: tuple[ExpectedToolCall, ...]
    allowed_alternatives: tuple[object, ...]
    expected_behavior: tuple[str, ...]
    gold_evidence: tuple[GoldEvidence, ...]
    gold_metric_evidence: dict[str, str]
    gold_metrics: dict[str, float]
    numeric_tolerances: dict[str, float]
    success_criteria: tuple[str, ...]
    reference_answer: str
    metadata: CaseMetadata
    tool_contract_version: str = Field(min_length=1)
    tool_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def to_runnable(self) -> RunnableCase:
        return RunnableCase(
            case_id=self.case_id,
            user_input=self.user_input,
            split=self.split,
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
        )
```

- [ ] **Step 4: Write the loader**

```python
# app/evaluation/loader.py
import json
from dataclasses import dataclass
from pathlib import Path

from app.experiments.cases import (
    EvaluationCaseManifest,
    _reject_duplicate_keys,
    load_runnable_cases,
)

from .models import EvaluationCase


@dataclass(frozen=True)
class EvaluationCaseBundle:
    manifest: EvaluationCaseManifest
    cases: tuple[EvaluationCase, ...]
    manifest_sha256: str


def load_evaluation_cases(case_dir: Path) -> EvaluationCaseBundle:
    runnable_bundle = load_runnable_cases(case_dir)
    raw = (case_dir / "cases.jsonl").read_bytes()
    if runnable_bundle.manifest.jsonl_sha256 != _sha256(raw):
        raise ValueError("cases.jsonl digest differs from runnable bundle manifest")

    cases = tuple(
        EvaluationCase.model_validate(
            json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        )
        for line in raw.decode("utf-8").splitlines()
        if line
    )

    # Integrity: the scorer's runnable view must equal exactly what the Agent run saw.
    projected = {case.case_id: case.to_runnable() for case in cases}
    for runnable in runnable_bundle.cases:
        if projected[runnable.case_id] != runnable:
            raise ValueError("full/runnable case projection mismatch")

    return EvaluationCaseBundle(
        manifest=runnable_bundle.manifest,
        cases=cases,
        manifest_sha256=runnable_bundle.manifest_sha256,
    )


def _sha256(raw: bytes) -> str:
    from hashlib import sha256

    return sha256(raw).hexdigest()
```

- [ ] **Step 5: Run tests and lint**

Run: `python3 -m pytest tests/test_evaluation_loader.py -v`
Expected: PASS (3 passed). Then `python3 -m ruff check app/evaluation/ tests/test_evaluation_loader.py` with no errors.

- [ ] **Step 6: Commit**

```bash
git add app/evaluation/models.py app/evaluation/loader.py tests/test_evaluation_loader.py
git commit -m "feat(evaluation): add full gold case model and integrity loader"
```

---

### Task 15: Deterministic normalization helpers

Pure functions for date, product-id, metric-set and number/percent normalization so every dimension compares canonical forms. No guessing: values that cannot be mapped return `None` and callers escalate to `needs_review`.

**Files:**
- Create: `app/evaluation/normalize.py`
- Test: `tests/test_evaluation_normalize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation_normalize.py
import pytest

from app.evaluation.normalize import (
    canonical_metric_key,
    normalize_date,
    normalize_number,
    normalize_product_ids,
    normalize_set,
    percent_to_fraction,
    within_tolerance,
)


def test_normalize_date_accepts_iso_and_derived_forms() -> None:
    assert normalize_date("2026-04-01") == "2026-04-01"
    assert normalize_date("2026/4/1") == "2026-04-01"
    assert normalize_date("not-a-date") is None


def test_normalize_product_ids_dedupes_and_sorts() -> None:
    assert normalize_product_ids(["P003", "P003", "P001"]) == ("P001", "P003")
    assert normalize_product_ids([]) == ()


def test_normalize_set_compares_as_a_set() -> None:
    assert normalize_set(["a", "b"]) == normalize_set(["b", "a"])


def test_normalize_number_handles_units_and_percent() -> None:
    assert normalize_number("1,234.5") == pytest.approx(1234.5)
    assert normalize_number("3,200,000") == pytest.approx(3_200_000)
    assert normalize_number("0.4%") == pytest.approx(0.4)
    assert normalize_number("bad") is None


def test_percent_to_fraction_round_trips() -> None:
    assert percent_to_fraction(0.4049) == pytest.approx(0.004049)


def test_canonical_metric_key_uses_alias_table() -> None:
    aliases = {"gmv_change_rate": ["销售额变化率", "gmv变化率"]}
    assert canonical_metric_key("本期销售额变化率", aliases) == "gmv_change_rate"
    assert canonical_metric_key("无关标签", aliases) is None


def test_within_tolerance_uses_absolute_case_tolerance() -> None:
    assert within_tolerance(0.0041, 0.0040, 0.001) is True
    assert within_tolerance(0.0060, 0.0040, 0.001) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evaluation_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation.normalize'`.

- [ ] **Step 3: Write the implementation**

```python
# app/evaluation/normalize.py
import re
from collections.abc import Iterable
from datetime import datetime

_DATE_PATTERNS = ("%Y-%m-%d", "%Y/%m/%d")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    for pattern in _DATE_PATTERNS:
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_product_ids(values: Iterable[object]) -> tuple[str, ...]:
    ids = {str(value).strip() for value in values if str(value).strip()}
    return tuple(sorted(ids))


def normalize_set(values: Iterable[object]) -> frozenset[str]:
    return frozenset(str(value).strip() for value in values if str(value).strip())


def normalize_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = _NUMBER_RE.search(value.replace(" ", ""))
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def percent_to_fraction(value: float) -> float:
    return value / 100.0


def canonical_metric_key(label: str, aliases: dict[str, list[str]]) -> str | None:
    text = label.strip().lower()
    for canonical, options in aliases.items():
        if any(option.lower() in text for option in options):
            return canonical
    return None


def within_tolerance(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance
```

- [ ] **Step 4: Run tests and lint**

Run: `python3 -m pytest tests/test_evaluation_normalize.py -v`
Expected: PASS (7 passed). Then `python3 -m ruff check app/evaluation/normalize.py tests/test_evaluation_normalize.py` with no errors.

- [ ] **Step 5: Commit**

```bash
git add app/evaluation/normalize.py tests/test_evaluation_normalize.py
git commit -m "feat(evaluation): add deterministic normalization helpers"
```

---

### Task 16: Ten-dimension deterministic scorer

Score one Case/version from the run record and frozen Case using canonical observed facts. Every dimension returns one of the five states with a machine-auditable rationale; the scorer never calls a model and never writes the Run.

**Files:**
- Create: `app/evaluation/scorer.py`
- Test: `tests/test_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scorer.py
from types import SimpleNamespace

from app.evaluation.scorer import score_case
from app.evaluation.loader import load_evaluation_cases

CASES = load_evaluation_cases("data/evaluation_cases/v1")


def _record(*, status="completed", answer="", executions=()):
    return SimpleNamespace(
        status=status,
        final_answer=answer,
        prior_tool_executions=executions,
        decision_trace=(),
    )


def _verdicts(case, record, version="v2"):
    score = score_case(case, record, agent_version=version)
    return {v.dimension: v.state for v in score.dimensions}, score


def test_completed_answer_with_gold_metrics_and_binding_passes() -> None:
    case = next(c for c in CASES.cases if c.business_task == "gmv_diagnosis")
    runtime = []
    for index, tool in enumerate(case.expected_tool_calls, start=1):
        runtime.append(
            {
                "name": tool.name,
                "arguments": tool.parameters,
                "result": {"result_id": f"R{index}", **case.gold_metrics},
            }
        )
    last_id = f"R{len(case.expected_tool_calls)}"
    answer = (
        f"本期销售额 {case.gold_metrics['current_gmv']:,.2f}[R1]，"
        f"销售额变化率 {case.gold_metrics['gmv_change_rate'] * 100:.2f}%[{last_id}]。"
    )
    states, score = _verdicts(case, _record(answer=answer, executions=runtime))

    assert states["execution_status"] == "pass"
    assert states["tool_selection"] == "pass"
    assert states["factual_accuracy"] == "pass"
    assert states["evidence_binding"] == "pass"
    assert score.task_success == "pass"


def test_v1_evidence_binding_is_not_applicable() -> None:
    case = next(c for c in CASES.cases if c.business_task == "gmv_diagnosis")
    states, _ = _verdicts(case, _record(answer="本期销售额稳定。"), version="v1")
    assert states["evidence_binding"] == "not_applicable"


def test_failed_execution_fails_execution_status() -> None:
    case = CASES.cases[0]
    states, score = _verdicts(case, _record(status="failed", answer=None))
    assert states["execution_status"] == "fail"
    assert score.task_success == "fail"


def test_wrong_cited_result_id_fails_evidence_binding() -> None:
    case = next(c for c in CASES.cases if c.business_task == "gmv_diagnosis")
    runtime = [{"name": "calculate_metrics", "result": {"result_id": "R9"}}]
    answer = "销售额变化率 0.40%[R1]。"
    states, _ = _verdicts(case, _record(answer=answer, executions=runtime))
    assert states["evidence_binding"] == "fail"


def test_unparseable_answer_metric_goes_needs_review() -> None:
    case = next(c for c in CASES.cases if c.business_task == "gmv_diagnosis")
    states, score = _verdicts(case, _record(answer="情况比较复杂，需要再看看。"))
    assert states["factual_accuracy"] in {"needs_review", "fail"}
    assert score.task_success in {"needs_review", "fail"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation.scorer'`.

- [ ] **Step 3: Write the implementation**

```python
# app/evaluation/scorer.py
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from .normalize import (
    normalize_date,
    normalize_number,
    normalize_product_ids,
    normalize_set,
    percent_to_fraction,
    within_tolerance,
)

State = Literal["pass", "fail", "not_applicable", "unavailable", "needs_review"]
_RUBRIC_PATH = Path("configs/evaluation/rubric_v1.yaml")
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]+)\]")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_DATE_KEYS = {"start_date", "end_date", "comparison_start_date", "comparison_end_date"}
_SET_KEYS = {"metrics", "group_by"}
_PRODUCT_KEYS = {"product_ids"}
_BOOL_KEYS = {"include_refunds"}
_RATE_KEYS = {"gmv_change_rate", "aov_change_rate", "cvr_change_rate"}


class DimensionVerdict(BaseModel):
    dimension: str
    state: State
    rationale: str
    evidence_refs: tuple[str, ...] = ()


class CaseScore(BaseModel):
    case_id: str
    agent_version: str
    dimensions: tuple[DimensionVerdict, ...]
    task_success: State


def load_rubric(path: Path = _RUBRIC_PATH) -> dict:
    return yaml.safe_load(path.read_bytes())


def _walk_metrics(value, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                output[str(key)] = float(item)
            else:
                _walk_metrics(item, output)
    elif isinstance(value, list):
        for item in value:
            _walk_metrics(item, output)


def _runtime_facts(record) -> tuple[set[str], dict[str, float], list[tuple[str, dict]]]:
    ids: set[str] = set()
    metrics: dict[str, float] = {}
    calls: list[tuple[str, dict]] = []
    for execution in record.prior_tool_executions:
        name = execution.get("name", "")
        arguments = execution.get("arguments", {})
        calls.append((name, arguments))
        result = execution.get("result", {})
        if isinstance(result.get("result_id"), str):
            ids.add(result["result_id"])
        _walk_metrics(result, metrics)
    return ids, metrics, calls


def _cited_ids(text: str) -> list[str]:
    return _CITATION_RE.findall(text)


def _answer_metrics(text: str, aliases: dict[str, list[str]]) -> dict[str, float]:
    observed: dict[str, float] = {}
    for canonical, options in aliases.items():
        for alias in options:
            for match in re.finditer(re.escape(alias), text):
                window = text[match.end() : match.end() + 14]
                number_match = _NUMBER_RE.search(window)
                if number_match is None:
                    continue
                value = normalize_number(number_match.group(0))
                if value is None:
                    continue
                if canonical in _RATE_KEYS and "%" in window:
                    value = percent_to_fraction(value)
                observed.setdefault(canonical, value)
    return observed


def _verdict(dimension: str, state: State, rationale: str) -> DimensionVerdict:
    return DimensionVerdict(dimension=dimension, state=state, rationale=rationale)


def _execution(record) -> DimensionVerdict:
    ok = record.status == "completed" and bool(
        isinstance(record.final_answer, str) and record.final_answer.strip()
    )
    return _verdict("execution_status", "pass" if ok else "fail", "run completed with answer" if ok else "no final answer")


def _tool_selection(case, calls) -> DimensionVerdict:
    actual = tuple(name for name, _ in calls)
    expected = tuple(tool.name for tool in case.expected_tool_calls)
    if actual == expected:
        return _verdict("tool_selection", "pass", "tool order matches expected")
    alternatives = tuple(case.allowed_alternatives)
    if any(tuple(alt) == actual for alt in alternatives):  # pragma: no cover - alt fixtures
        return _verdict("tool_selection", "pass", "tool order matches allowed alternative")
    return _verdict("tool_selection", "fail", f"tools {actual} != expected {expected}")


def _parameters(case, calls) -> DimensionVerdict:
    actual_map = {name: arguments for name, arguments in calls}
    for expected in case.expected_tool_calls:
        arguments = actual_map.get(expected.name)
        if arguments is None:
            return _verdict("parameter_accuracy", "fail", f"missing call {expected.name}")
        for key, expected_value in expected.parameters.items():
            actual_value = arguments.get(key)
            if key in _DATE_KEYS:
                if normalize_date(actual_value) != normalize_date(expected_value):
                    return _verdict("parameter_accuracy", "fail", f"{key} mismatch")
            elif key in _PRODUCT_KEYS:
                if normalize_product_ids(actual_value or []) != normalize_product_ids(expected_value or []):
                    return _verdict("parameter_accuracy", "fail", f"{key} mismatch")
            elif key in _SET_KEYS:
                if normalize_set(actual_value or []) != normalize_set(expected_value or []):
                    return _verdict("parameter_accuracy", "fail", f"{key} mismatch")
            elif key in _BOOL_KEYS:
                if bool(actual_value) != bool(expected_value):
                    return _verdict("parameter_accuracy", "fail", f"{key} mismatch")
            elif actual_value != expected_value:
                return _verdict("parameter_accuracy", "fail", f"{key} mismatch")
    return _verdict("parameter_accuracy", "pass", "all expected parameters match")


def _numeric_facts(case) -> list[str]:
    core = load_rubric()["core_facts"].get(case.business_task, [])
    return [key for key in core if key in case.gold_metrics]


def _factual(case, observed) -> DimensionVerdict:
    required = _numeric_facts(case)
    if not required:
        return _verdict("factual_accuracy", "not_applicable", "no numeric gold facts")
    compared = False
    for key in required:
        if key not in observed:
            return _verdict("factual_accuracy", "fail", f"missing core metric {key}")
        tolerance = case.numeric_tolerances.get(key, 0.0)
        if not within_tolerance(observed[key], case.gold_metrics[key], tolerance):
            return _verdict("factual_accuracy", "fail", f"{key} outside tolerance")
        compared = True
    return _verdict("factual_accuracy", "pass" if compared else "needs_review", "core metrics match gold")


def _completeness(case, observed) -> DimensionVerdict:
    core = load_rubric()["core_facts"].get(case.business_task, [])
    numeric = set(observed)
    missing = [key for key in core if key in case.gold_metrics and key not in numeric]
    if missing:
        return _verdict("required_output_completeness", "fail", f"missing {missing}")
    return _verdict("required_output_completeness", "pass", "core outputs covered")


def _evidence_for_version(
    case, cited, runtime_ids, agent_version: str
) -> DimensionVerdict:
    if agent_version == "v1":
        return _verdict("evidence_binding", "not_applicable", "v1 made no evidence promise")
    if not cited:
        return _verdict("evidence_binding", "fail", "v2 answer cites no result_id")
    if any(ref not in runtime_ids for ref in cited):
        return _verdict("evidence_binding", "fail", "cites result_id not produced this case")
    return _verdict("evidence_binding", "pass", "all citations bind to runtime results")


def _unsupported(case, observed, runtime_metrics) -> DimensionVerdict:
    for key, value in observed.items():
        supported = any(
            within_tolerance(value, runtime_value, case.numeric_tolerances.get(key, 1e-9))
            for _, runtime_value in runtime_metrics.items()
        )
        if not supported:
            return _verdict("unsupported_claim", "fail", f"{key} not found in tool results")
    return _verdict("unsupported_claim", "pass", "claims map to tool results")


def _instruction(case, text) -> DimensionVerdict:
    requires_direct = "直接回答" in case.user_input
    asks_back = isinstance(text, str) and text.strip().endswith("？")
    if requires_direct and asks_back:
        return _verdict("instruction_following", "fail", "asked back instead of answering")
    return _verdict("instruction_following", "pass", "direct answer provided")


def _insufficiency(case, text) -> DimensionVerdict:
    if case.business_task != "data_insufficient":
        return _verdict("data_insufficiency_handling", "not_applicable", "not an insufficient-data case")
    states_limit = isinstance(text, str) and ("限制" in text or "无法" in text)
    return _verdict(
        "data_insufficiency_handling",
        "pass" if states_limit else "fail",
        "limitation stated" if states_limit else "no limitation stated",
    )


def _recommendation(case, text) -> DimensionVerdict:
    needs_rec = case.business_task in {"next_week_priorities"}
    if not needs_rec:
        return _verdict("recommendation_support", "not_applicable", "no recommendation required")
    has_rec = isinstance(text, str) and ("建议" in text or "优先" in text)
    return _verdict(
        "recommendation_support",
        "pass" if has_rec else "fail",
        "recommendation present" if has_rec else "recommendation missing",
    )


def _task_success(verdicts: dict[str, DimensionVerdict], case) -> State:
    rubric = load_rubric()
    gate = rubric["task_success_gate"]
    mapping = rubric["success_criteria_map"]
    resolved = {}
    for name in gate:
        dimension = mapping.get(name, name)
        resolved[name] = verdicts[dimension].state
    if any(state == "needs_review" for state in resolved.values()):
        return "needs_review"
    return "pass" if all(state == "pass" for state in resolved.values()) else "fail"


def score_case(case, record, *, agent_version: str) -> CaseScore:
    text = record.final_answer if isinstance(record.final_answer, str) else ""
    rubric = load_rubric()
    runtime_ids, runtime_metrics, calls = _runtime_facts(record)
    observed = _answer_metrics(text, rubric["metric_aliases"])

    verdicts = [
        _execution(record),
        _tool_selection(case, calls),
        _parameters(case, calls),
        _factual(case, observed),
        _completeness(case, observed),
        _evidence_for_version(case, _cited_ids(text), runtime_ids, agent_version),
        _unsupported(case, observed, runtime_metrics),
        _instruction(case, text),
        _insufficiency(case, text),
        _recommendation(case, text),
    ]
    by_name = {verdict.dimension: verdict for verdict in verdicts}
    return CaseScore(
        case_id=case.case_id,
        agent_version=agent_version,
        dimensions=tuple(verdicts),
        task_success=_task_success(by_name, case),
    )
```

The Task Success gate maps the six gate terms in the rubric onto dimension states through `success_criteria_map`; Evidence Binding is V2-only and never enters the shared gate.


- [ ] **Step 4: Run tests and lint**

Run: `python3 -m pytest tests/test_scorer.py -v`
Expected: PASS (5 passed). Then `python3 -m ruff check app/evaluation/scorer.py tests/test_scorer.py` with no errors.

- [ ] **Step 5: Commit**

```bash
git add app/evaluation/scorer.py tests/test_scorer.py
git commit -m "feat(evaluation): add ten-dimension deterministic scorer"
```

---

### Task 17: Scoring artifact writer and reader

Bind the scoring output immutably to the two Runs, the frozen Case Bundle, the frozen Rubric and the scorer source. Atomic publish under `outputs/evaluations/<evaluation_id>/`, with a strict reader.

**Files:**
- Create: `app/evaluation/evaluation_artifacts.py`
- Test: `tests/test_evaluation_artifacts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation_artifacts.py
import json
import re
from pathlib import Path

import pytest

from app.evaluation.evaluation_artifacts import (
    EvaluationError,
    load_evaluation,
    write_evaluation,
)
from app.evaluation.loader import load_evaluation_cases

BUNDLE = load_evaluation_cases("data/evaluation_cases/v1")


def _make_run(root: Path, version: str) -> Path:
    run_dir = root / f"{version}-run"
    run_dir.mkdir()
    lines = []
    for index, case in enumerate(BUNDLE.cases, start=1):
        lines.append(
            json.dumps(
                {
                    "sequence": index,
                    "case_id": case.case_id,
                    "split": case.split,
                    "dataset_id": case.dataset_id,
                    "dataset_version": case.dataset_version,
                    "agent_version": version,
                    "repeat_index": 1,
                    "user_input": case.user_input,
                    "status": "completed",
                    "final_answer": "",
                    "error_code": None,
                    "prior_tool_executions": [],
                    "decision_trace": [],
                    "latency_ms": 0,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
                ensure_ascii=False,
            )
        )
    (run_dir / "case_runs.jsonl").write_text("\n".join(lines) + "\n")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": f"{version}-id", "agent_version": version})
    )
    (run_dir / "summary.json").write_text(json.dumps({"total": len(BUNDLE.cases)}))
    return run_dir


def test_write_then_load_roundtrip(tmp_path: Path) -> None:
    v1 = _make_run(tmp_path, "v1")
    v2 = _make_run(tmp_path, "v2")
    output_root = tmp_path / "evaluations"

    path = write_evaluation(
        v1_run_path=v1,
        v2_run_path=v2,
        case_dir=Path("data/evaluation_cases/v1"),
        output_root=output_root,
    )

    published = load_evaluation(path)
    assert len(published.scores) == len(BUNDLE.cases) * 2
    assert published.manifest.case_count == len(BUNDLE.cases)
    assert re.match(r"^[0-9a-f]{64}$", published.manifest.scorer_source_sha256)
    for version in ("v1", "v2"):
        assert published.summary["task_success"][version]["fail"] == len(BUNDLE.cases)
    assert (path / "evaluation_manifest.json").exists()
    assert (path / "case_scores.jsonl").exists()


def test_write_refuses_to_overwrite(tmp_path: Path) -> None:
    v1 = _make_run(tmp_path, "v1")
    v2 = _make_run(tmp_path, "v2")
    output_root = tmp_path / "evaluations"
    first = write_evaluation(v1, v2, Path("data/evaluation_cases/v1"), output_root)
    with pytest.raises(EvaluationError):
        write_evaluation(v1, v2, Path("data/evaluation_cases/v1"), output_root, evaluation_id=first.name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evaluation_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation.evaluation_artifacts'`.

- [ ] **Step 3: Write the implementation**

```python
# app/evaluation/evaluation_artifacts.py
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ..experiments.llm_artifacts import RealCaseRunRecord
from .freeze import FrozenConfig, load_frozen_configs
from .loader import load_evaluation_cases
from .scorer import CaseScore, score_case


class EvaluationError(Exception):
    """Raised when evaluation inputs are invalid or publication fails."""


class RunBinding(BaseModel):
    run_id: str
    path: str
    run_manifest_sha256: str
    case_runs_sha256: str


class BundleBinding(BaseModel):
    case_dir: str
    manifest_sha256: str


class EvaluationManifest(BaseModel):
    evaluation_id: str
    created_at: str
    case_count: int
    rubric: FrozenConfig
    v1_run: RunBinding
    v2_run: RunBinding
    case_bundle: BundleBinding
    scorer_source_sha256: str


class PublishedEvaluation(BaseModel):
    manifest: EvaluationManifest
    scores: tuple[CaseScore, ...]
    summary: dict


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _load_records(run_path: Path) -> dict[str, RealCaseRunRecord]:
    records = {}
    for line in (run_path / "case_runs.jsonl").read_text().splitlines():
        record = RealCaseRunRecord.model_validate(json.loads(line))
        records[record.case_id] = record
    return records


def _build_summary(scores: tuple[CaseScore, ...], case_count: int) -> dict:
    versions = ("v1", "v2")
    task_counts = {
        version: {"pass": 0, "fail": 0, "needs_review": 0, "not_applicable": 0, "unavailable": 0}
        for version in versions
    }
    dimension_fail = {version: {} for version in versions}
    dimension_review = {version: {} for version in versions}
    for score in scores:
        task_counts[score.agent_version][score.task_success] += 1
        for verdict in score.dimensions:
            if verdict.state == "fail":
                dimension_fail[score.agent_version][verdict.dimension] = (
                    dimension_fail[score.agent_version].get(verdict.dimension, 0) + 1
                )
            if verdict.state == "needs_review":
                dimension_review[score.agent_version][verdict.dimension] = (
                    dimension_review[score.agent_version].get(verdict.dimension, 0) + 1
                )
    return {
        "case_count": case_count,
        "task_success": task_counts,
        "dimension_failures": dimension_fail,
        "dimension_needs_review": dimension_review,
    }


def write_evaluation(
    v1_run_path: Path,
    v2_run_path: Path,
    case_dir: Path,
    output_root: Path,
    *,
    evaluation_id: str | None = None,
    scorer_path: Path = Path("app/evaluation/scorer.py"),
) -> Path:
    bundle = load_evaluation_cases(case_dir)
    v1_records = _load_records(v1_run_path)
    v2_records = _load_records(v2_run_path)
    expected = {case.case_id for case in bundle.cases}
    if set(v1_records) != expected or set(v2_records) != expected:
        raise EvaluationError("run case sets do not match frozen case bundle")

    if evaluation_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        evaluation_id = f"evaluation-v1__{stamp}__{_sha256(scorer_path)[:8]}"
    target = output_root / evaluation_id
    if target.exists():
        raise EvaluationError(f"refusing to overwrite {evaluation_id}")

    scores: list[CaseScore] = []
    for case in bundle.cases:
        scores.append(score_case(case, v1_records[case.case_id], agent_version="v1"))
        scores.append(score_case(case, v2_records[case.case_id], agent_version="v2"))

    frozen = {config.name: config for config in load_frozen_configs()}
    manifest = EvaluationManifest(
        evaluation_id=evaluation_id,
        created_at=datetime.now(UTC).isoformat(),
        case_count=len(bundle.cases),
        rubric=frozen["rubric"],
        v1_run=RunBinding(
            run_id=json.loads((v1_run_path / "run_manifest.json").read_text())["run_id"],
            path=str(v1_run_path),
            run_manifest_sha256=_sha256(v1_run_path / "run_manifest.json"),
            case_runs_sha256=_sha256(v1_run_path / "case_runs.jsonl"),
        ),
        v2_run=RunBinding(
            run_id=json.loads((v2_run_path / "run_manifest.json").read_text())["run_id"],
            path=str(v2_run_path),
            run_manifest_sha256=_sha256(v2_run_path / "run_manifest.json"),
            case_runs_sha256=_sha256(v2_run_path / "case_runs.jsonl"),
        ),
        case_bundle=BundleBinding(case_dir=str(case_dir), manifest_sha256=bundle.manifest_sha256),
        scorer_source_sha256=_sha256(scorer_path),
    )
    summary = _build_summary(tuple(scores), len(bundle.cases))

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root) as temporary:
        staging = Path(temporary)
        (staging / "evaluation_manifest.json").write_bytes(_canonical_bytes(manifest.model_dump()))
        (staging / "case_scores.jsonl").write_bytes(
            b"".join(_canonical_bytes(score.model_dump()) for score in scores)
        )
        (staging / "summary.json").write_bytes(_canonical_bytes(summary))
        staging.rename(target)
    return target


def load_evaluation(path: Path) -> PublishedEvaluation:
    manifest = EvaluationManifest.model_validate(
        json.loads((path / "evaluation_manifest.json").read_text())
    )
    scores = tuple(
        CaseScore.model_validate(json.loads(line))
        for line in (path / "case_scores.jsonl").read_text().splitlines()
    )
    summary = json.loads((path / "summary.json").read_text())
    return PublishedEvaluation(manifest=manifest, scores=scores, summary=summary)
```

- [ ] **Step 4: Run tests and lint**

Run:

`python3 -m pytest tests/test_evaluation_artifacts.py -v`
Expected: PASS (2 passed). Then `python3 -m ruff check app/evaluation/evaluation_artifacts.py tests/test_evaluation_artifacts.py` with no errors.

- [ ] **Step 5: Commit**

```bash
git add app/evaluation/evaluation_artifacts.py tests/test_evaluation_artifacts.py
git commit -m "feat(evaluation): add bound scoring artifact writer and reader"
```

---

### Task 18: Scoring CLI, console script, Makefile target

Offline entry point that scores a paired Run without any model call. Register `evaluate-runs` and add the matching Makefile target.

**Files:**
- Create: `app/evaluation/cli.py`
- Test: `tests/test_evaluation_cli.py`
- Modify: `pyproject.toml`
- Modify: `Makefile`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation_cli.py
import json
from pathlib import Path

from app.evaluation.cli import evaluate_main
from app.evaluation.loader import load_evaluation_cases

BUNDLE = load_evaluation_cases("data/evaluation_cases/v1")


def _make_run(root: Path, version: str) -> Path:
    run_dir = root / f"{version}-run"
    run_dir.mkdir()
    lines = [
        json.dumps(
            {
                "sequence": index,
                "case_id": case.case_id,
                "split": case.split,
                "dataset_id": case.dataset_id,
                "dataset_version": case.dataset_version,
                "agent_version": version,
                "repeat_index": 1,
                "user_input": case.user_input,
                "status": "completed",
                "final_answer": "",
                "error_code": None,
                "prior_tool_executions": [],
                "decision_trace": [],
                "latency_ms": 0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            ensure_ascii=False,
        )
        for index, case in enumerate(BUNDLE.cases, start=1)
    ]
    (run_dir / "case_runs.jsonl").write_text("\n".join(lines) + "\n")
    (run_dir / "run_manifest.json").write_text(json.dumps({"run_id": f"{version}-id"}))
    (run_dir / "summary.json").write_text(json.dumps({"total": len(BUNDLE.cases)}))
    return run_dir


def test_evaluate_main_success(tmp_path: Path) -> None:
    v1 = _make_run(tmp_path, "v1")
    v2 = _make_run(tmp_path, "v2")
    output_root = tmp_path / "evaluations"

    exit_code = evaluate_main(
        ["--v1-run", str(v1), "--v2-run", str(v2), "--output-root", str(output_root)]
    )

    assert exit_code == 0
    assert output_root.exists()
    assert len(list(output_root.iterdir())) == 1


def test_evaluate_main_error_on_missing_run(tmp_path: Path) -> None:
    exit_code = evaluate_main(
        [
            "--v1-run", str(tmp_path / "missing"),
            "--v2-run", str(tmp_path / "missing2"),
            "--output-root", str(tmp_path / "evaluations"),
        ]
    )
    assert exit_code == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evaluation_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation.cli'`.

- [ ] **Step 3: Write the implementation**

```python
# app/evaluation/cli.py
import argparse
from pathlib import Path

from .evaluation_artifacts import EvaluationError, write_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluate-runs")
    parser.add_argument("--v1-run", required=True, type=Path)
    parser.add_argument("--v2-run", required=True, type=Path)
    parser.add_argument("--case-dir", default=Path("data/evaluation_cases/v1"), type=Path)
    parser.add_argument("--output-root", default=Path("outputs/evaluations"), type=Path)
    return parser


def evaluate_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = write_evaluation(
            v1_run_path=args.v1_run,
            v2_run_path=args.v2_run,
            case_dir=args.case_dir,
            output_root=args.output_root,
        )
    except (EvaluationError, FileNotFoundError, ValueError) as exc:
        print(f"evaluate-runs: {exc}")
        return 1
    print(f"evaluation published: {path}")
    return 0
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, add to the existing `[project.scripts]` table (alongside `real-llm-pilot` / `real-llm-formal` from Task 11):

```toml
evaluate-runs = "app.evaluation.cli:evaluate_main"
```

- [ ] **Step 5: Add the Makefile target**

In the `Makefile`, add (next to the real-llm targets from Task 11):

```makefile
.PHONY: evaluate-runs
evaluate-runs:
	evaluate-runs --v1-run $(V1_RUN) --v2-run $(V2_RUN)
```

- [ ] **Step 6: Add the output dir to `.gitignore`**

Append to `.gitignore`:

```gitignore
outputs/evaluations/
```

- [ ] **Step 7: Run tests, lint, and verify the entry point**

Run: `python3 -m pytest tests/test_evaluation_cli.py -v`
Expected: PASS (2 passed). Then:

`python3 -m ruff check app/evaluation/cli.py tests/test_evaluation_cli.py` with no errors.
`pip install -e .` then `evaluate-runs --help` should print usage.

- [ ] **Step 8: Commit**

```bash
git add app/evaluation/cli.py tests/test_evaluation_cli.py pyproject.toml Makefile .gitignore
git commit -m "feat(evaluation): add offline scoring cli and console script"
```

---

### Batch B gate

- [ ] **Gate B-1: frozen assets and scorer all committed**

Verify:

```bash
git status --short   # clean
ls configs/evaluation/rubric_v1.yaml configs/evaluation/error_taxonomy_v1.yaml configs/evaluation/statistics_v1.yaml
python3 -m pytest tests/test_evaluation_freeze.py tests/test_evaluation_loader.py tests/test_evaluation_normalize.py tests/test_scorer.py tests/test_evaluation_artifacts.py tests/test_evaluation_cli.py
```

Expected: all three versioned YAMLs present, every Batch B test passes, and the freeze manifest lists each asset with a 64-hex SHA-256. No new files in this gate; it is a checkpoint only.

---

## Batch C: Formal 100-Case paired Run (real model)

Batch C adds no code. It executes the already-committed formal CLI against the frozen 100-Case set and verifies the Run and comparison artifacts. No auto-retry; any failed logical record stays in the artifact and is scored later, not rerun.

### Task 19: Formal preflight checks (human-supervised)

Confirm the inputs before any model token is spent. Every check must pass; do not proceed otherwise.

**Files:** none (read-only verification).

- [ ] **Step 1: Verify credentials and settings come only from the environment**

Run:

```bash
python3 - <<'PY'
import os
for name in ("LLM_API_KEY", "LLM_MODEL"):
    assert os.environ.get(name), f"missing {name}"
print("LLM_API_KEY present:", "LLM_API_KEY" in os.environ)
print("LLM_MODEL:", os.environ["LLM_MODEL"])
print("LLM_BASE_URL:", os.environ.get("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"))
print("LLM_TEMPERATURE:", os.environ.get("LLM_TEMPERATURE", "0"))
PY
```

Expected: key present (value never printed), model printed, base URL the Ark endpoint, temperature `0`. If `LLM_TEMPERATURE` is set to anything other than `0`, stop.

- [ ] **Step 2: Verify the frozen Case set**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from app.evaluation.loader import load_evaluation_cases

bundle = load_evaluation_cases(Path("data/evaluation_cases/v1"))
assert len(bundle.cases) == 100, len(bundle.cases)
splits = {case.split for case in bundle.cases}
clusters = {case.statistical_cluster_id for case in bundle.cases}
print("cases:", len(bundle.cases))
print("splits:", sorted(splits))
print("clusters:", len(clusters))
PY
```

Expected: exactly 100 Cases, splits `development` (70) and `public_validation` (30), exactly 50 clusters (two Cases each).

- [ ] **Step 3: Verify run settings and dataset integrity**

Run:

```bash
python3 -m pytest tests/test_real_llm_run.py tests/test_comparisons.py -q
git status --short
```

Expected: tests pass; the working tree is clean (only generated/ignored outputs allowed). Confirm the frozen settings are `max_steps=8`, `concurrency=1`, `repeats=1`, no retry (asserted by Task 9 tests).

- [ ] **Step 4: Human approval to spend the formal budget**

Pause. Present the model name, 100-Case count, and that this makes 200 single-pass calls with no retries. Proceed to Task 20 only after explicit approval.

---

### Task 20: Execute the formal paired Run

Run both versions against all 100 Cases with alternating order, then stop and inspect.

**Files:** none (produces artifacts under `outputs/`).

- [ ] **Step 1: Run the formal CLI**

Run:

```bash
real-llm-formal
```

(or `python3 -m app.experiments.llm_cli formal`). Expected: process exits `0`; two Run directories under `outputs/real_runs/` and one comparison directory under `outputs/paired_comparisons/`.

- [ ] **Step 2: Record the published paths**

Run:

```bash
ls -dt outputs/real_runs/* | head -2
ls -dt outputs/paired_comparisons/* | head -1
```

Write down the V1 Run path, V2 Run path, and comparison path for Tasks 21–24. Do not rename or edit these directories.

---

### Task 21: Verify formal Run and comparison artifacts

- [ ] **Step 1: Verify logical record counts and no retries**

Run (substitute the two Run directories):

```bash
wc -l outputs/real_runs/<v1-run-id>/case_runs.jsonl
wc -l outputs/real_runs/<v2-run-id>/case_runs.jsonl
```

Expected: exactly 100 lines each (200 logical records total). Confirm each `case_id` appears once per file and `repeat_index` is `1` on every record.

- [ ] **Step 2: Verify the shared comparison manifest and fairness**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from app.experiments.comparisons import verify_comparison

verify_comparison(Path("outputs/paired_comparisons/<comparison-id>"))
print("comparison verified")
PY
```

Expected: `comparison verified` — same model/params/settings, both runs reference the same frozen Case set and a shared `comparison_group_id`, and version order alternates.

- [ ] **Step 3: Gate C checkpoint**

Confirm:

```bash
git status --short   # outputs are ignored; tree remains clean
```

Expected: atomic publish completed, fairness validation passed, 200 logical records present, and nothing in the Runs was modified after publication. Gate C is a checkpoint; no commit.

---

## Batch D: Deterministic scoring, E1–E10 attribution, RCA and Error Shift

No model calls and no Run edits. Inputs: the two formal Run directories plus the frozen Case Bundle and frozen taxonomy. Outputs: one evaluation directory (Batch B writer) and one error-analysis directory.

### Task 22: Score the formal pair and build attribution engine

Run the offline scorer on the formal Run pair, then implement earliest-causal E1–E10 attribution.

**Files:**
- Create: `app/error_analysis/__init__.py`
- Create: `app/error_analysis/taxonomy.py`
- Test: `tests/test_error_attribution.py`

- [ ] **Step 1: Score the formal pair (operational)**

Run (substitute the formal Run paths from Task 20):

```bash
evaluate-runs --v1-run outputs/real_runs/<v1-run-id> --v2-run outputs/real_runs/<v2-run-id>
```

Expected: exit `0`; one directory under `outputs/evaluations/` containing 200 CaseScore records. Confirm:

```bash
wc -l outputs/evaluations/<evaluation-id>/case_scores.jsonl   # 200
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_error_attribution.py
from app.error_analysis.taxonomy import ErrorAttribution, attribute_errors
from app.evaluation.scorer import CaseScore, DimensionVerdict


def _score(states):
    return CaseScore(
        case_id="CASE_003",
        agent_version="v2",
        dimensions=tuple(
            DimensionVerdict(dimension=name, state=state, rationale="")
            for name, state in states.items()
        ),
        task_success="fail",
    )


def _all_pass(name):
    base = {
        "execution_status": "pass",
        "tool_selection": "pass",
        "parameter_accuracy": "pass",
        "factual_accuracy": "pass",
        "required_output_completeness": "pass",
        "evidence_binding": "not_applicable",
        "unsupported_claim": "pass",
        "instruction_following": "pass",
        "data_insufficiency_handling": "not_applicable",
        "recommendation_support": "not_applicable",
    }
    base.update(name)
    return base


def test_pass_case_has_no_error() -> None:
    score = _score(_all_pass({}))
    score = score.model_copy(update={"task_success": "pass"})
    result = attribute_errors(score)
    assert result.attribution_status == "no_error"
    assert result.primary_error is None
    assert result.secondary_errors == ()


def test_earliest_causal_becomes_primary() -> None:
    score = _score(_all_pass({"tool_selection": "fail", "factual_accuracy": "fail"}))
    result = attribute_errors(score)
    assert result.primary_error == "E1"
    assert result.secondary_errors == ("E4",)
    assert result.attribution_status == "attributed"


def test_duplicate_codes_are_deduped() -> None:
    score = _score(_all_pass({"evidence_binding": "fail", "unsupported_claim": "fail"}))
    result = attribute_errors(score)
    assert result.primary_error == "E6"
    assert result.secondary_errors == ()


def test_needs_review_dimension_is_ambiguous() -> None:
    score = _score(_all_pass({"factual_accuracy": "needs_review", "required_output_completeness": "fail"}))
    result = attribute_errors(score)
    assert result.attribution_status == "ambiguous"
    assert result.primary_error == "E8"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_error_attribution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.error_analysis'`.

- [ ] **Step 4: Write the implementation**

```python
# app/error_analysis/__init__.py
```

```python
# app/error_analysis/taxonomy.py
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

_TAXONOMY_PATH = Path("configs/evaluation/error_taxonomy_v1.yaml")

AttributionStatus = Literal["attributed", "no_error", "ambiguous"]


class ErrorAttribution(BaseModel):
    case_id: str
    agent_version: str
    task_success: str
    primary_error: str | None
    secondary_errors: tuple[str, ...]
    first_causal_event_sequence: tuple[str, ...]
    attribution_status: AttributionStatus


def load_taxonomy(path: Path = _TAXONOMY_PATH) -> dict:
    return yaml.safe_load(path.read_bytes())


def attribute_errors(score, *, taxonomy: dict | None = None) -> ErrorAttribution:
    taxonomy = taxonomy or load_taxonomy()
    dimension_map = taxonomy["dimension_error_map"]
    precedence = taxonomy["causal_precedence"]

    if score.task_success == "pass":
        return ErrorAttribution(
            case_id=score.case_id,
            agent_version=score.agent_version,
            task_success=score.task_success,
            primary_error=None,
            secondary_errors=(),
            first_causal_event_sequence=(),
            attribution_status="no_error",
        )

    ambiguous = any(
        verdict.state == "needs_review" and verdict.dimension in dimension_map
        for verdict in score.dimensions
    )
    failed_dimensions = [
        verdict.dimension for verdict in score.dimensions if verdict.state == "fail"
    ]
    ordered_codes = sorted(
        {dimension_map[dimension] for dimension in failed_dimensions if dimension in dimension_map},
        key=precedence.index,
    )
    primary = ordered_codes[0] if ordered_codes else None
    secondary = tuple(ordered_codes[1:])
    sequence = tuple(
        dimension
        for dimension in failed_dimensions
        if dimension in dimension_map and dimension_map[dimension] in ordered_codes
    )
    status: AttributionStatus = "ambiguous" if ambiguous else "attributed"
    return ErrorAttribution(
        case_id=score.case_id,
        agent_version=score.agent_version,
        task_success=score.task_success,
        primary_error=primary,
        secondary_errors=secondary,
        first_causal_event_sequence=sequence,
        attribution_status=status,
    )
```

- [ ] **Step 5: Run tests and lint**

Run: `python3 -m pytest tests/test_error_attribution.py -v`
Expected: PASS (4 passed). Then `python3 -m ruff check app/error_analysis/taxonomy.py tests/test_error_attribution.py` with no errors.

- [ ] **Step 6: Commit**

```bash
git add app/error_analysis/__init__.py app/error_analysis/taxonomy.py tests/test_error_attribution.py
git commit -m "feat(error-analysis): add earliest-causal e1-e10 attribution"
```

---

### Task 23: RCA record builder

Turn an attribution into an audit record with deterministic root-cause and intervention templates. The builder does not infer beyond the attribution; ambiguous records carry a mandatory human review note.

**Files:**
- Create: `app/error_analysis/rca.py`
- Test: `tests/test_rca.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rca.py
from app.error_analysis.rca import build_rca
from app.error_analysis.taxonomy import ErrorAttribution


def _attribution(**overrides):
    base = {
        "case_id": "CASE_003",
        "agent_version": "v2",
        "task_success": "fail",
        "primary_error": "E2",
        "secondary_errors": ("E4",),
        "first_causal_event_sequence": ("parameter_accuracy", "factual_accuracy"),
        "attribution_status": "attributed",
    }
    base.update(overrides)
    return ErrorAttribution(**base)


def test_build_rca_attributed_record() -> None:
    attribution = _attribution()
    rca = build_rca(attribution, symptom=("parameter_accuracy", "factual_accuracy"))
    assert rca.primary_error == "E2"
    assert rca.secondary_errors == ("E4",)
    assert rca.root_cause
    assert rca.intervention
    assert rca.review_note == ""


def test_build_rca_ambiguous_requires_review_note() -> None:
    attribution = _attribution(attribution_status="ambiguous", primary_error="E8")
    rca = build_rca(attribution, symptom=("required_output_completeness",))
    assert rca.attribution_status == "ambiguous"
    assert rca.review_note
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rca.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.error_analysis.rca'`.

- [ ] **Step 3: Write the implementation**

```python
# app/error_analysis/rca.py
from pydantic import BaseModel

from .taxonomy import ErrorAttribution

_RCA_TEMPLATES: dict[str, tuple[str, str]] = {
    "E1": ("wrong tool selected for the requested analysis", "bind the requested task to the correct tool in the prompt"),
    "E2": ("tool arguments did not match the requested scope", "require exact date/metric/product parameters before the call"),
    "E3": ("data retrieval failed at the tool/runtime layer", "harden retrieval error handling and surface the failure explicitly"),
    "E4": ("metric computation did not match the gold calculation", "fix the calculation path and require metric cross-checks"),
    "E5": ("reasoning chain did not connect the facts to the conclusion", "require stepwise fact-to-conclusion reasoning"),
    "E6": ("claim or citation not grounded in this case's tool results", "require every claim to cite an in-case result_id"),
    "E7": ("answer did not follow the user's instruction", "make the requested format/directness constraint explicit"),
    "E8": ("required output element was omitted", "add a completion checklist for the required core facts"),
    "E9": ("recommendation was not supported by the evidence", "require recommendations to cite supporting metrics"),
    "E10": ("insufficient-data limitation was not handled correctly", "require an explicit limitation statement instead of overclaiming"),
}


class RcaRecord(BaseModel):
    case_id: str
    agent_version: str
    symptom: tuple[str, ...]
    primary_error: str | None
    secondary_errors: tuple[str, ...]
    first_causal_event_sequence: tuple[str, ...]
    root_cause: str
    intervention: str
    attribution_status: str
    review_note: str


def build_rca(attribution: ErrorAttribution, *, symptom: tuple[str, ...]) -> RcaRecord:
    primary = attribution.primary_error
    root_cause, intervention = _RCA_TEMPLATES.get(primary, ("", ""))
    review_note = ""
    if attribution.attribution_status == "ambiguous":
        review_note = "attribution is ambiguous; a human must confirm the primary cause"
    return RcaRecord(
        case_id=attribution.case_id,
        agent_version=attribution.agent_version,
        symptom=symptom,
        primary_error=primary,
        secondary_errors=attribution.secondary_errors,
        first_causal_event_sequence=attribution.first_causal_event_sequence,
        root_cause=root_cause,
        intervention=intervention,
        attribution_status=attribution.attribution_status,
        review_note=review_note,
    )
```

- [ ] **Step 4: Run tests and lint**

Run: `python3 -m pytest tests/test_rca.py -v`
Expected: PASS (2 passed). Then `python3 -m ruff check app/error_analysis/rca.py tests/test_rca.py` with no errors.

- [ ] **Step 5: Commit**

```bash
git add app/error_analysis/rca.py tests/test_rca.py
git commit -m "feat(error-analysis): add deterministic rca record builder"
```

---

### Task 24: RCA artifact, Error Shift matrix, analyze-errors CLI

Persist 200 RCA records and the v1→v2 primary-error transition matrix, then expose the offline `analyze-errors` entry point.

**Files:**
- Create: `app/error_analysis/artifacts.py`
- Create: `app/error_analysis/cli.py`
- Test: `tests/test_error_shift.py`
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_error_shift.py
import json
from pathlib import Path

from app.error_analysis.artifacts import build_shift_matrix, write_error_analysis
from app.error_analysis.taxonomy import ErrorAttribution
from app.error_analysis.cli import analyze_main
from app.evaluation.scorer import CaseScore, DimensionVerdict


def _attribution(case_id, version, code):
    status = "no_error" if code is None else "attributed"
    return ErrorAttribution(
        case_id=case_id,
        agent_version=version,
        task_success="pass" if code is None else "fail",
        primary_error=code,
        secondary_errors=(),
        first_causal_event_sequence=(),
        attribution_status=status,
    )


def test_matrix_classifies_resolved_introduced_persisted_and_shifted() -> None:
    pairs = {
        "CASE_001": (_attribution("CASE_001", "v1", "E2"), _attribution("CASE_001", "v2", None)),
        "CASE_002": (_attribution("CASE_002", "v1", None), _attribution("CASE_002", "v2", "E6")),
        "CASE_003": (_attribution("CASE_003", "v1", "E4"), _attribution("CASE_003", "v2", "E4")),
        "CASE_004": (_attribution("CASE_004", "v1", "E1"), _attribution("CASE_004", "v2", "E2")),
    }
    matrix = build_shift_matrix(pairs)
    assert matrix["resolved"]["E2"] == 1
    assert matrix["introduced"]["E6"] == 1
    assert matrix["persisted"]["E4"] == 1
    assert matrix["shifted"]["E1->E2"] == 1


def _score(case_id, version, task_success):
    return CaseScore(
        case_id=case_id,
        agent_version=version,
        dimensions=(DimensionVerdict(dimension="execution_status", state="pass", rationale=""),),
        task_success=task_success,
    )


def test_write_and_cli_roundtrip(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "evaluations" / "eval-1"
    evaluation_dir.mkdir(parents=True)
    scores = [
        _score("CASE_001", "v1", "pass"),
        _score("CASE_001", "v2", "pass"),
        _score("CASE_002", "v1", "fail"),
        _score("CASE_002", "v2", "pass"),
    ]
    (evaluation_dir / "case_scores.jsonl").write_text(
        "\n".join(json.dumps(score.model_dump()) for score in scores) + "\n"
    )
    output_root = tmp_path / "error_analysis"

    path = write_error_analysis(evaluation_dir=evaluation_dir, output_root=output_root)
    assert (path / "rca_records.jsonl").exists()
    assert (path / "error_shift.json").exists()

    exit_code = analyze_main(
        ["--evaluation", str(evaluation_dir), "--output-root", str(tmp_path / "cli_out")]
    )
    assert exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_error_shift.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.error_analysis.artifacts'`.

- [ ] **Step 3: Write the implementation**

```python
# app/error_analysis/artifacts.py
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from ..evaluation.scorer import CaseScore
from .rca import build_rca
from .taxonomy import ErrorAttribution, attribute_errors


class ErrorAnalysisError(Exception):
    """Raised when evaluation input cannot be turned into an analysis."""


def build_shift_matrix(pairs: dict[str, tuple[ErrorAttribution, ErrorAttribution]]) -> dict:
    matrix = {
        "resolved": defaultdict(int),
        "introduced": defaultdict(int),
        "persisted": defaultdict(int),
        "shifted": defaultdict(int),
    }
    for _, (v1, v2) in pairs.items():
        v1_code = v1.primary_error
        v2_code = v2.primary_error
        if v1_code is not None and v2_code is None:
            matrix["resolved"][v1_code] += 1
        elif v1_code is None and v2_code is not None:
            matrix["introduced"][v2_code] += 1
        elif v1_code == v2_code:
            matrix["persisted"][v1_code] += 1
        else:
            matrix["shifted"][f"{v1_code}->{v2_code}"] += 1
    return {key: dict(value) for key, value in matrix.items()}


def _load_scores(evaluation_dir: Path) -> list[CaseScore]:
    return [
        CaseScore.model_validate(json.loads(line))
        for line in (evaluation_dir / "case_scores.jsonl").read_text().splitlines()
    ]


def write_error_analysis(*, evaluation_dir: Path, output_root: Path) -> Path:
    scores = _load_scores(evaluation_dir)
    by_case: dict[str, dict[str, CaseScore]] = defaultdict(dict)
    for score in scores:
        by_case[score.case_id][score.agent_version] = score
    if any(set(versions) != {"v1", "v2"} for versions in by_case.values()):
        raise ErrorAnalysisError("every case requires both v1 and v2 scores")

    attributions = {}
    rca_lines = []
    for case_id, versions in by_case.items():
        v1_attribution = attribute_errors(versions["v1"])
        v2_attribution = attribute_errors(versions["v2"])
        attributions[case_id] = (v1_attribution, v2_attribution)
        for score in (versions["v1"], versions["v2"]):
            attribution = v1_attribution if score.agent_version == "v1" else v2_attribution
            symptom = tuple(
                verdict.dimension for verdict in score.dimensions if verdict.state == "fail"
            )
            rca = build_rca(attribution, symptom=symptom)
            rca_lines.append(json.dumps(rca.model_dump(), ensure_ascii=False))

    matrix = build_shift_matrix(attributions)
    analysis_id = evaluation_dir.name.replace("evaluation", "analysis", 1)
    target = output_root / analysis_id
    if target.exists():
        raise ErrorAnalysisError(f"refusing to overwrite {analysis_id}")

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root) as temporary:
        staging = Path(temporary)
        (staging / "rca_records.jsonl").write_text("\n".join(rca_lines) + "\n")
        (staging / "error_shift.json").write_text(
            json.dumps(matrix, ensure_ascii=False, sort_keys=True)
        )
        staging.rename(target)
    return target
```

```python
# app/error_analysis/cli.py
import argparse
from pathlib import Path

from .artifacts import ErrorAnalysisError, write_error_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze-errors")
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/error_analysis"), type=Path)
    return parser


def analyze_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = write_error_analysis(evaluation_dir=args.evaluation, output_root=args.output_root)
    except (ErrorAnalysisError, FileNotFoundError, ValueError) as exc:
        print(f"analyze-errors: {exc}")
        return 1
    print(f"error analysis published: {path}")
    return 0
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, add to `[project.scripts]`:

```toml
analyze-errors = "app.error_analysis.cli:analyze_main"
```

- [ ] **Step 5: Add the Makefile target and ignore rule**

In the `Makefile`:

```makefile
.PHONY: analyze-errors
analyze-errors:
	analyze-errors --evaluation $(EVALUATION)
```

In `.gitignore`:

```gitignore
outputs/error_analysis/
```

- [ ] **Step 6: Run tests, lint, and verify the entry point**

Run: `python3 -m pytest tests/test_error_shift.py -v`
Expected: PASS (2 passed). Then:

`python3 -m ruff check app/error_analysis/artifacts.py app/error_analysis/cli.py tests/test_error_shift.py` with no errors.
`pip install -e .` then `analyze-errors --help` should print usage.

- [ ] **Step 7: Commit**

```bash
git add app/error_analysis/artifacts.py app/error_analysis/cli.py tests/test_error_shift.py pyproject.toml Makefile .gitignore
git commit -m "feat(error-analysis): add rca artifacts error shift matrix and cli"
```

### Batch D gate

- [ ] **Gate D-1: full deterministic analysis committed and runnable**

Run:

```bash
git status --short
python3 -m pytest tests/test_error_attribution.py tests/test_rca.py tests/test_error_shift.py -q
```

Expected: clean tree; all Batch D tests pass. On the formal artifacts, `outputs/error_analysis/<analysis-id>/` holds exactly 200 RCA records (100 Cases × 2 versions) and a matrix whose four category counts sum to 100. No model calls were made.

---

## Batch E: Paired metrics, cluster bootstrap and McNemar

Offline only. Resampling unit is the whole statistical cluster; the sole effect estimate direction is v2−v1. Fewer than 10 clusters forces descriptive-only output.

### Task 25: Statistics config and descriptive paired metrics

**Files:**
- Create: `app/statistics/__init__.py`
- Create: `app/statistics/config.py`
- Create: `app/statistics/paired.py`
- Test: `tests/test_paired_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paired_metrics.py
from app.evaluation.scorer import CaseScore, DimensionVerdict
from app.statistics.config import load_statistics_config
from app.statistics.paired import descriptive_metrics, latency_token


def _score(version, task_success, dimension_states):
    return CaseScore(
        case_id="CASE_001",
        agent_version=version,
        dimensions=tuple(
            DimensionVerdict(dimension=name, state=state, rationale="")
            for name, state in dimension_states.items()
        ),
        task_success=task_success,
    )


def test_config_values_are_frozen() -> None:
    config = load_statistics_config()
    assert config["bootstrap"]["iterations"] == 10000
    assert config["bootstrap"]["seed"] == 20261009
    assert config["delta"]["direction"] == "v2_minus_v1"


def test_descriptive_metrics_compute_rates_and_delta() -> None:
    v1 = _score("v1", "fail", {"factual_accuracy": "fail", "evidence_binding": "not_applicable"})
    v2 = _score("v2", "pass", {"factual_accuracy": "pass", "evidence_binding": "pass"})
    metrics = descriptive_metrics((v1, v2))
    assert metrics["task_success_rate"]["v1"] == 0.0
    assert metrics["task_success_rate"]["v2"] == 1.0
    assert metrics["task_success_delta"] == 1.0
    assert metrics["needs_review"] == {"v1": 0, "v2": 0}


def test_latency_token_handles_unavailable_tokens() -> None:
    result = latency_token(
        (
            {"agent_version": "v1", "latency_ms": 120, "total_tokens": None},
            {"agent_version": "v1", "latency_ms": 200, "total_tokens": None},
        )
    )
    assert result["v1"]["latency_p50"] == 120
    assert result["v1"]["tokens_state"] == "unavailable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_paired_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.statistics'`.

- [ ] **Step 3: Write the implementation**

```python
# app/statistics/__init__.py
```

```python
# app/statistics/config.py
from pathlib import Path

import yaml

_CONFIG_PATH = Path("configs/evaluation/statistics_v1.yaml")


def load_statistics_config(path: Path = _CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_bytes())
```

```python
# app/statistics/paired.py
from collections.abc import Iterable

_VERSIONS = ("v1", "v2")


def descriptive_metrics(scores: Iterable) -> dict:
    scores = tuple(scores)
    totals = {version: 0 for version in _VERSIONS}
    passes = {version: 0 for version in _VERSIONS}
    reviews = {version: 0 for version in _VERSIONS}
    for score in scores:
        totals[score.agent_version] += 1
        if score.task_success == "pass":
            passes[score.agent_version] += 1
        if score.task_success == "needs_review":
            reviews[score.agent_version] += 1
    rates = {
        version: passes[version] / totals[version] if totals[version] else None
        for version in _VERSIONS
    }
    return {
        "n": totals,
        "task_success_rate": rates,
        "task_success_delta": rates["v2"] - rates["v1"]
        if rates["v1"] is not None and rates["v2"] is not None
        else None,
        "needs_review": reviews,
    }


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    index = round(q * (len(ordered) - 1))
    return ordered[index]


def latency_token(records: Iterable[dict]) -> dict:
    by_version: dict[str, list[dict]] = {}
    for record in records:
        by_version.setdefault(record["agent_version"], []).append(record)
    result = {}
    for version, items in by_version.items():
        latencies = [item["latency_ms"] for item in items]
        tokens = [item["total_tokens"] for item in items if item.get("total_tokens") is not None]
        result[version] = {
            "latency_p50": _percentile(latencies, 0.50),
            "latency_p95": _percentile(latencies, 0.95),
            "tokens_state": "available" if tokens else "unavailable",
            "tokens_total": sum(tokens) if tokens else None,
        }
    return result
```

- [ ] **Step 4: Run tests and lint**

Run: `python3 -m pytest tests/test_paired_metrics.py -v`
Expected: PASS (3 passed). Then `python3 -m ruff check app/statistics tests/test_paired_metrics.py` with no errors.

- [ ] **Step 5: Commit**

```bash
git add app/statistics/__init__.py app/statistics/config.py app/statistics/paired.py tests/test_paired_metrics.py
git commit -m "feat(statistics): add frozen config and descriptive paired metrics"
```

---

### Task 26: Cluster bootstrap for the paired delta

Resample whole clusters with equal cluster weight, recompute v2−v1 over 10,000 replicates, and return percentile 95% CI. Below the frozen minimum cluster count the output is descriptive only.

**Files:**
- Create: `app/statistics/bootstrap.py`
- Test: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bootstrap.py
from types import SimpleNamespace

from app.evaluation.scorer import CaseScore
from app.statistics.bootstrap import bootstrap_delta
from app.statistics.config import load_statistics_config


def _bundle(n_clusters):
    cases = []
    for cluster in range(n_clusters):
        for sub in range(2):
            case_id = f"CASE_{cluster * 2 + sub + 1:03d}"
            cases.append(
                SimpleNamespace(case_id=case_id, statistical_cluster_id=f"CL{cluster:02d}")
            )
    return SimpleNamespace(cases=cases)


def _scores(n_clusters, v1_pass, v2_pass):
    scores = []
    for cluster in range(n_clusters):
        for sub in range(2):
            case_id = f"CASE_{cluster * 2 + sub + 1:03d}"
            scores.append(
                CaseScore(
                    case_id=case_id,
                    agent_version="v1",
                    dimensions=(),
                    task_success="pass" if v1_pass else "fail",
                )
            )
            scores.append(
                CaseScore(
                    case_id=case_id,
                    agent_version="v2",
                    dimensions=(),
                    task_success="pass" if v2_pass else "fail",
                )
            )
    return scores


def test_constant_delta_has_point_estimate_and_ci() -> None:
    bundle = _bundle(12)
    scores = _scores(12, False, True)
    result = bootstrap_delta(bundle, scores, config=load_statistics_config())
    assert result["state"] == "inferential"
    assert result["clusters"] == 12
    assert result["point_delta"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0


def test_below_minimum_clusters_is_descriptive_only() -> None:
    bundle = _bundle(8)
    scores = _scores(8, False, True)
    result = bootstrap_delta(bundle, scores, config=load_statistics_config())
    assert result["state"] == "descriptive_only"
    assert result["point_delta"] == 1.0
    assert "ci_low" not in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bootstrap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.statistics.bootstrap'`.

- [ ] **Step 3: Write the implementation**

```python
# app/statistics/bootstrap.py
import numpy as np

from .config import load_statistics_config


def _cluster_rates(bundle, scores) -> tuple[list[str], np.ndarray, np.ndarray]:
    cluster_of = {case.case_id: case.statistical_cluster_id for case in bundle.cases}
    grouped: dict[str, dict[str, list[int]]] = {}
    for score in scores:
        versions = grouped.setdefault(cluster_of[score.case_id], {"v1": [], "v2": []})
        versions[score.agent_version].append(1 if score.task_success == "pass" else 0)
    cluster_ids = sorted(grouped)
    v1 = np.array(
        [sum(grouped[cluster]["v1"]) / len(grouped[cluster]["v1"]) for cluster in cluster_ids]
    )
    v2 = np.array(
        [sum(grouped[cluster]["v2"]) / len(grouped[cluster]["v2"]) for cluster in cluster_ids]
    )
    return cluster_ids, v1, v2


def bootstrap_delta(bundle, scores, *, config: dict | None = None) -> dict:
    config = config or load_statistics_config()
    cluster_ids, v1, v2 = _cluster_rates(bundle, scores)
    n_clusters = len(cluster_ids)
    point_delta = float(v2.mean() - v1.mean())
    if n_clusters < config["descriptive_only_below_clusters"]:
        return {"state": "descriptive_only", "clusters": n_clusters, "point_delta": point_delta}

    rng = np.random.default_rng(config["bootstrap"]["seed"])
    iterations = config["bootstrap"]["iterations"]
    confidence = config["bootstrap"]["confidence"]
    alpha = 1 - confidence
    deltas = np.empty(iterations)
    for index in range(iterations):
        sampled = rng.integers(0, n_clusters, n_clusters)
        deltas[index] = v2[sampled].mean() - v1[sampled].mean()
    ci_low, ci_high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "state": "inferential",
        "clusters": n_clusters,
        "point_delta": point_delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence": confidence,
        "iterations": iterations,
        "seed": config["bootstrap"]["seed"],
    }
```

- [ ] **Step 4: Run tests and lint**

Run: `python3 -m pytest tests/test_bootstrap.py -v`
Expected: PASS (2 passed). Then `python3 -m ruff check app/statistics/bootstrap.py tests/test_bootstrap.py` with no errors.

- [ ] **Step 5: Commit**

```bash
git add app/statistics/bootstrap.py tests/test_bootstrap.py
git commit -m "feat(statistics): add cluster bootstrap paired delta"
```

---

### Task 27: Cluster-level Exact McNemar

A cluster counts as success only when both of its Cases pass. Run the two-sided Exact McNemar on the 50 cluster pairs; zero discordant pairs return `not_applicable` rather than a p-value.

**Files:**
- Create: `app/statistics/mcnemar.py`
- Test: `tests/test_mcnemar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcnemar.py
from types import SimpleNamespace

from app.evaluation.scorer import CaseScore
from app.statistics.mcnemar import cluster_pairs, exact_mcnemar
from app.statistics.config import load_statistics_config


def _bundle(n_clusters):
    cases = []
    for cluster in range(n_clusters):
        for sub in range(2):
            cases.append(
                SimpleNamespace(
                    case_id=f"CASE_{cluster * 2 + sub + 1:03d}",
                    statistical_cluster_id=f"CL{cluster:02d}",
                )
            )
    return SimpleNamespace(cases=cases)


def _scores(n_clusters, *, v1_pass, v2_pass):
    scores = []
    for cluster in range(n_clusters):
        for sub in range(2):
            case_id = f"CASE_{cluster * 2 + sub + 1:03d}"
            scores.extend(
                [
                    CaseScore(case_id=case_id, agent_version="v1", dimensions=(),
                              task_success="pass" if v1_pass else "fail"),
                    CaseScore(case_id=case_id, agent_version="v2", dimensions=(),
                              task_success="pass" if v2_pass else "fail"),
                ]
            )
    return scores


def test_zero_discordant_is_not_applicable() -> None:
    bundle = _bundle(5)
    scores = _scores(5, v1_pass=True, v2_pass=True)
    pairs = cluster_pairs(bundle, scores)
    result = exact_mcnemar(pairs, config=load_statistics_config())
    assert result["state"] == "not_applicable"
    assert result["p_value"] is None


def test_exact_p_value_for_all_discordant_in_one_direction() -> None:
    bundle = _bundle(12)
    scores = _scores(12, v1_pass=False, v2_pass=True)
    pairs = cluster_pairs(bundle, scores)
    result = exact_mcnemar(pairs, config=load_statistics_config())
    assert result["state"] == "computed"
    assert result["b"] == 0
    assert result["c"] == 12
    assert result["p_value"] == 2 * (0.5 ** 12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mcnemar.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.statistics.mcnemar'`.

- [ ] **Step 3: Write the implementation**

```python
# app/statistics/mcnemar.py
from math import comb

from .config import load_statistics_config


def cluster_pairs(bundle, scores) -> list[tuple[bool, bool]]:
    cluster_of = {case.case_id: case.statistical_cluster_id for case in bundle.cases}
    grouped: dict[str, dict[str, list[int]]] = {}
    for score in scores:
        versions = grouped.setdefault(cluster_of[score.case_id], {"v1": [], "v2": []})
        versions[score.agent_version].append(1 if score.task_success == "pass" else 0)
    return [
        (
            all(grouped[cluster]["v1"]),
            all(grouped[cluster]["v2"]),
        )
        for cluster in sorted(grouped)
    ]


def exact_mcnemar(pairs: list[tuple[bool, bool]], *, config: dict | None = None) -> dict:
    config = config or load_statistics_config()
    v1_only = sum(v1_success and not v2_success for v1_success, v2_success in pairs)
    v2_only = sum(not v1_success and v2_success for v1_success, v2_success in pairs)
    discordant = v1_only + v2_only
    if discordant == 0:
        return {"state": config["mcnemar"]["zero_discordant_state"], "b": 0, "c": 0, "p_value": None}
    extreme = min(v1_only, v2_only)
    tail = sum(comb(discordant, index) for index in range(extreme + 1)) / 2**discordant
    p_value = min(1.0, 2 * tail)
    return {
        "state": "computed",
        "method": config["mcnemar"]["method"],
        "cluster_success_rule": config["mcnemar"]["cluster_success_rule"],
        "b": v1_only,
        "c": v2_only,
        "p_value": p_value,
    }
```

- [ ] **Step 4: Run tests and lint**

Run: `python3 -m pytest tests/test_mcnemar.py -v`
Expected: PASS (2 passed). Then `python3 -m ruff check app/statistics/mcnemar.py tests/test_mcnemar.py` with no errors.

- [ ] **Step 5: Commit**

```bash
git add app/statistics/mcnemar.py tests/test_mcnemar.py
git commit -m "feat(statistics): add cluster exact mcnemar test"
```

---

### Task 28: Aggregate statistics CLI, artifact, and limitations

Compose descriptive metrics, cluster bootstrap, cluster McNemar and latency/token into one immutable statistics artifact that also records the study's limitations. Register `aggregate-statistics`.

**Files:**
- Create: `app/statistics/artifacts.py`
- Create: `app/statistics/cli.py`
- Test: `tests/test_aggregate_statistics.py`
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregate_statistics.py
import json
from pathlib import Path

from app.statistics.cli import aggregate_main


def _evaluation(root: Path) -> Path:
    evaluation = root / "evaluations" / "evaluation-v1__demo__abcd1234"
    evaluation.mkdir(parents=True)

    def verdict(version, task_success):
        return {
            "case_id": "CASE_001",
            "agent_version": version,
            "dimensions": [],
            "task_success": task_success,
        }

    lines = [
        json.dumps(verdict("v1", "fail"), ensure_ascii=False),
        json.dumps(verdict("v2", "pass"), ensure_ascii=False),
    ]
    (evaluation / "case_scores.jsonl").write_text("\n".join(lines) + "\n")
    return evaluation


def test_aggregate_main_descriptive_only_on_single_cluster(tmp_path: Path) -> None:
    evaluation = _evaluation(tmp_path)
    output_root = tmp_path / "statistics"

    exit_code = aggregate_main(
        [
            "--evaluation", str(evaluation),
            "--case-dir", "data/evaluation_cases/v1",
            "--output-root", str(output_root),
        ]
    )

    assert exit_code == 0
    published = next(output_root.iterdir())
    payload = json.loads((published / "statistics.json").read_text())
    assert payload["bootstrap"]["state"] == "descriptive_only"
    assert payload["limitations"]
```

Note: this synthetic test supplies only one cluster of scores against the 100-Case bundle, which makes cluster grouping incomplete; the implementation must tolerate missing clusters by evaluating only clusters present in the scores (descriptive use). The formal path always supplies all 50 clusters.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_aggregate_statistics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.statistics.artifacts'`.

- [ ] **Step 3: Write the implementation**

```python
# app/statistics/artifacts.py
import json
import tempfile
from pathlib import Path

from ..evaluation.loader import load_evaluation_cases
from ..evaluation.scorer import CaseScore
from .bootstrap import bootstrap_delta
from .config import load_statistics_config
from .mcnemar import cluster_pairs, exact_mcnemar
from .paired import descriptive_metrics, latency_token


class StatisticsError(Exception):
    """Raised when statistics inputs are invalid."""


def _load_scores(evaluation_dir: Path) -> tuple[CaseScore, ...]:
    return tuple(
        CaseScore.model_validate(json.loads(line))
        for line in (evaluation_dir / "case_scores.jsonl").read_text().splitlines()
    )


def _load_run_records(run_path: Path) -> tuple[dict, ...]:
    return tuple(
        json.loads(line) for line in (run_path / "case_runs.jsonl").read_text().splitlines()
    )


_LIMITATIONS = [
    "single execution per case (n_repeats=1); temperature 0 reduces but does not guarantee provider-level determinism",
    "100 synthetic cases and 50 clusters; results do not generalize to production distributions",
    "cluster bootstrap and exact McNemar are exploratory, not corrected for multiple comparisons",
    "tokens/cost are unavailable when the provider does not return usage; latency is environment-dependent",
    "cases marked needs_review are excluded from pass counts and must be resolved before causal claims",
]


def write_statistics(
    *,
    evaluation_dir: Path,
    case_dir: Path,
    output_root: Path,
    v1_run_path: Path | None = None,
    v2_run_path: Path | None = None,
) -> Path:
    bundle = load_evaluation_cases(case_dir)
    scores = _load_scores(evaluation_dir)
    config = load_statistics_config()

    # restrict the bundle to clusters represented in the scores
    score_clusters = {
        next(
            case.statistical_cluster_id
            for case in bundle.cases
            if case.case_id == score.case_id
        )
        for score in scores
    }
    scoped_cases = [case for case in bundle.cases if case.statistical_cluster_id in score_clusters]
    scoped_bundle = type(bundle)(manifest=bundle.manifest, cases=scoped_cases, manifest_sha256=bundle.manifest_sha256)

    payload = {
        "evaluation_id": evaluation_dir.name,
        "descriptive": descriptive_metrics(scores),
        "bootstrap": bootstrap_delta(scoped_bundle, scores, config=config),
        "mcnemar": exact_mcnemar(cluster_pairs(scoped_bundle, scores), config=config),
        "limitations": _LIMITATIONS,
    }
    if v1_run_path is not None and v2_run_path is not None:
        records = _load_run_records(v1_run_path) + _load_run_records(v2_run_path)
        payload["latency_token"] = latency_token(records)

    statistics_id = evaluation_dir.name.replace("evaluation", "statistics", 1)
    target = output_root / statistics_id
    if target.exists():
        raise StatisticsError(f"refusing to overwrite {statistics_id}")

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root) as temporary:
        staging = Path(temporary)
        canonical = (
            json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        (staging / "statistics.json").write_text(canonical)
        staging.rename(target)
    return target
```

```python
# app/statistics/cli.py
import argparse
from pathlib import Path

from .artifacts import StatisticsError, write_statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aggregate-statistics")
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--case-dir", default=Path("data/evaluation_cases/v1"), type=Path)
    parser.add_argument("--v1-run", default=None, type=Path)
    parser.add_argument("--v2-run", default=None, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/statistics"), type=Path)
    return parser


def aggregate_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = write_statistics(
            evaluation_dir=args.evaluation,
            case_dir=args.case_dir,
            output_root=args.output_root,
            v1_run_path=args.v1_run,
            v2_run_path=args.v2_run,
        )
    except (StatisticsError, FileNotFoundError, ValueError) as exc:
        print(f"aggregate-statistics: {exc}")
        return 1
    print(f"statistics published: {path}")
    return 0
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, add to `[project.scripts]`:

```toml
aggregate-statistics = "app.statistics.cli:aggregate_main"
```

- [ ] **Step 5: Add the Makefile target and ignore rule**

In the `Makefile`:

```makefile
.PHONY: aggregate-statistics
aggregate-statistics:
	aggregate-statistics --evaluation $(EVALUATION) --v1-run $(V1_RUN) --v2-run $(V2_RUN)
```

In `.gitignore`:

```gitignore
outputs/statistics/
```

- [ ] **Step 6: Run tests, lint, and verify the entry point**

Run: `python3 -m pytest tests/test_aggregate_statistics.py -v`
Expected: PASS (1 passed). Then:

`python3 -m ruff check app/statistics/artifacts.py app/statistics/cli.py tests/test_aggregate_statistics.py` with no errors.
`pip install -e .` then `aggregate-statistics --help` should print usage.

- [ ] **Step 7: Run the full suite**

Run: `python3 -m pytest`
Expected: entire suite green, including Phase 3 frozen-asset tests.

- [ ] **Step 8: Commit**

```bash
git add app/statistics/artifacts.py app/statistics/cli.py tests/test_aggregate_statistics.py pyproject.toml Makefile .gitignore
git commit -m "feat(statistics): add aggregate cli artifact and limitations"
```

### Batch E gate

- [ ] **Gate E-1: formal statistics published**

Run (on the formal evaluation from Task 22 and formal Run paths from Task 20):

```bash
aggregate-statistics \
  --evaluation outputs/evaluations/<evaluation-id> \
  --v1-run outputs/real_runs/<v1-run-id> \
  --v2-run outputs/real_runs/<v2-run-id>
```

Expected: exit `0`; one directory under `outputs/statistics/` containing the paired rates, cluster bootstrap on the 50 clusters (10,000 replicates, seed `20261009`, v2−v1 delta with percentile 95% CI), Exact McNemar on 50 cluster pairs (or `not_applicable`), latency/token, and the limitations list. No model calls were made.

---

## Final: acceptance checklist

- [ ] Pilot: 20 Development Cases, human-supervised, technical failure ≤ 10% (Task 12).
- [ ] Formal: 100 frozen Cases, 200 logical records, alternating order, fairness verified (Batch C).
- [ ] Scoring: 200 deterministic CaseScores, manifest bound to Runs/Bundle/Rubric/scorer SHA (Task 17).
- [ ] Attribution: E1–E10 earliest-causal, 200 RCA records, Error Shift matrix (Batch D).
- [ ] Statistics: cluster bootstrap + Exact McNemar + latency/token + limitations (Batch E).
- [ ] At no point were the API key, Gold, or expected answers written into Run artifacts.



















