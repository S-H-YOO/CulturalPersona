## 🧪 Experiment Overview

| | |
|---|---|
| **Model** | Qwen2.5-14B-Instruct |
| **Task** | Cultural Adaptation via LLM Debate |

> **Research Question:** LLM 간의 Debate(Interaction)을 통한 Cultural Adaptation 성능 향상이  
> 단순히 *더 많은 정보를 알고 Test를 하기 때문*인지 검증하는 확인 실험

---

## ⚙️ Experiment Conditions

### `A` — Interaction + Target Persona
```
Korea Persona ↔ Target Cultural Persona
```
실제 target cultural persona와 직접 대화하는 조건

### `B` — Interaction + JSON Agent
```
Korea Persona ↔ JSON Cultural Background Agent
```
정리된 JSON cultural background를 주입한 agent와 대화하는 조건

---

## ❓ Core Comparison

| Condition | Partner | Question |
|-----------|---------|----------|
| **A** | Real target cultural persona | 실제 페르소나와의 대화가 더 효과적인가? |
| **B** | JSON-based cultural agent | 정리된 배경지식 주입만으로도 동등한 효과를 내는가? |
