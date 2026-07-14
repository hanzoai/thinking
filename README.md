# thinking

Fold *how hard to think* into the native shape each upstream model accepts.

Callers speak two dialects — Anthropic sends a token **budget**, OpenAI sends an
effort **ordinal** — and every upstream reasons by its own control. Between them
sits one neutral value, a `Depth` ordinal: each dialect is a constructor into it,
each upstream a projection out of it.

```go
d := thinking.Budget(32000)        // Anthropic "ultrathink" → Max
d := thinking.Effort("high")       // OpenAI reasoning_effort → High

v := thinking.Of("glm-5.2")        // GLM
d.Fields(v)                        // {"reasoning_effort": "max"}

thinking.Max.Fields(thinking.Of("kimi-k2.6"))
// {"thinking": {"type": "enabled", "preserve_thinking": "all"}}

thinking.High.Fields(thinking.Of("qwen3.5-397b-a17b"))
// {"enable_thinking": true}
```

One home for the map, so every serving layer folds the same way. Clear
`thinking.Keys` on the outbound body before merging `Fields`, so the fold is the
sole writer and no field leaks across a vocabulary boundary.

## Vocabularies

| Vocab    | upstreams                                   | control                                   |
|----------|---------------------------------------------|-------------------------------------------|
| `GLM`    | GLM-5.\*, DeepSeek V4, the DO-AI default     | `reasoning_effort ∈ {high, max}`          |
| `OpenAI` | o-series                                     | `reasoning_effort ∈ {low, medium, high}`  |
| `Qwen`   | Qwen 3.\*                                     | `enable_thinking` gate                    |
| `Kimi`   | Kimi K2.\*                                    | `thinking` object (`preserve_thinking`)   |

The fold is total and monotone — deeper in, harder out — and lossy where a
vocabulary is coarser (GLM has no low/medium; OpenAI has no max).
