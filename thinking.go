// Copyright 2026 Hanzo AI Inc. All Rights Reserved.

// Package thinking folds how-hard-to-think into the native shape each upstream
// model accepts. Callers speak two dialects — Anthropic sends a token budget,
// OpenAI sends an effort ordinal — and every upstream reasons by its own control:
// GLM/DeepSeek by a reasoning_effort ordinal, Qwen by an enable_thinking gate,
// Kimi by a thinking object. Between them sits ONE neutral value, a Depth ordinal:
// each dialect is a constructor into it, each upstream a projection out of it.
//
// This is the sole home of that map, so ai and zen share one table (as they
// already share hanzoai/decimal and hanzoai/money). The fold is total and
// monotone — deeper in, harder out — and lossy where a vocabulary is coarser.
package thinking

// Depth is how hard to think, as an ordinal — the neutral value both wire dialects
// normalize into and every upstream projects out of. Off means the caller declined
// to think; the rest ascend.
type Depth int

const (
	Off  Depth = iota // don't think — the instant path
	Low               // light
	Mid               // moderate
	High              // hard
	Max               // hardest
)

// deep is the token budget at or above which a request is the hardest tier — Claude
// Code's "ultrathink" is ~32k, "think" ~4k. mid is the moderate threshold.
const (
	mid  = 4096
	deep = 16384
)

// Budget folds an Anthropic thinking.budget_tokens into a Depth. Zero (thinking
// absent or disabled) is Off; any positive budget is on, at a tier set by size.
func Budget(tokens int) Depth {
	switch {
	case tokens <= 0:
		return Off
	case tokens < mid:
		return Low
	case tokens < deep:
		return Mid
	default:
		return Max
	}
}

// Effort folds an OpenAI reasoning_effort ordinal into a Depth. An unknown or empty
// value is Off.
func Effort(ordinal string) Depth {
	switch ordinal {
	case "low":
		return Low
	case "medium":
		return Mid
	case "high":
		return High
	case "max":
		return Max
	default:
		return Off
	}
}

// On reports whether the caller asked to think at all. Off writes no field, leaving
// the upstream at its own default.
func (d Depth) On() bool { return d > Off }

// Vocab is the reasoning control an upstream accepts, keyed by request shape — not
// by brand. Pick it with Of.
type Vocab string

const (
	GLM    Vocab = "glm"    // reasoning_effort ∈ {none, high, max}: GLM-5.*, DeepSeek V4, the DO-AI default (reasons unless told none)
	OpenAI Vocab = "openai" // reasoning_effort ∈ {low, medium, high}: o-series
	Qwen   Vocab = "qwen"   // enable_thinking gate
	Kimi   Vocab = "kimi"   // a thinking object {type, preserve_thinking}
)

// Of returns the vocabulary an upstream model id accepts, keyed by id prefix — the
// one place brand maps to control shape. Unknown ids fold as GLM: the DO-AI family
// (glm-5.*, deepseek-*, minimax-*, nemotron-*, llama*) is the universal upstream and
// its two-tier ordinal is the safe supposition.
func Of(upstream string) Vocab {
	m := lower(upstream)
	switch {
	case prefix(m, "qwen") || prefix(m, "alibaba-qwen"):
		return Qwen
	case prefix(m, "kimi"):
		return Kimi
	case prefix(m, "gpt-") || prefix(m, "openai-gpt") ||
		prefix(m, "o1") || prefix(m, "o3") || prefix(m, "o4"):
		return OpenAI
	default:
		return GLM
	}
}

// Fields projects the depth into the request fields a vocabulary accepts, as a map
// to merge into an upstream chat body — {} when the vocabulary takes no field for the
// depth. It is the sole producer of these keys, so a caller clears Keys first: a stale
// field from another vocabulary must never reach this upstream.
//
// Off is not always empty: a GLM-family upstream reasons BY DEFAULT, so Off (the instant
// path) must actively send reasoning_effort:"none" — the absence of the field would leave
// the model reasoning, streaming a long content:null preamble that reads as an empty
// completion. Vocabularies that default OFF (Qwen/Kimi gates, OpenAI o-series) write
// nothing for Off, as before.
func (d Depth) Fields(v Vocab) map[string]any {
	switch v {
	case GLM, OpenAI:
		if e := d.ordinal(v); e != "" {
			return map[string]any{"reasoning_effort": e}
		}
	case Qwen:
		if d.On() {
			return map[string]any{"enable_thinking": true}
		}
	case Kimi:
		if d.On() {
			think := map[string]any{"type": "enabled"}
			if d == Max {
				think["preserve_thinking"] = "all" // keep the trace across turns for long-horizon coding
			}
			return map[string]any{"thinking": think}
		}
	}
	return map[string]any{}
}

// ordinal folds the depth to a reasoning_effort value valid for an ordinal
// vocabulary (GLM or OpenAI). GLM reasons by default, so Off maps to "none" (silence the
// preamble) and it has no low/medium — Low/Mid/High coerce up to "high", Max to "max".
// OpenAI's o-series reasons only when asked, so Off maps to "" (no field); it has
// {low,medium,high} and no "max", so Max coerces down to "high".
func (d Depth) ordinal(v Vocab) string {
	switch v {
	case OpenAI:
		switch d {
		case Low:
			return "low"
		case Mid:
			return "medium"
		case High, Max:
			return "high"
		}
	case GLM:
		switch d {
		case Off:
			// GLM-family upstreams on DO-AI (glm-5.*, deepseek-*, minimax-*, …)
			// REASON BY DEFAULT: sending no reasoning_effort makes them emit a long
			// content:null reasoning stream before any answer token, which a plain
			// OpenAI client renders as an empty completion. Off means "instant path",
			// so it must actively request none — the one value that silences the
			// preamble and streams the answer immediately. (Verified against DO-AI:
			// glm-5.2/deepseek-v4-pro honor "none" and emit zero reasoning chunks.)
			return "none"
		case Low, Mid, High:
			return "high"
		case Max:
			return "max"
		}
	}
	return ""
}

// Keys is every field any vocabulary may write. A caller clears these before
// applying Fields so the fold is the sole writer and nothing leaks across a
// vocabulary boundary.
var Keys = []string{"reasoning_effort", "enable_thinking", "thinking_budget", "thinking"}

// lower and prefix keep the package free of a strings import — a shared leaf earns
// its keep by staying minimal.
func lower(s string) string {
	b := []byte(s)
	for i, c := range b {
		if c >= 'A' && c <= 'Z' {
			b[i] = c + ('a' - 'A')
		}
	}
	return string(b)
}

func prefix(s, p string) bool { return len(s) >= len(p) && s[:len(p)] == p }
