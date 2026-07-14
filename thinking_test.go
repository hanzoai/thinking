// Copyright 2026 Hanzo AI Inc. All Rights Reserved.

package thinking

import (
	"encoding/json"
	"testing"
)

// Both wire dialects normalize into the same Depth ordinal: an Anthropic budget and
// an OpenAI effort that mean the same thing land on the same value.
func TestConstructors(t *testing.T) {
	cases := []struct {
		name string
		got  Depth
		want Depth
	}{
		{"budget 0 → off", Budget(0), Off},
		{"budget negative → off", Budget(-1), Off},
		{"budget 2k → low", Budget(2048), Low},
		{"budget 4k → mid", Budget(4096), Mid},
		{"budget 10k → mid", Budget(10000), Mid},
		{"budget 16k → max", Budget(16384), Max},
		{"budget 32k → max", Budget(32000), Max},
		{"effort '' → off", Effort(""), Off},
		{"effort low", Effort("low"), Low},
		{"effort medium → mid", Effort("medium"), Mid},
		{"effort high", Effort("high"), High},
		{"effort max", Effort("max"), Max},
		{"effort junk → off", Effort("turbo"), Off},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got %d, want %d", c.name, c.got, c.want)
		}
	}
}

func TestOf(t *testing.T) {
	cases := map[string]Vocab{
		"glm-5.2":            GLM,
		"deepseek-v4-pro":    GLM,
		"deepseek-4-flash":   GLM,
		"minimax-m2.5":       GLM,
		"llama3.3-70b":       GLM,
		"qwen3.5-397b-a17b":  Qwen,
		"alibaba-qwen3-32b":  Qwen,
		"kimi-k2.6":          Kimi,
		"gpt-5.3-codex":      OpenAI,
		"o3-mini":            OpenAI,
		"something-unknown":  GLM,
	}
	for id, want := range cases {
		if got := Of(id); got != want {
			t.Errorf("Of(%q) = %q, want %q", id, got, want)
		}
	}
}

// The reasoning_effort ordinal projection: GLM has only {high,max}; OpenAI has
// {low,medium,high} and no max.
func TestOrdinal(t *testing.T) {
	cases := []struct {
		depth Depth
		vocab Vocab
		want  string
	}{
		{Off, GLM, ""},
		{Low, GLM, "high"}, // GLM coerces sub-max up to high
		{Mid, GLM, "high"},
		{High, GLM, "high"},
		{Max, GLM, "max"},
		{Off, OpenAI, ""},
		{Low, OpenAI, "low"},
		{Mid, OpenAI, "medium"},
		{High, OpenAI, "high"},
		{Max, OpenAI, "high"}, // OpenAI has no max; coerce down to high
	}
	for _, c := range cases {
		if got := c.depth.ordinal(c.vocab); got != c.want {
			t.Errorf("%d.ordinal(%q) = %q, want %q", c.depth, c.vocab, got, c.want)
		}
	}
}

// Fields projects the depth into each upstream's native request shape.
func TestFields(t *testing.T) {
	j := func(m map[string]any) string { b, _ := json.Marshal(m); return string(b) }

	// GLM / OpenAI → reasoning_effort.
	if got := j(Max.Fields(GLM)); got != `{"reasoning_effort":"max"}` {
		t.Errorf("Max GLM → %s", got)
	}
	if got := j(Mid.Fields(OpenAI)); got != `{"reasoning_effort":"medium"}` {
		t.Errorf("Mid OpenAI → %s", got)
	}
	// Off writes nothing on any vocabulary.
	for _, v := range []Vocab{GLM, OpenAI, Qwen, Kimi} {
		if got := j(Off.Fields(v)); got != `{}` {
			t.Errorf("Off %q → %s, want {}", v, got)
		}
	}
	// Qwen → enable_thinking gate (no budget cap: the ordinal has none).
	if got := j(High.Fields(Qwen)); got != `{"enable_thinking":true}` {
		t.Errorf("High Qwen → %s", got)
	}
	// Kimi → thinking object; only Max preserves the trace across turns.
	if got := j(High.Fields(Kimi)); got != `{"thinking":{"type":"enabled"}}` {
		t.Errorf("High Kimi → %s", got)
	}
	if got := j(Max.Fields(Kimi)); got != `{"thinking":{"preserve_thinking":"all","type":"enabled"}}` {
		t.Errorf("Max Kimi → %s", got)
	}
}

func TestOn(t *testing.T) {
	if Off.On() {
		t.Error("Off.On() = true, want false")
	}
	for _, d := range []Depth{Low, Mid, High, Max} {
		if !d.On() {
			t.Errorf("%d.On() = false, want true", d)
		}
	}
}
