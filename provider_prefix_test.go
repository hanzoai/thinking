package thinking
import "testing"

// Gateway slugs carry a provider prefix; the vocabulary must key on the model family
// underneath it, or an o-series call sends reasoning_effort=max (which o-series rejects).
func TestOf_StripsProviderPrefix(t *testing.T) {
	cases := map[string]Vocab{
		"openai-o3":            OpenAI,
		"openai-o1":            OpenAI,
		"openai-gpt-oss-120b":  OpenAI,
		"gpt-5.6-sol":          OpenAI,
		"alibaba-qwen3-32b":    Qwen,
		"qwen3.5-397b-a17b":    Qwen,
		"kimi-k2.6":            Kimi,
		"deepseek-v4-pro":      GLM,
		"glm-5.2":              GLM,
		"nvidia-nemotron-3-super-120b": GLM,
	}
	for id, want := range cases {
		if got := Of(id); got != want {
			t.Errorf("Of(%q) = %q, want %q", id, got, want)
		}
	}
	// the bug: o-series must never receive "max"
	if e := Max.Fields(Of("openai-o3"))["reasoning_effort"]; e != "high" {
		t.Errorf("openai-o3 at Max must coerce to high, got %v", e)
	}
}
