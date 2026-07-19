package thinking
import "testing"

// Both caller dialects fold to one neutral depth, and each upstream projects it back
// into its own shape. This pins the two new pieces: OpenAI's "minimal" floor, and the
// Anthropic outbound (depth -> thinking budget) so routing to opus/fable is native.
func TestMinimalAndAnthropicMapping(t *testing.T) {
	if Effort("minimal") != Minimal {
		t.Fatal("reasoning_effort=minimal must fold to Minimal")
	}
	if Effort("off") != Off || Effort("disabled") != Off {
		t.Fatal("explicit off/disabled must fold to Off")
	}
	// OpenAI keeps minimal; GLM has no floor so it rounds up to high.
	if got := Minimal.Fields(OpenAI)["reasoning_effort"]; got != "minimal" {
		t.Errorf("OpenAI Minimal = %v, want minimal", got)
	}
	if got := Minimal.Fields(GLM)["reasoning_effort"]; got != "high" {
		t.Errorf("GLM Minimal = %v, want high (coarse floor)", got)
	}
	// Anthropic outbound: a depth becomes a native thinking object with a budget.
	hi := High.Fields(Anthropic)["thinking"].(map[string]any)
	if hi["type"] != "enabled" || hi["budget_tokens"].(int) != deep {
		t.Errorf("Anthropic High = %v, want {enabled, %d}", hi, deep)
	}
	if len(Off.Fields(Anthropic)) != 0 {
		t.Error("Anthropic Off must emit no thinking block")
	}
	// Budget round-trips through the same neutral value both ways.
	if Budget(32768) != Max || Max.Tokens() != ultra {
		t.Error("Budget/Tokens must be inverse at the ultra tier")
	}
}
