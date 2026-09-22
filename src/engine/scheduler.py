"""
Dynamic Step Scheduler for Project Reflex.
Allocates diffusion denoising steps and canvas footprint dynamically based on schema complexity:
  - Tier 1 (Atomic): 1 step, L <= 8
  - Tier 2 (Parametric Primitive): 2-4 steps, L in [8, 24] with convergence early-stopping
  - Tier 3 (Generative Synthesis): 12-20 steps, L in [64, 256]
"""

from dataclasses import dataclass
from typing import Optional
from ..schema.inspector import Tier, ToolComplexity


@dataclass
class ExecutionPlan:
    """Execution plan calculated by DynamicStepScheduler."""
    tool_name: str
    tier: Tier
    canvas_length: int
    max_denoise_steps: int
    early_stopping: bool = True
    min_steps: int = 1


class DynamicStepScheduler:
    """
    Adaptive Step Allocator for Project Reflex.
    Enforces the core principle: Compute Matches Entropy.
    """

    def __init__(self, default_generative_steps: int = 20, default_generative_length: int = 128):
        self.default_generative_steps = default_generative_steps
        self.default_generative_length = default_generative_length

    def schedule(
        self,
        complexity: ToolComplexity,
        requested_max_tokens: Optional[int] = None,
    ) -> ExecutionPlan:
        """
        Calculates optimal execution plan for a tool call.

        Args:
            complexity: Analyzed ToolComplexity profile.
            requested_max_tokens: Optional user-specified max token ceiling.

        Returns:
            ExecutionPlan with compute budget and canvas allocation.
        """
        if complexity.tier == Tier.ATOMIC:
            plan = ExecutionPlan(
                tool_name=complexity.tool_name,
                tier=Tier.ATOMIC,
                canvas_length=4,
                max_denoise_steps=1,
                early_stopping=False,
                min_steps=1,
            )
        elif complexity.tier == Tier.PARAMETRIC_PRIMITIVE:
            # Low-entropy primitive parameters: 2 to 4 steps is sufficient
            plan = ExecutionPlan(
                tool_name=complexity.tool_name,
                tier=Tier.PARAMETRIC_PRIMITIVE,
                canvas_length=complexity.estimated_canvas_length,
                max_denoise_steps=min(4, max(2, complexity.recommended_denoise_steps)),
                early_stopping=True,
                min_steps=2,
            )
        else:
            # Unbounded generative synthesis
            gen_len = requested_max_tokens or complexity.estimated_canvas_length or self.default_generative_length
            gen_len = min(max(gen_len, 64), 256)
            plan = ExecutionPlan(
                tool_name=complexity.tool_name,
                tier=Tier.GENERATIVE_SYNTHESIS,
                canvas_length=gen_len,
                max_denoise_steps=self.default_generative_steps,
                early_stopping=False,
                min_steps=12,
            )

        print(
            f"[SCHEDULER] Tool: {plan.tool_name} | "
            f"Tier: {plan.tier.value.upper()} | "
            f"Canvas: {plan.canvas_length} | "
            f"Steps: {plan.max_denoise_steps}"
        )
        return plan

