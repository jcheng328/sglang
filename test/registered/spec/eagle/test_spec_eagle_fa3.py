"""fa3 attention backend -- Hopper-only (FlashAttention-3 is sm_90).

fa3 is the real H200 default for MHA spec at topk=1, so this also covers the
"what an H200 user actually runs" path. Requires the large (Hopper) runner.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecPerfKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_cuda_ci(est_time=780, stage="base-b", runner_config="1-gpu-large")


class TestEagle3Fa3(Eagle3Base, SpecCorrectnessKit, SpecAccuracyKit, SpecLogprobKit):
    """EAGLE3 spec v2 topk=1 on fa3 (the H200 default backend)."""

    attention_backend = "fa3"
    disable_overlap = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Fa3Page256(
    EagleLlama2Base,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecPerfKit,
    SpecFeatureKit,
):
    """EAGLE/Llama-2 topk=5 tree on fa3 + page_size=256 (spec v1)."""

    spec_topk = 5
    spec_steps = 8
    attention_backend = "fa3"
    page_size = 256
    chunked_prefill_size = 4096  # must be divisible by page_size (256)
    cuda_graph_max_bs_decode = 5
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Fa3PageMajor(
    EagleLlama2Base,
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
):
    """EAGLE/Llama-2 topk=8 tree on fa3 + --enable-page-major-kv-layout.

    page_size stays at the fixture default (1): server_args rejects fa3 +
    page-major + page_size>1 + topk>1 together (the cascade-attention
    "expand" path would reshape-copy the whole per-layer K/V pool under the
    envelope's non-page-contiguous page stride). This is the only test that
    exercises that expand path (flashattention_backend's use_cascade_attn
    branch) under the page-major layout -- the one FA3 code change this
    feature makes.
    """

    attention_backend = "fa3"
    extra_args = ("--enable-page-major-kv-layout",)
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()
