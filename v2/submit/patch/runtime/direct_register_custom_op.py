import torch
from torch.library import Library, infer_schema
from collections.abc import Callable


DEFAULT_DISPATCH_KEY = "CUDA"


# 从vllm中迁移过来的，用于注册自定义op
def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
):
    if mutates_args is None:
        mutates_args = []

    if target_lib is None:
        raise TypeError(
            "direct_register_custom_op(...): target_lib must not be None. "
            "Create once: `from torch.library import Library` then "
            "`lmslim_lib = Library('lmslim', 'FRAGMENT')` and pass `target_lib=lmslim_lib`."
        )

    if dispatch_key is None:
        dispatch_key = DEFAULT_DISPATCH_KEY

    schema_str = infer_schema(op_func, mutates_args=mutates_args)

    my_lib = target_lib
    # DUP_REG_GUARD (generic, idempotent registration):
    # On gfx928, lmslim/envs.py hardcodes LMSLIM_USE_LIGHTOP=0 (the env-var
    # branch is only taken for gfx936/gfx938), so lmslim.quantize.quant_ops
    # registers ops into the same "lmslim" torch library namespace that this
    # module also registers into. Whichever module is imported second raises
    # RuntimeError("... Duplicate registration ...") and kills the SGLang
    # server at import time (sglang .../fp8_utils.py imports lightop.gemm_ops).
    # Make registration idempotent: if the op is already defined, keep the
    # existing implementation instead of raising.
    try:
        my_lib.define(op_name + schema_str, tags=tags)
    except RuntimeError as _e:  # DUP_REG_GUARD
        if op_name in str(_e):
            print(
                f"[lightop] DUP_REG_GUARD: {op_name} already registered by "
                f"another module, keeping existing impl",
                flush=True,
            )
            return
        raise
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)