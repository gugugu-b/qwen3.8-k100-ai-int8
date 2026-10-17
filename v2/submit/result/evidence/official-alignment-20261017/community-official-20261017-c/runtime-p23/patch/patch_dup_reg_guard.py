import pathlib

p = pathlib.Path(
    "/usr/local/lib/python3.10/dist-packages/"
    "lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py"
)
src = p.read_text()

if "DUP_REG_GUARD" in src:
    print("already patched")
    raise SystemExit(0)

old = """    schema_str = infer_schema(op_func, mutates_args=mutates_args)

    my_lib = target_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)"""

new = """    schema_str = infer_schema(op_func, mutates_args=mutates_args)

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
        my_lib._register_fake(op_name, fake_impl)"""

if old not in src:
    raise SystemExit("ANCHOR NOT FOUND - file layout differs")

p.write_text(src.replace(old, new))
print("patched OK (generic guard)")
