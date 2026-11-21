# Running the examples

Run `torchrun --standalone --nproc-per-node=2 ./parallelism_examples/tensor_parallel_swiglu.py`

## Ring attention

Run `uv run torchrun --standalone --nproc-per-node=4 ./parallelism_examples/ring_attention_context_parallel.py`.

## GPipe pipeline parallelism

Run `uv run torchrun --standalone --nproc-per-node=8 ./parallelism_examples/pipeline_parallel_gpipe.py`.

## Block matrix multiplication

Run `uv run python ./parallelism_examples/block_matmul_check.py`.

If you encounter some IPV6 error due to c10d, use this:

```bash
torchrun \
    --nnodes=1 \
    --nproc-per-node=2 \
    --rdzv-backend=static \
    --master-addr=127.0.0.1 \
    --master-port=29500 \
    ./parallelism_examples/tensor_parallel_swiglu.py
```
