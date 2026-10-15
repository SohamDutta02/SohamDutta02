I build inference infrastructure for large language models: KV-cache memory management, request scheduling, and the benchmarks that show whether a change actually helped. Mostly Rust, CUDA and Python.

Looking for roles in ML infrastructure and inference engineering.\
[sohamdutta.dev](https://sohamdutta.dev) · [sohamsouthpoint02@gmail.com](mailto:sohamsouthpoint02@gmail.com) · [LinkedIn](https://www.linkedin.com/in/soham-dutta-0ab074361)

### Projects

**[paged-kv-cache](https://github.com/SohamDutta02/paged-kv-cache)** · Rust, CUDA · [live demo](https://paged-kv-cache.onrender.com)\
A KV-cache memory manager modeled on vLLM's PagedAttention, written from scratch: a refcounted block allocator, per-sequence block tables with copy-on-write forking, prefix caching, a continuous-batching scheduler with preemption, and a CUDA backend. It runs Qwen2.5-0.5B with every layer's K/V in its blocks, generates the same tokens as Hugging Face `transformers`, and serves an OpenAI-compatible API with Prometheus metrics.
- Fits 4.6× the batch of max-length reservation, and 1.5× that of an allocator told every output length in advance.
- With prefix caching, a chat workload where most requests share a system prompt computes 69% fewer prompt tokens.

**[faultline](https://github.com/SohamDutta02/Chaos-engineering-harness-for-agent-runtimes)** · Python, TypeScript\
A fault-injection harness for tool-using LLM agents. A proxy between the agent and its tools injects rate limits, timeouts, truncated JSON, stale or subtly wrong data, and writes that commit but report failure. Each trial is graded on the state of the world the agent leaves behind, and fault schedules are seeded so every architecture faces the same failures.
- In the control sweep, client retries lift the pass rate under rate limiting from 50% to 94%, but do nothing for stale or wrong data.

### Working on

- **Communication–compute overlap:** PyTorch autograd hooks that bucket gradients and launch NCCL all-reduce while the backward pass is still running.
- **RadixAttention benchmarks:** SGLang's prefix reuse on multi-turn agent sessions, multi-agent fan-out, RAG pipelines and self-consistency sampling.

### Upstream

- [pytorch/pytorch#200950](https://github.com/pytorch/pytorch/issues/200950): every `start_processes` example in the `torch.distributed.elastic` docs uses the pre-`LogsSpecs` signature and fails.

### Before this

Research on stochastic models at IIT Bombay's FedEx Centre for Advanced Logistics and Analytics (MDPs, MCMC samplers, SDE models of asset prices) and at the Indian Statistical Institute, Kolkata (simulated annealing to search for mutually unbiased bases). Amazon ML Summer School scholar; Indian Academy of Sciences Summer Research Fellow. B.Tech in CSE from Heritage Institute of Technology, Kolkata.
