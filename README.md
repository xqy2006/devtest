# modeltrace-pair (research branch)

Offline experiments separating GPT-6 Astra from GPT-6.1 Sol by the numbers in their responses
(numbers only; no model API calls). Runs on GitHub-hosted runners because the local machine has
2 CPUs and ~250MB of free memory.

- `research/runner_pair.py`: experiments (see its docstring); `research/pilot_long_eval.py`: the
  pre-registered N=1000 pilot evaluation (`--adapt` part).
- `data/`: historical panel, the 2026-10-11 Codex pair supplement, and the 4Router N=1000 pilot
  (`accepted.jsonl`; labels are the relay's reported model names).
- Trigger: change the `RUN` file and push; results arrive as the `all-results` artifact.
