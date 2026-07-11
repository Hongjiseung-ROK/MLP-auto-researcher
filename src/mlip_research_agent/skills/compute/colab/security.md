# Security notes: colab_preflight

## Authentication boundary
- CLI authentication is **external and user-owned**: `gcloud auth
  application-default login` with scopes `openid`, `cloud-platform`,
  `userinfo.email`, `colaboratory`. Verify with `colab sessions` /
  `colab whoami`. The skill and the compute adapters never read, write,
  copy, or log the ADC token (`~/.config/colab-cli/token.json`).
- Never paste tokens into chat, source files, or shell commands that enter
  history. There is no code path in this repo that accepts a raw token.

## What this skill never does
- Never runs interactive CLI commands (`colab auth`, `colab drivemount`,
  `colab repl`, `colab console`) — they hang non-TTY agents and `colab auth`
  injects VM-side GCP credentials, which this workflow does not use.
- Never reads browser cookies, Drive contents, SSH keys, or workspace files
  outside `ctx.step_dir` and the policy/config files it validates.
- Never starts a session for a full campaign: workload classes are
  validator-bounded to preflight/smoke/failure-reproduction.

## Session hygiene (real mode, when enabled)
- Sessions are stopped in a `finally` block (mirrors `colab run` semantics:
  self-cleaning even on error) so idle VMs never burn compute units.
- Every log line passes `router.redact_secrets` before persistence.
- The workload runs only after the immutable attestation gets ALLOW; a T4,
  H100, or unknown device terminates the job as `policy_denied`.

## Reviewed upstream
`google-colab-cli` v0.6.0 (official googlecolab GitHub org, Apache-2.0) —
full review in `docs/SKILL_SECURITY_REVIEW.md`, including the two behaviors
that shaped this design: A100 40/80GB are indistinguishable at request time,
and unrecognized `--gpu` values silently fall back to A100. Both are
neutralized by runtime attestation against `configs/compute_policy.yaml`.
