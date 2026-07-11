# Third-Party SKILL Security Review

Policy: any externally sourced SKILL is untrusted until reviewed here. A skill
is enabled only after source identification, script inspection, credential-
behavior analysis, and version pinning.

---

## Review 1: `colab-research` workspace skill + `google-colab-cli`

- **Reviewed**: 2026-07-11, by the research agent (session-recorded).
- **Decision**: **ACCEPT with caveats** (below). Pinned version: v0.6.0.

### Candidates considered
`colab-research` (CLI wrapper skill — selected), `colab-finetune`,
`colab-unsloth-fine-tuning`, `colab-chemsmart-config-test` (notebook-recipe
skills, not CLI wrappers — not applicable for the compute adapter).

### Source and version
- Skill: `~/.claude/skills/colab-research/SKILL.md` — documentation only, **no
  executable scripts** in the skill directory.
- Underlying tool: `google-colab-cli` **v0.6.0**, installed via `uv tool`
  (entrypoint `~/.local/bin/colab`), metadata Homepage/Repository:
  `https://github.com/googlecolab/google-colab-cli` (official Google Colab
  GitHub organization), License: Apache-2.0.

### Script inspection (20 Python files in `colab_cli/`)
- **No** `verify=False` / TLS bypass.
- **No** browser-cookie or keychain access (`browser_cookie`, `cookiejar`,
  `.mozilla`, `Chrome/Default`, `keychain`: zero hits).
- **No** obfuscated execution (`eval(`, `exec(compile`: zero hits). The two
  `base64.b64decode` uses are legitimate Jupyter-protocol payloads (notebook
  file contents in `contents.py:84`, image outputs in `utils.py:89`).
- Network endpoints limited to Google services: `colab.pa.googleapis.com`,
  `colab.research.google.com`, `oauth2.googleapis.com`, `www.googleapis.com`,
  `sdk.cloud.google.com`.

### Authentication behavior
- CLI auth via **gcloud Application Default Credentials** (scopes: `openid`,
  `cloud-platform`, `userinfo.email`, `colaboratory`) or oauth2; token stored
  by the tool at `~/.config/colab-cli/token.json`. Standard Google OAuth; no
  session scraping; no excessive scopes beyond Colab's own API.
- On this host the CLI is **already authenticated** (`colab sessions`
  succeeds). No credentials were viewed, copied, or logged during review.

### Caveats that shaped our wrapper (`skills/compute/colab/`)
1. `--gpu` accepts only `T4|L4|G4|A100|H100` — **A100 40GB vs 80GB cannot be
   distinguished at request time** ⇒ runtime attestation against
   `configs/compute_policy.yaml` is mandatory before any workload runs.
2. **Unrecognized `--gpu` values silently fall back to A100** ⇒ our validators
   reject any value outside the exact accepted set before invoking the CLI.
3. Interactive subcommands (`colab auth`, `drivemount`, `repl`, `console`)
   hang non-TTY agents ⇒ forbidden in agent code paths (see the skill's
   `security.md`).
4. Unstopped sessions burn compute units up to a 24h cap ⇒ wrapper guarantees
   `stop_session()` in a `finally` block; prefer `colab run` semantics.

### Rejected behaviors checked and absent
Browser-session scraping, credential exfiltration, TLS-verification disabling,
obfuscated code, requests for non-Colab Google account permissions.

---

## Review procedure for future skills
1. Identify source repository and exact commit/release; pin it.
2. Inspect every executable script (not just documentation).
3. Grep for credential/cookie/TLS/obfuscation patterns; map network hosts.
4. Characterize the authentication flow; refuse skills that want raw tokens.
5. Record the review here before first use; re-review on version bumps.
