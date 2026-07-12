# VESSL agent setup

Only the current VESSL Cloud product (`https://cloud.vessl.ai`) and
`vesslctl` are supported. The legacy `vessl` CLI and legacy Organization /
Project model are forbidden.

## Current local state

- `vesslctl` was not found on PATH on 2026-07-12.
- The installer was not run. The official command awaiting explicit approval is:

  ```bash
  curl -fsSL https://api.cloud.vessl.ai/cli/install.sh | bash
  ```

- After installation, the user authenticates through the browser with
  `vesslctl auth login`. Never provide credentials to this repository.
- The bundled official skill therefore could not be installed or hashed.
  Once the CLI exists, run without `--force`:

  ```bash
  vesslctl skill install --target cross-client
  vesslctl skill install --target claude-code
  vesslctl skill show --output json
  ```

  Canonical targets are `~/.agents/skills` and `~/.claude/skills`.

## Documentation MCP

Claude Code was registered through verified client syntax:

```bash
claude mcp add --scope project --transport http \
  vessl-docs https://docs.cloud.vessl.ai/mcp
```

The server is recorded in `.mcp.json` and remains pending interactive client
approval. Local `codex mcp add --help` exposes HTTP registration but no verified
project scope. Do not modify an undocumented project config. If user-global
registration is acceptable, the documented manual command is:

```bash
codex mcp add vessl-docs --url https://docs.cloud.vessl.ai/mcp
```

This command was not run because it is not project-scoped.
