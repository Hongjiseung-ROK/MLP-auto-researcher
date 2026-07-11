# Example: materials_project_recon

`example_input.json` surveys the Cu chemical system, fetching only the four
listed summary fields (everything else is dropped before anything touches
disk). Requires repo-root `api.env` with a Materials Project key on first run;
identical repeat queries are served from `data/cache/materials_project/`
without network access. Reconnaissance only — outputs are not claim-eligible
dataset sources (see `../SKILL.md`).

Other operations:

```json
{"operation": "auth_check"}
{"operation": "get_structures", "material_ids": ["mp-30"]}
```
