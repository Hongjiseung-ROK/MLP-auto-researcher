# Track 1 evidence map

| Output | Required binding | Current state |
|---|---|---|
| Research specification | exact plugin, code, data, split, checkpoint, and protocol hashes | template only |
| Agent workflow | prompts/actions, human gates, two-Job identities, receipts | schema implemented |
| Results | independently graded aggregate artifacts, including negative/inconclusive outcomes | no VESSL run |
| Short paper | copied official 2–4 page template plus frozen evidence hashes | not drafted |
| Self-review | same paper and evidence version | not run |

The machine-readable starting point is
`paper/track1/evidence-map.template.json`. An entry may become frozen only when
its referenced artifact exists and its SHA-256 verifies. Agent text, Tea Time,
review consensus, cost cards, and infrastructure metrics are not scientific
evidence. `claim_eligible` remains false until a separately authorized run and
H5 review establish otherwise.
