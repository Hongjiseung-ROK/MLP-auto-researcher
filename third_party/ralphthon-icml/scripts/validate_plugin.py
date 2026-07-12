#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Final, NamedTuple


ROOT: Final = Path(__file__).resolve().parents[1]
PLUGIN_FIELDS: Final = (
    "name",
    "version",
    "description",
    "homepage",
    "repository",
    "license",
    "keywords",
    "skills",
    "interface",
)
MARKDOWN_LINK: Final = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
AGENT_INTERFACE_FIELDS: Final = (
    "display_name",
    "short_description",
    "default_prompt",
)
AGENT_INTERFACE_ENTRY: Final = re.compile(
    r"^  ([A-Za-z_][A-Za-z0-9_]*):(?:[ \t]*(.*))?$"
)


class ValidationError(NamedTuple):
    path: Path
    message: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def yaml_scalar_is_nonempty(raw_value: str) -> bool:
    value = raw_value.strip()
    if not value or value.startswith("#"):
        return False
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    return value not in {"", '\"\"', "''", "~"} and value.lower() != "null"


def frontmatter(text: str) -> list[str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return []
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return lines[1:index]
    return []


def top_level_values(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def discover_skill_paths(root: Path) -> list[Path]:
    return sorted((root / "skills").glob("*/SKILL.md"))


def add_error(
    errors: list[ValidationError], root: Path, path: Path, message: str
) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    errors.append(ValidationError(relative, message))


def validate_manifest(root: Path, errors: list[ValidationError]) -> None:
    path = root / ".codex-plugin" / "plugin.json"
    if not path.is_file():
        add_error(errors, root, path, "required file is missing")
        return
    try:
        manifest = json.loads(read_text(path))
    except json.JSONDecodeError as error:
        add_error(errors, root, path, f"invalid JSON: {error}")
        return

    for field in PLUGIN_FIELDS:
        if field not in manifest:
            add_error(errors, root, path, f"missing field: {field}")
    if manifest.get("name") != "ralphthon-icml":
        add_error(errors, root, path, "name must be ralphthon-icml")
    if manifest.get("skills") != "./skills/":
        add_error(errors, root, path, "skills must point to ./skills/")
    if not manifest.get("keywords"):
        add_error(errors, root, path, "keywords must not be empty")
    if "commands" in manifest:
        add_error(errors, root, path, "commands are unsupported; use skills only")


def validate_skill_files(
    root: Path, paths: list[Path], errors: list[ValidationError]
) -> set[str]:
    names: set[str] = set()
    for path in paths:
        lines = frontmatter(read_text(path))
        if not lines:
            add_error(errors, root, path, "frontmatter is missing")
            continue
        values = top_level_values(lines)
        name = values.get("name", "")
        description = values.get("description", "")
        if not name:
            add_error(errors, root, path, "missing frontmatter field: name")
        if not description:
            add_error(errors, root, path, "missing frontmatter field: description")
        if name and name != path.parent.name:
            add_error(
                errors,
                root,
                path,
                f"frontmatter name must match directory name: {path.parent.name}",
            )
        if name in names:
            add_error(errors, root, path, f"duplicate skill name: {name}")
        if name:
            names.add(name)
    return names


def validate_local_markdown_links(
    root: Path, errors: list[ValidationError]
) -> None:
    paths = [root / "README.md", *sorted((root / "skills").rglob("*.md"))]
    for path in paths:
        if not path.is_file():
            continue
        for match in MARKDOWN_LINK.finditer(read_text(path)):
            raw_target = match.group(1).strip().strip("<>")
            if (
                not raw_target
                or raw_target.startswith("#")
                or raw_target.startswith(("https://", "http://", "mailto:"))
            ):
                continue
            target = raw_target.split("#", maxsplit=1)[0]
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                add_error(
                    errors,
                    root,
                    path,
                    f"local Markdown link does not resolve: {raw_target}",
                )


def validate_agent_metadata(root: Path, errors: list[ValidationError]) -> None:
    for agents_dir in sorted((root / "skills").glob("*/agents")):
        path = agents_dir / "openai.yaml"
        if not path.is_file():
            add_error(errors, root, path, "agents/ requires openai.yaml")
            continue
        lines = read_text(path).splitlines()
        try:
            interface_index = next(
                index for index, line in enumerate(lines) if line == "interface:"
            )
        except StopIteration:
            add_error(errors, root, path, "missing exact top-level interface mapping")
            continue
        values: dict[str, str] = {}
        for line in lines[interface_index + 1 :]:
            if line and not line.startswith(" "):
                break
            match = AGENT_INTERFACE_ENTRY.fullmatch(line)
            if match is None:
                continue
            key, raw_value = match.groups(default="")
            values[key] = raw_value if yaml_scalar_is_nonempty(raw_value) else ""
        for field in AGENT_INTERFACE_FIELDS:
            if not values.get(field):
                add_error(
                    errors,
                    root,
                    path,
                    f"missing nonempty UI metadata field: {field}",
                )


def validate_repository(root: Path = ROOT) -> list[ValidationError]:
    root = root.resolve()
    errors: list[ValidationError] = []
    readme = root / "README.md"
    if not readme.is_file():
        add_error(errors, root, readme, "required file is missing")

    validate_manifest(root, errors)
    skill_paths = discover_skill_paths(root)
    if not skill_paths:
        add_error(errors, root, root / "skills", "no skills discovered")
    validate_skill_files(root, skill_paths, errors)
    validate_local_markdown_links(root, errors)
    validate_agent_metadata(root, errors)
    legacy_commands = root / "commands"
    if legacy_commands.exists():
        add_error(
            errors,
            root,
            legacy_commands,
            "commands/ is not supported; integrate behavior into skills",
        )
    return errors


def validate(root: Path = ROOT) -> list[ValidationError]:
    return validate_repository(root)


def main() -> int:
    errors = validate_repository(ROOT)
    if errors:
        print("Validation failed")
        for error in errors:
            print(f"- {error.path}: {error.message}")
        return 1

    skill_names = [path.parent.name for path in discover_skill_paths(ROOT)]
    print("Validation passed")
    print(f"- plugin: {ROOT.name}")
    print(f"- skills ({len(skill_names)}): {', '.join(skill_names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
