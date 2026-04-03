# speak-to-user

This repository is the larger umbrella workspace for apps, packages, skills, docs, and vendor code that support local speech and user-facing accessibility tooling.

## Repository Layout

The intended top-level structure is:

- `apps/` for application entry points and host apps
- `packages/` for reusable package dependencies consumed by the larger workspace
- `skills/` for skill-related work
- `docs/` for repository documentation
- `vendor/` for vendored third-party or external code when needed

## SpeakSwiftly

`SpeakSwiftly` should live in this repository as a Git submodule at:

```text
packages/SpeakSwiftly
```

The source of truth for `SpeakSwiftly` remains its own standalone repository. The submodule in `speak-to-user` is the integration copy used by the larger workspace.

That means the expected workflow is:

- develop `SpeakSwiftly` in its standalone repository first
- push changes to the `SpeakSwiftly` remote
- cut or use a tagged `SpeakSwiftly` release when `speak-to-user` is ready to adopt a newer revision
- update the `packages/SpeakSwiftly` submodule pointer here on a branch and land that bump through a pull request against the monorepo
- use tagged releases for the monorepo itself when publishing umbrella milestones or coordinated workspace states

This keeps `SpeakSwiftly` independently versioned while still letting `speak-to-user` pin an exact commit.

## SpeakSwiftlyMCP

`SpeakSwiftlyMCP` should live in this repository as a Git submodule at:

```text
packages/SpeakSwiftlyMCP
```

The source of truth for `SpeakSwiftlyMCP` remains its own standalone repository. The submodule in `speak-to-user` is the integration copy used by the larger workspace.

That means the expected workflow is:

- develop `SpeakSwiftlyMCP` in its standalone repository first
- push changes to the `SpeakSwiftlyMCP` remote
- cut or use a tagged `SpeakSwiftlyMCP` release when `speak-to-user` is ready to adopt a newer revision
- update the `packages/SpeakSwiftlyMCP` submodule pointer here on a branch and land that bump through a pull request against the monorepo
- use tagged releases for the monorepo itself when publishing umbrella milestones or coordinated workspace states

This keeps `SpeakSwiftlyMCP` independently versioned while still letting `speak-to-user` pin an exact commit.
