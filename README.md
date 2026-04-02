# speak-to-user

This repository is the larger umbrella workspace for apps, packages, skills, docs, and vendor code that support local speech and user-facing accessibility tooling.

## Repository Layout

The intended top-level structure is:

- `apps/` for application entry points and host apps
- `packages/` for reusable package dependencies consumed by the larger workspace
- `skills/` for skill-related work
- `docs/` for repository documentation
- `vendor/` for vendored third-party or external code when needed

The tracked submodules currently live at:

- `packages/SpeakSwiftly`
- `apps/speak-to-user-server`
- `apps/speak-to-user-mcp`

Important: a fresh `git worktree` of this monorepo does not have populated submodule working trees until you initialize them. The first required step inside a new monorepo worktree is:

```bash
git submodule update --init --recursive
```

Until that command runs, submodule directories may look empty and `git submodule status` from that worktree may show uninitialized entries. That is incomplete checkout state, not evidence that the submodules were removed from repository history.

## SpeakSwiftly

`SpeakSwiftly` should live in this repository as a Git submodule at:

```text
packages/SpeakSwiftly
```

The source of truth for `SpeakSwiftly` remains its own standalone repository. The submodule in `speak-to-user` is the integration copy used by the larger workspace.

That means the expected workflow is:

- create a separate worktree from the clean local monorepo checkout on `main`
- run `git submodule update --init --recursive` inside that new worktree before inspecting or changing submodule state
- develop `SpeakSwiftly` in its standalone repository first
- push changes to the `SpeakSwiftly` remote
- cut or use a tagged `SpeakSwiftly` release when `speak-to-user` is ready to adopt a newer revision
- fetch tags in `packages/SpeakSwiftly`, check out the intended tagged release there, and verify the superproject diff only changes that one pointer
- land that `packages/SpeakSwiftly` submodule bump through a pull request against the monorepo
- use tagged releases for the monorepo itself when publishing umbrella milestones or coordinated workspace states

This keeps `SpeakSwiftly` independently versioned while still letting `speak-to-user` pin an exact commit.
