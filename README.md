# speak-to-user

Umbrella repository for Gale's local-first speech tooling workspace across reusable runtime packages, local HTTP and MCP hosts, apps, integrations, and future distribution surfaces.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Repository Layout](#repository-layout)
- [Development](#development)
- [Verification](#verification)
- [License](#license)

## Overview

This repository is the integration and planning home for the broader `speak-to-user` workspace. It keeps the current source-of-truth runtime pieces pinned together while leaving room for future app, extension, plugin, and distribution work that spans multiple repositories.

### Motivation

The goal is to keep the accessibility stack coherent without forcing every integration decision into one codebase too early. `SpeakSwiftly` can stay independently versioned as the speech runtime, both the Python and Swift-native host tracks can evolve in parallel, and the umbrella repository can track how those pieces are intended to fit together over time.

Today the workspace is intentionally small:

- `packages/SpeakSwiftly` is the pinned speech runtime package submodule.
- `packages/SpeakSwiftlyMCP` is the pinned Swift-native MCP host package submodule.
- `packages/SpeakSwiftlyServer` is the pinned Swift-native shared HTTP-and-MCP host package submodule.
- `apps/SayBar` is the pinned macOS menu bar app submodule.
- `mcps/speak-to-user-dev` is the pinned FastMCP coordinator submodule for serialized local dev chores across the workspace.
- `apps/speak-to-user-mcp` is the pinned Python MCP host submodule.
- `apps/speak-to-user-server` is the pinned Python HTTP host submodule.

The roadmap also tracks planned sibling or future workspace additions that are not yet vendored here, including editor integrations, browser surfaces, plugin distribution, and broader app distribution work.

## Setup

Initialize or refresh the current pinned workspace components:

```bash
git submodule update --init --recursive
```

That is the only required top-level setup command today. Each submodule keeps its own setup and verification instructions in its own repository documentation:

- [`SayBar`](https://github.com/gaelic-ghost/SayBar)
- [`speak-to-user-dev`](https://github.com/gaelic-ghost/speak-to-user-dev)
- [`SpeakSwiftly`](https://github.com/gaelic-ghost/SpeakSwiftly)
- [`SpeakSwiftlyMCP`](https://github.com/gaelic-ghost/SpeakSwiftlyMCP)
- [`SpeakSwiftlyServer`](https://github.com/gaelic-ghost/SpeakSwiftlyServer)
- [`speak-to-user-mcp`](https://github.com/gaelic-ghost/speak-to-user-mcp)
- [`speak-to-user-server`](https://github.com/gaelic-ghost/speak-to-user-server)

## Usage

This repository is not a single runnable app or package. Its current role is:

- pinning the active submodule revisions used together
- documenting the workspace shape and future integration intent
- landing submodule bumps and umbrella documentation changes through pull requests

If you want to run code, build the runtime, or start a local service, use the relevant submodule's README instead of treating the umbrella repository itself as the executable entrypoint.

## Repository Layout

The intended top-level structure is:

- `apps/` for end-user apps, service hosts, and future interactive app surfaces such as MCP Apps
- `mcps/` for vendored MCP service repositories that remain standalone source-of-truth service projects
- `packages/` for reusable runtime or library dependencies
- `skills/` for skill-related workspace assets
- `docs/` for umbrella documentation
- `vendor/` for vendored external code when needed

Current pinned workspace components:

- `packages/SpeakSwiftly`
  - Swift package submodule for the speech runtime and typed `SpeakSwiftlyCore` library surface
- `packages/SpeakSwiftlyMCP`
  - Swift package submodule for the Swift-native MCP host track
- `packages/SpeakSwiftlyServer`
  - Swift package submodule for the Swift-native shared HTTP-and-MCP host track
- `apps/SayBar`
  - macOS `MenuBarExtra` app submodule pinned to the standalone SayBar repository for app-level UI, settings, and service supervision work
- `mcps/speak-to-user-dev`
  - Python FastMCP coordinator submodule for serial local-dev workflows such as heavy e2e runs, docs sweeps, and submodule pin orchestration
- `apps/speak-to-user-mcp`
  - Python FastMCP host submodule for spoken replies and local MCP access
- `apps/speak-to-user-server`
  - Python FastAPI host submodule for app-friendly local HTTP access

Planned or sibling components that the umbrella roadmap also tracks:

- future OpenAI MCP App surfaces
  - interactive app UIs that may sit on top of the vendored MCP services while still living under `apps/`
- `SpeakSwiftlyMCP`
  - the Swift-native MCP-host package sibling to the current Python MCP server

## Development

Treat this repository as the integration and planning layer, not as the source of truth for every implementation detail.

- Keep the clean base checkout on `main` and do feature work in fresh worktrees.
- Update submodule pointers narrowly and intentionally.
- Keep vendored MCP services under `mcps/` and reserve `apps/` for interactive app surfaces and service hosts.
- Keep umbrella docs explicit about what is already vendored here versus what is still a sibling repository or planned future addition.
- Do not let umbrella documentation imply that a planned distribution surface already exists when it is only a roadmap item.
- Keep implementation-specific setup, build, and test instructions in the relevant submodule repositories.
- Prefer bumping `apps/SayBar` to tagged standalone SayBar releases instead of arbitrary branch tips.

The umbrella roadmap in [`ROADMAP.md`](https://github.com/gaelic-ghost/speak-to-user/blob/main/ROADMAP.md) tracks both current integration work and the larger slate of future distribution surfaces, including package distribution, apps, plugins, extensions, MCP Apps, and feedback tooling.

## Verification

The practical top-level verification loop is:

1. Confirm the umbrella checkout is clean and on the intended branch.
2. Run `git submodule update --init --recursive` and confirm the pinned components resolve successfully.
3. Review the current submodule pointers and the umbrella docs together before landing workspace-level changes.
4. Run the implementation-specific verification commands inside the affected submodule repositories when a change touches them.

This repository does not currently define a single top-level build, test, or lint command because the workspace is made up of independently verified submodules.

## License

Apache License 2.0. See [LICENSE](https://github.com/gaelic-ghost/speak-to-user/blob/main/LICENSE) and [NOTICE](https://github.com/gaelic-ghost/speak-to-user/blob/main/NOTICE).
