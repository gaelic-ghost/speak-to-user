# speak-to-user

## Workspace role

- Treat this repository as the umbrella integration and planning layer for the broader speech-tooling workspace.
- Keep umbrella documentation here focused on pinned submodules, workspace structure, operating model, and cross-repository roadmap planning.
- Do not let umbrella docs imply that a planned app, package, extension, plugin, or service already lives in this repository when it is still only a sibling repo or a roadmap item.
- When a workspace component is still external to this monorepo, name that boundary explicitly in the docs.
- Treat `mcps/` as the umbrella home for vendored MCP service repositories.
- Treat `apps/` as the umbrella home for end-user apps, service hosts, and future interactive app surfaces such as MCP Apps.

## Checkout discipline

- Treat the clean local monorepo checkout at `/Users/galew/Workspace/speak-to-user` as a protected base checkout.
- Keep that clean local monorepo checkout on `main` and keep it clean.
- Never switch that clean local monorepo checkout onto a feature branch.
- Never do feature work, submodule add work, submodule update work, or documentation edits directly in that clean local monorepo checkout.
- For any monorepo change, create a separate `git worktree` from the clean local `main` checkout first, then create the feature branch in that separate worktree.
- Open pull requests from those separate worktree branches against `main`.
- Do not use the clean local monorepo checkout as the branch-working directory unless Gale explicitly asks for that exact exception.

## Documentation

- Keep `README.md`, `ROADMAP.md`, and `AGENTS.md` aligned when the umbrella workspace scope changes in a meaningful way.
- Keep implementation-specific setup, build, run, and test instructions in the relevant submodule or sibling repository instead of duplicating them in umbrella docs.
- When updating umbrella docs, preserve the distinction between:
  - current submodules already vendored here
  - sibling repositories that are part of the broader workspace but not vendored here yet
  - future roadmap items that do not exist as repositories or products yet
- Treat the umbrella `ROADMAP.md` as the home for cross-repository milestones such as distribution tracks, app adoption, plugin surfaces, extensions, and feedback systems.
- Now that `apps/SayBar` is vendored here, describe it as a current app submodule and keep deeper app implementation guidance in the standalone SayBar repository.
- When documenting the top-level workspace shape, explain why a component lives under `packages/`, `mcps/`, or `apps/` instead of leaving that split implicit.

## Submodules

- Add or update one submodule concern per branch and per pull request unless Gale explicitly asks to bundle them.
- Keep submodule pointer updates narrowly scoped and verify the exact target tag or commit before opening a pull request.
- Do not edit code or docs inside a submodule from an umbrella-docs branch unless Gale explicitly asks for that submodule work too.
- When a docs-only umbrella branch needs submodules present for context, initialize them in the worktree but leave their pointers unchanged unless the branch is specifically about a submodule bump.
- For `apps/SayBar`, prefer pinning the submodule to tagged standalone SayBar releases and keep the standalone SayBar repository as the source of truth for app development and release notes.
