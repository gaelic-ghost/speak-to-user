# Project Roadmap

## Vision

- Build an umbrella workspace that keeps Gale's local speech stack coherent across runtime packages, service hosts, apps, extensions, plugins, and accessibility-focused distribution channels.

## Product principles

- Keep source-of-truth ownership explicit: umbrella docs and pins live here, while implementation details stay in the relevant submodule or sibling repository.
- Prefer narrow, well-named distribution surfaces over sprawling coordination layers or vague platform promises.
- Keep accessibility and operator clarity first-class across apps, plugins, MCP surfaces, and reporting flows.
- Treat the Python hosts and the Swift-native siblings as parallel product tracks until one path is intentionally retired.
- Keep roadmap items honest about what is already in the monorepo, what is a sibling repository today, and what is still only planned.

## Milestone Progress

- [x] Milestone 0: Umbrella docs and workspace operating model
- [ ] Milestone 1: `SpeakSwiftly` Swift package distribution
- [ ] Milestone 2: `SpeakSwiftlyMCP` Swift package distribution
- [ ] Milestone 3: `SpeakSwiftlyServer` Swift package distribution
- [ ] Milestone 4: Bring `SayBar` into the umbrella workspace
- [ ] Milestone 5: macOS distribution for `SayBar.app`
- [ ] Milestone 6: Python package distribution for `speak-to-user-mcp` and `speak-to-user-server`
- [ ] Milestone 7: `speak-to-user-dev` coordinator adoption and distribution
- [ ] Milestone 8: Agent-skills slate planning
- [ ] Milestone 9: Codex and Claude plugin distribution for skills and MCPs
- [ ] Milestone 10: Codex and Claude subagent configuration management
- [ ] Milestone 11: MCP Apps for the MCP services
- [ ] Milestone 12: XcodeKit extension and Xcode build plugins
- [ ] Milestone 13: Zed extension
- [ ] Milestone 14: Zen Browser extension and Firefox distribution
- [ ] Milestone 15: Discord user-install app
- [ ] Milestone 16: iOS and iPadOS system speech app
- [ ] Milestone 17: Centralized accessible feedback and bug reporting

## Milestone 0: Umbrella docs and workspace operating model

Scope:

- [x] Establish a real umbrella `README.md`, `ROADMAP.md`, and `AGENTS.md`.
- [x] Document the current submodules and the broader sibling-repo plan without pretending everything already lives here.
- [x] Keep the protected-base-checkout and fresh-worktree workflow explicit.

Tickets:

- [x] Document the current `apps/` and `packages/` submodules.
- [x] Document `mcps/` as the home for vendored MCP service repositories and keep `apps/` for interactive app surfaces and service hosts.
- [x] Clarify that top-level setup is just submodule initialization today.
- [x] Add a checklist-style umbrella roadmap for future distribution work.
- [x] Keep implementation-specific build and run steps delegated to the relevant submodule repositories.

Exit criteria:

- [x] The umbrella repo explains what it is for, what it currently contains, and how future workspace components relate to it.
- [x] Workspace planning no longer depends on chat history alone.

## Milestone 1: `SpeakSwiftly` Swift package distribution

Scope:

- [ ] Plan and document the supported SwiftPM distribution story for `packages/SpeakSwiftly`.
- [ ] Make semver and package-consumer expectations explicit at the umbrella level.

Tickets:

- [ ] Track the adoption path for the standalone `SpeakSwiftly` package beyond adjacent-local-checkout workflows.
- [ ] Document how umbrella consumers should decide between pinning tags and pinning commits for `packages/SpeakSwiftly`.
- [ ] Add an umbrella release-planning checklist item for package-consumer regression checks when `SpeakSwiftly` distribution changes.

Exit criteria:

- [ ] The workspace has an explicit supported package-distribution plan for `SpeakSwiftly`.

## Milestone 2: `SpeakSwiftlyMCP` Swift package distribution

Scope:

- [ ] Plan the distribution story for the Swift-native MCP-host package sibling to `apps/speak-to-user-mcp`.
- [ ] Keep the distinction clear between the current Python MCP server and the future Swift-native package.

Tickets:

- [ ] Define how `SpeakSwiftlyMCP` is expected to be versioned and distributed once it is adopted into the broader workspace story.
- [ ] Decide how the umbrella repo should reference the Swift-native MCP host while the Python MCP server remains the shipped default.
- [ ] Document the expected dependency boundary between `SpeakSwiftlyMCP` and `SpeakSwiftlyCore`.

Exit criteria:

- [ ] The workspace has a concrete package-distribution plan for the Swift-native MCP host track.

## Milestone 3: `SpeakSwiftlyServer` Swift package distribution

Scope:

- [ ] Plan the distribution story for the Swift-native localhost server sibling to `apps/speak-to-user-server`.
- [ ] Keep the distinction clear between the current Python HTTP server and the future Swift-native package.

Tickets:

- [ ] Define how `SpeakSwiftlyServer` is expected to be versioned and distributed once it is adopted into the broader workspace story.
- [ ] Decide how the umbrella repo should reference the Swift-native server while the Python server remains the current shipped host.
- [ ] Document the expected dependency boundary between `SpeakSwiftlyServer` and `SpeakSwiftlyCore`.

Exit criteria:

- [ ] The workspace has a concrete package-distribution plan for the Swift-native HTTP server track.

## Milestone 4: Bring `SayBar` into the umbrella workspace

Scope:

- [ ] Add the `SayBar` app to the umbrella-repo planning and integration model.
- [ ] Decide whether `SayBar` should become a submodule, a vendored app project, or stay a sibling repository with explicit coordination rules.

Tickets:

- [ ] Document the intended relationship between `SayBar` and the current `apps/` submodules.
- [ ] Decide where `SayBar` should live relative to the umbrella repo before broader app distribution work begins.
- [ ] Define the first integration checkpoints for speech-server supervision, profile visibility, and queue visibility.

Exit criteria:

- [ ] The workspace has an explicit plan for how `SayBar` joins the umbrella repository story.

## Milestone 5: macOS distribution for `SayBar.app`

Scope:

- [ ] Plan a real macOS distribution path for `SayBar.app`.
- [ ] Make signing, packaging, update, and installation expectations explicit before shipping the app.

Tickets:

- [ ] Decide the intended macOS distribution channel or channels for `SayBar.app`.
- [ ] Define the minimum release requirements for app signing, packaging, and update handling.
- [ ] Document how `SayBar.app` will locate, install, or supervise the speech-server components it depends on.

Exit criteria:

- [ ] The workspace has a concrete `SayBar.app` macOS distribution plan instead of an app-shaped placeholder.

## Milestone 6: Python package distribution for `speak-to-user-mcp` and `speak-to-user-server`

Scope:

- [ ] Plan Python package distribution for the two current Python service projects in `apps/`.
- [ ] Keep package-distribution planning separate from the longer-term Swift-native replacement tracks.

Tickets:

- [ ] Define whether `speak-to-user-mcp` should be distributed as an installable Python package beyond local `uv` workflows.
- [ ] Define whether `speak-to-user-server` should be distributed as an installable Python package beyond local `uv` workflows.
- [ ] Document versioning and release expectations for both Python projects while they remain first-class workspace components.

Exit criteria:

- [ ] The umbrella roadmap covers package-distribution plans for both Python services.

## Milestone 7: `speak-to-user-dev` coordinator adoption and distribution

Scope:

- [ ] Adopt the `speak-to-user-dev` coordinator as the vendored MCP-service home for serialized workspace development chores.
- [ ] Keep the distinction explicit between the raw coordinator service under `mcps/` and future interactive app surfaces that may sit above it under `apps/`.

Tickets:

- [ ] Keep `mcps/speak-to-user-dev` pinned as the umbrella submodule path for the standalone coordinator repository.
- [ ] Document which local-dev chores should eventually flow through the coordinator first, including heavy e2e runs, docs sweeps, and submodule pin orchestration.
- [ ] Define the release and pin-bump expectations for `mcps/speak-to-user-dev` inside the umbrella repo.

Exit criteria:

- [ ] The umbrella repo treats `mcps/speak-to-user-dev` as a first-class vendored MCP-service component with clear ownership boundaries.

## Milestone 8: Agent-skills slate planning

Scope:

- [ ] Plan the initial slate of agent skills needed across speech, accessibility, service control, app flows, and distribution work.
- [ ] Keep the skills plan grounded in real user or operator workflows instead of generic capability wishlists.

Tickets:

- [ ] Identify the first high-value skill families that belong under the workspace umbrella.
- [ ] Map which skills should live as standalone skills versus which should remain project-local guidance.
- [ ] Document the expected relationship between skills, MCP services, and app surfaces.

Exit criteria:

- [ ] The workspace has an explicit skills slate instead of scattered future ideas.

## Milestone 9: Codex and Claude plugin distribution for skills and MCPs

Scope:

- [ ] Plan distribution for Codex and Claude plugin surfaces that package the workspace skills and MCP capabilities.
- [ ] Keep plugin-distribution planning aligned with the actual MCP and skills boundaries.

Tickets:

- [ ] Define the first Codex plugin distribution targets.
- [ ] Define the first Claude plugin distribution targets.
- [ ] Document how plugin packaging should reuse the same underlying MCP and skills surfaces without duplicating product logic.

Exit criteria:

- [ ] The workspace has a documented plugin-distribution plan for Codex and Claude surfaces.

## Milestone 10: Codex and Claude subagent configuration management

Scope:

- [ ] Plan how Codex agent roles and Claude Code subagents should be configured around the workspace MCP and skills surfaces.
- [ ] Keep agent-config planning grounded in real operator workflows instead of creating redundant prompt layers.

Tickets:

- [ ] Define the first Codex agent-role distribution targets for workspace coordination work.
- [ ] Define the first Claude Code subagent targets for queue-aware workspace workflows.
- [ ] Document how those agent and subagent configs should reuse the same underlying skills and MCP tools without duplicating product logic.

Exit criteria:

- [ ] The workspace has an explicit agent and subagent configuration plan for Codex and Claude Code.

## Milestone 11: MCP Apps for the MCP services

Scope:

- [ ] Add a first-class roadmap track for MCP Apps on top of the MCP services.
- [ ] Keep the planning explicit about which MCP services should gain interactive app surfaces first.

Tickets:

- [ ] Define the first MCP App targets for the current and future MCP services.
- [ ] Document the relationship between tool APIs, resources, prompts, and interactive app views.
- [ ] Identify the minimum app surfaces needed for profiles, queue visibility, playback state, and operator feedback.

Exit criteria:

- [ ] The workspace has an explicit MCP Apps plan instead of treating UI surfaces as an afterthought.

## Milestone 12: XcodeKit extension and Xcode build plugins

Scope:

- [ ] Plan an XcodeKit Source Editor Extension and Xcode build-plugin story for the workspace.
- [ ] Include a `swift source mode` for the speech normalizer so Swift code can be spoken or transformed in a predictable source-aware way.

Tickets:

- [ ] Define the first Source Editor Extension workflows that are actually useful for Gale's speech and accessibility setup.
- [ ] Define the first `XcodeBuild` or `SwiftBuild` command-plugin workflows worth shipping.
- [ ] Plan the `swift source mode` normalization behavior and how it differs from the default prose-oriented speech normalization pipeline.

Exit criteria:

- [ ] The workspace has a concrete Xcode extension and build-plugin plan with a defined Swift-source normalization track.

## Milestone 13: Zed extension

Scope:

- [ ] Plan a Zed editor extension surface for the workspace.
- [ ] Keep the extension plan tied to real speech, status, and operator workflows.

Tickets:

- [ ] Identify the first Zed workflows worth supporting.
- [ ] Define whether the extension should talk to the MCP host, the HTTP server, or both.
- [ ] Document the minimum UI and command surface for the first release.

Exit criteria:

- [ ] The workspace has a documented Zed-extension plan.

## Milestone 14: Zen Browser extension and Firefox distribution

Scope:

- [ ] Plan a Zen Browser extension surface with a web-oriented normalization mode.
- [ ] Include Firefox-compatible distribution planning alongside the Zen-specific experience.

Tickets:

- [ ] Define the first browser workflows worth supporting for speech and accessibility.
- [ ] Plan a `web mode` for the normalizer that treats browser content differently from Swift or CLI text.
- [ ] Document the distribution expectations for Zen Browser and Firefox packaging.

Exit criteria:

- [ ] The workspace has a documented browser-extension and Firefox-distribution plan with a web-mode normalization track.

## Milestone 15: Discord user-install app

Scope:

- [ ] Plan a user-install Discord app for the workspace.
- [ ] Keep the scope grounded in the actual speech and agent workflows that belong in Discord.

Tickets:

- [ ] Define the first Discord user-install workflows worth supporting.
- [ ] Decide which backend surface should own Discord traffic.
- [ ] Document installation, authorization, and runtime expectations for the Discord app track.

Exit criteria:

- [ ] The workspace has a concrete Discord user-install app plan.

## Milestone 16: iOS and iPadOS system speech app

Scope:

- [ ] Plan an iOS and iPadOS app that integrates with Apple's newer custom speech-provider APIs for broader system use.
- [ ] Keep the plan explicit about where the app depends on Apple platform constraints versus shared workspace runtime behavior.

Tickets:

- [ ] Document the intended relationship between the mobile app and the existing speech runtime stack.
- [ ] Define the first system-speech workflows worth supporting on Apple mobile platforms.
- [ ] Track the Apple-platform integration requirements for custom speech-provider APIs, app architecture, and distribution.

Exit criteria:

- [ ] The workspace has a concrete mobile-system-speech plan instead of a vague future app idea.

## Milestone 17: Centralized accessible feedback and bug reporting

Scope:

- [ ] Build a centralized, accessible path for bug reports, operator feedback, and quality signals across the workspace.
- [ ] Keep the reporting surface usable from apps, browser surfaces, MCP tools, and editor integrations.

Tickets:

- [ ] Define the minimum shared feedback model used across workspace components.
- [ ] Identify the first reporting entrypoints that should exist across apps, services, and extensions.
- [ ] Document accessibility requirements for the reporting flow before implementation starts.

Exit criteria:

- [ ] The workspace has a concrete, accessibility-first feedback and bug-reporting plan that spans its major product surfaces.
