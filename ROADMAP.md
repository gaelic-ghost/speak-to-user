# Project Roadmap

## Vision

- Build a local-first shared speech service that coding agents and users can rely on for fast, expressive playback, reusable voice workflows, and approachable setup on Apple hardware.
- Keep the project understandable and hackable while expanding beyond one hard-coded TTS path.

## Product principles

- Keep the service local-first, inspectable, and easy to self-host.
- Prefer direct FastMCP features over extra wrappers when the protocol already provides what we need.
- Treat playback reliability and accessibility as first-class product requirements.
- Be honest about current latency and hardware limits instead of hiding them behind vague promises.
- Add new backends and UI surfaces carefully so the project stays comprehensible.

## Milestone Progress

- [x] Milestone 0: Shared local HTTP alpha
- [ ] Milestone 1: Clone quality and profile quick-start
- [ ] Milestone 2: Control surface and client ergonomics
- [ ] Milestone 3: App UI and progress feedback
- [ ] Milestone 4: Performance and backend expansion
- [ ] Milestone 5: Distribution and network reach
- [ ] Milestone 6: Remote-device experience

## Milestone 0: Shared local HTTP alpha

Scope:

- [x] Ship the shared HTTP FastMCP service architecture.
- [x] Support resident voice-design and clone models.
- [x] Support reference-audio cloning and reusable speech profiles.
- [x] Document local launchd deployment, Codex integration, and current limitations.

Tickets:

- [x] Replace stdio-only deployment with shared local HTTP MCP.
- [x] Add observability for queueing, synthesis, playback, and failure diagnosis.
- [x] Add `speak_text_as_clone`, speech profiles, and reusable profile playback.
- [x] Publish the project as an early showcase release.

Exit criteria:

- [x] Multiple Codex sessions can share one running local service.
- [x] The repo documents the current architecture and tool surface accurately.
- [x] The project is ready to tag as `v0.2.0-alpha`.

## Milestone 1: Clone quality and profile quick-start

Scope:

- [ ] Make cloning easier to start with and more useful for repeated use.
- [ ] Improve quality when moving between the lightweight clone model and the richer voice-design model.

Tickets:

- [ ] Add a workflow that uses the 0.6B clone model to bootstrap or condition the 1.7B voice-design model.
- [ ] Add built-in preset clone profiles for easy quick-start demos and out-of-the-box usage.
- [ ] Improve guidance and tooling for collecting higher-quality reference audio and transcripts.
- [ ] Add lightweight profile-management improvements where they reduce repeated setup friction.

Exit criteria:

- [ ] New users can hear a convincing clone path quickly without building a profile library from scratch.
- [ ] The 0.6B and 1.7B model workflows feel connected instead of isolated.

## Milestone 2: Control surface and client ergonomics

Scope:

- [ ] Make configuration safer and easier for users and agents to understand and adjust.
- [ ] Provide stronger guidance for reliable agent-side usage.

Tickets:

- [ ] Add a better user-facing and agent-facing way to inspect and adjust settings and configuration.
- [ ] Publish an agent skill for this server and include example `AGENTS.md` guidance for adopters.
- [ ] Clarify which knobs are operational, which are quality-related, and which should stay internal defaults.

Exit criteria:

- [ ] Common configuration changes no longer require source edits or guesswork.
- [ ] A new user can wire the server into their own agent setup with clear examples.

## Milestone 3: App UI and progress feedback

Scope:

- [ ] Expose a friendlier interactive surface on top of the MCP server.
- [ ] Give connected clients better live feedback while long-running TTS jobs are in flight.

Tickets:

- [ ] Add a FastMCP App or UI surface for common playback, clone, and profile workflows.
- [ ] Add FastMCP progress feedback so clients can reflect queueing, synthesis, and playback progress in real time.
- [ ] Improve task and event visibility where it helps users understand long first-audio latency.

Exit criteria:

- [ ] Users can discover and use the main workflows without memorizing raw tool arguments.
- [ ] Clients receive meaningful progress updates during long-running jobs.

## Milestone 4: Performance and backend expansion

Scope:

- [ ] Reduce time-to-first-audio and expand beyond the current Qwen-only path.
- [ ] Explore backend options that materially improve Apple Silicon performance.

Tickets:

- [ ] Explore integrating Swift or Metal FlashAttention 2 for a practical performance boost on supported Apple hardware.
- [ ] Evaluate support for Microsoft's realtime and cloning models alongside the current Qwen stack.
- [ ] Continue tuning chunking, buffering, and model-runtime settings based on measured first-audio latency.

Exit criteria:

- [ ] The project has at least one credible path toward lower first-audio latency.
- [ ] Alternate backend support is documented and bounded rather than hand-wavy.

## Milestone 5: Distribution and network reach

Scope:

- [ ] Make the project easier to run outside a terminal-only developer setup.
- [ ] Support access beyond localhost when explicitly desired.

Tickets:

- [ ] Distribute the service as a Python package with a self-contained `wavbuffer` payload, including a more formal wheel/package-data path for the bundled native binary.
- [ ] Package the service as a macOS menu bar app.
- [ ] Add a supported way to make the service LAN-accessible.
- [ ] Document the security and operational tradeoffs of exposing the service beyond localhost.

Exit criteria:

- [ ] Users can run the service comfortably without living in a terminal all day.
- [ ] LAN access is possible with explicit configuration and documented guardrails.

## Milestone 6: Remote-device experience

Scope:

- [ ] Extend the local speech service into a practical multi-device setup.

Tickets:

- [ ] Build an iOS remote app so a self-hosted Mac mini or similar machine can drive TTS on an iPhone.
- [ ] Define the control and playback model for remote devices without breaking the local-first architecture.

Exit criteria:

- [ ] A user can self-host the service on one Apple device and use it comfortably from another.
