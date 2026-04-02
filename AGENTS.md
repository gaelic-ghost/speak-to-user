# speak-to-user

## Checkout discipline

- Treat the clean local monorepo checkout at `/Users/galew/Workspace/speak-to-user` as a protected base checkout.
- Keep that clean local monorepo checkout on `main` and keep it clean.
- Never switch that clean local monorepo checkout onto a feature branch.
- Never do feature work, submodule add work, submodule update work, or documentation edits directly in that clean local monorepo checkout.
- For any monorepo change, create a separate `git worktree` from the clean local `main` checkout first, then create the feature branch in that separate worktree.
- Immediately after creating a new monorepo worktree, run `git submodule update --init --recursive` inside that worktree before inspecting repo state, reading submodule paths, or updating any submodule pointer.
- Treat a fresh worktree with uninitialized submodules as incomplete checkout state, not as evidence that tracked submodules are missing from the repository.
- Open pull requests from those separate worktree branches against `main`.
- Do not use the clean local monorepo checkout as the branch-working directory unless Gale explicitly asks for that exact exception.

## Submodules

- The tracked monorepo submodules currently live at:
  - `packages/SpeakSwiftly`
  - `apps/speak-to-user-server`
  - `apps/speak-to-user-mcp`
- When checking monorepo submodule state, use `git submodule status` only after the worktree submodules have been initialized.
- Add or update one submodule concern per branch and per pull request unless Gale explicitly asks to bundle them.
- Keep submodule pointer updates narrowly scoped and verify the exact target tag or commit before opening a pull request.
- For a normal submodule bump:
  - create the worktree from the clean base checkout
  - run `git submodule update --init --recursive`
  - fetch the target submodule tags inside the specific submodule you are updating
  - check out the intended tagged release in that submodule
  - confirm the superproject diff shows only the intended pointer change
  - commit and push the superproject branch
