# speak-to-user

## Checkout discipline

- Treat the clean local monorepo checkout at `/Users/galew/Workspace/speak-to-user` as a protected base checkout.
- Keep that clean local monorepo checkout on `main` and keep it clean.
- Never switch that clean local monorepo checkout onto a feature branch.
- Never do feature work, submodule add work, submodule update work, or documentation edits directly in that clean local monorepo checkout.
- For any monorepo change, create a separate `git worktree` from the clean local `main` checkout first, then create the feature branch in that separate worktree.
- Open pull requests from those separate worktree branches against `main`.
- Do not use the clean local monorepo checkout as the branch-working directory unless Gale explicitly asks for that exact exception.

## Submodules

- Add or update one submodule concern per branch and per pull request unless Gale explicitly asks to bundle them.
- Keep submodule pointer updates narrowly scoped and verify the exact target tag or commit before opening a pull request.
