These rules apply to the whole repository. Folders can add their own rules in their own `CLAUDE.md`
(e.g. `cuda_book/CLAUDE.md`).

## Stacked pull requests

- When a change splits into a sequence of dependent pieces, make one PR per logical piece and stack
  them. Each piece's branch is built on top of the previous piece's branch, and each PR's base is
  the branch below it, so every PR shows only its own piece. Each layer should build and pass its
  own tests.
- Before building a stack, tell the user the planned layers and how they will be stacked, and wait
  for their agreement.
- By default, stack with GitHub's native stacked PRs, using the `gh stack` CLI extension
  (`gh extension install github/gh-stack`). It needs Git 2.36 or later, and stacked PRs must be
  enabled for the repository.
  - `gh stack init <branch>` starts the stack with its first branch. Commit the first piece there.
  - `gh stack add <branch>` adds a branch on top of the stack. Commit the next piece there, and
    repeat for each piece.
  - `gh stack submit` pushes every branch and opens the linked PRs.
  - To change a lower piece (e.g. after review), switch to its branch with
    `gh stack checkout <branch>` or `gh stack down`, and commit there rather than in a higher layer.
    Then run `gh stack rebase --upstack` and `gh stack push`.
  - PRs merge bottom-up, with `gh stack merge` or from the merge box on GitHub. Afterwards,
    `gh stack sync --prune` rebases the remaining branches onto `main` and removes merged ones.
- If `gh stack` can't be used (the extension is missing, Git is older than 2.36, or it exits with
  code 9 because stacked PRs aren't enabled), tell the user and ask before falling back to stacking
  by hand: open the PRs with `gh pr create --base <branch below>`, then link them into a stack on
  GitHub.

> NOTE: if any of this is outdated notify the user.
