---
name: split-commits
description: Divide repository changes into coherent, independently reviewable Git commits when the user asks to split, separate, or organize work into multiple commits. Do not use for a routine single commit, a request only to plan commit boundaries, or history rewriting.
---

# Split changes into commits

Inspect the working tree, staged changes, and recent commit messages before choosing boundaries. Group changes by purpose so each commit has a clear reason and leaves the repository in a sensible state. A file may contribute to multiple commits when its changes can be staged safely by hunk.

Preserve the user's work. Account for every changed and untracked file, including changes already staged. Do not reset, discard, amend, rebase, or force-push as part of this workflow. If an unrelated or sensitive change cannot be assigned confidently, leave it untouched and explain it.

For each group:

1. Stage only that group's changes. Use patch staging for mixed files and inspect the cached diff before committing. Do not use `git add -A` unless every remaining change belongs to that commit.
2. Run relevant checks when needed to establish that the group is independently valid. Do not introduce artificial commits solely to satisfy a preferred count.
3. Write a message matching the repository's established style and commit. Keep hooks and signing enabled.
4. Verify the resulting commit and remaining working tree before starting the next group.

At the end, report the commits in order and identify any changes left uncommitted. Do not push unless the user also requested it.
