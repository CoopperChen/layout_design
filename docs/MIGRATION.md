# Where the interconnect version lives

This branch does not plan wires. The previous pipeline — terminal assignment, synthesize, polish, smooth, and interconnect G-code — is `origin/master` at commit `93cff25`.

```powershell
git fetch origin
git checkout master
```

Do not merge this branch into `master` if you still need interconnects. Check out `electrode-only-layout-design` to return to electrode G-code.

Subject files on this branch are only `data/subjects/{id}/`. See [DATA_LAYOUT.md](DATA_LAYOUT.md).
