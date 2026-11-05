# Where the interconnect version lives

This branch does not run the old wire planner. `interconnect` is the pre-search harness view: clicked terminals, balanced assignment, odd slot swaps, even bends, and a polar UV lift onto `head.stl`. Polish, smooth, the genetic search, and interconnect G-code stay on `origin/master` at commit `93cff25`.

```powershell
git fetch origin
git checkout master
```

Do not merge this branch into `master` if you still need interconnects. Check out `electrode-only-layout-design` to return to electrode G-code.

Subject files on this branch are only `data/subjects/{id}/`. See [DATA_LAYOUT.md](DATA_LAYOUT.md).
