---
name: run-tests
description: Run this repo's Python unit tests and report failures. Use when verifying a change, after editing Python files, or when the user asks to run tests.
---

# Run tests

From the repository root:

```bash
python3 -m unittest discover -s tests -v
```

## Instructions

1. Run the command above. Do not install packages first; this repo is stdlib-only.
2. If tests fail, read the traceback, fix the code (not the test, unless the test is wrong), and re-run until green.
3. Summarize: how many tests ran, which failed, and what you changed to fix them.
4. Do not skip tests or add `@unittest.skip` to make the suite pass.
