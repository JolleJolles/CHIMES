# Publishing CHIMES 0.3.0 on GitHub

The package version and dated changelog are prepared in the repository. The
release notes to paste into GitHub are in `docs/releases/0.3.0.md`. An empty
Unreleased section is reserved for subsequent work; the roadmap remains separate.

## Publish from GitHub

1. Open https://github.com/JolleJolles/CHIMES/releases/new.
2. Choose **v0.3.0**, selecting **Create new tag** if it does not exist, with
   **main** as the target. Confirm the version-preparation commit is on main and
   its Compatibility workflow has passed before publishing.
3. Use **CHIMES 0.3.0 — Schooling and interactive exploration** as the title.
4. Paste `docs/releases/0.3.0.md` into the description.
5. Optionally attach the wheel, source distribution and `SHA256SUMS.txt` from
   `dist/v0.3.0/` on the prepared computer. GitHub automatically supplies source
   ZIP and tar archives for the tag, so manually uploading those is unnecessary.
6. Leave **This is a pre-release** unchecked for a regular 0.3.0 release, select
   **Set as latest release**, then choose **Publish release**. Use **Save draft**
   instead if you want to review it first.

Do not create another commit between selecting main and publishing unless that
new commit is intended for 0.3.0. The resulting tag should identify the prepared,
verified release commit. Do not move a published tag to a later commit.

This creates a GitHub release, not a PyPI publication. The release-preparation
work does not create the tag or publish the release automatically.

## Rebuild release assets if needed

Use a packaging environment with current `build`, `setuptools` and `wheel`:

```sh
python -m pip install build 'setuptools>=77.0.3' wheel
python -m build --outdir dist/v0.3.0
```

The expected files are `chimesim-0.3.0-py3-none-any.whl` and
`chimesim-0.3.0.tar.gz`. The source distribution includes the tutorial,
documentation, schooling example and reference-test fixtures.

After making any source change, rebuild and revalidate assets before uploading.
Do not reuse stale artifacts from another version or commit.

GitHub's release-creation instructions:
https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
