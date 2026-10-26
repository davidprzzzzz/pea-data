# Weekly sync procedure (run by the scheduled Claude task)

Artifact: see `ARTIFACT_URL` below (PEA RV Screener). Data is produced by `.github/workflows/screen.yml`
(Sundays 17:23 UTC) which runs `screener.py` with yfinance and commits `data/latest.json`.

ARTIFACT_URL = (filled in after first publish)

1. **Fresh data.** `git pull`. If `data/latest.json` `meta.as_of` is older than the last trading day
   (Friday), run `gh workflow run screen.yml -R davidprzzzzz/pea-data --ref main`, `gh run watch`, `git pull`.
   If the run fails, report the error from `gh run view --log-failed` and stop.
2. **Digest.** `python tools/brief.py .` — macro regime, sector tailwinds, every idea with its stats.
3. **Macro review (the qualitative layer).** Web-search the week's euro-area backdrop (ECB, inflation prints,
   PMIs, Bund yields/spreads, energy, geopolitics, earnings season) and, for each SIGNAL idea plus any WATCH
   idea scoring >= 60, check whether the macro backdrop supports the long leg *right now* and whether the
   statistical cheapness has a fundamental explanation (profit warning, M&A, index event, guidance cut,
   dividend detachment) that would make it a value trap. Confirm the next earnings date on the company's
   investor-relations calendar. Write `/tmp/overlay.json`:
   ```json
   {"generated_at": "<ISO>", "narrative": "<4-6 sentences, French, the week's macro backdrop + what it means for RV>",
    "week_view": "<one line: régime + where the screen finds edge>",
    "ideas": {"<idea id>": {"verdict": "favorable|neutre|défavorable",
                            "note": "<2-3 sentences French: why the trade does/doesn't make sense macro-wise, any news explaining the dislocation>",
                            "catalysts": "<confirmed dates on the horizon>"}}}
   ```
   Never invent dates. Flag any PEA-eligibility doubt (HQ, redomiciliation, primary listing) in the note.
4. **Versions.** `ArtifactData list` on `ideas` (limit 100, out_dir /tmp/v), `get` meta/state and
   universe/all if they exist -> write `/tmp/versions.json` mapping `"ideas/r01": version, "meta/state": v, "universe/all": v`.
5. **Build + write.** `python tools/build_docs.py . /tmp/sync-out /tmp/versions.json /tmp/overlay.json`,
   then send each batch of `/tmp/sync-out/plan.json` with `ArtifactData batch` (entries carry op/if_version/file_path).
   On a version conflict, re-list and rebuild once.
6. **Report** (French, short): as-of date, signal count, top 3-5 ideas with verdict and expected move,
   ideas rejected on macro/news grounds, data warnings. Research/education, not investment advice.
