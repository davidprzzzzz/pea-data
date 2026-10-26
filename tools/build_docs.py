#!/usr/bin/env python3
"""Turn data/latest.json (+ Claude's macro overlay) into ArtifactData batches.

usage: python tools/build_docs.py <repo_dir> <out_dir> <versions.json> [overlay.json]

versions.json : {"meta/state": 12, "universe/all": 4, "ideas/r01": 7, ...}  (missing = new doc)
overlay.json  : {"narrative": "...", "week_view": "...",
                 "ideas": {"<idea id>": {"verdict": "favorable|neutre|défavorable",
                                         "note": "...", "catalysts": "..."}}}
Writes <out_dir>/docs/*.json and <out_dir>/plan.json (list of batches, each <= 40 writes).
"""
import json
import os
import sys

SLOTS = 40


def main():
    repo, out, vfile = sys.argv[1], sys.argv[2], sys.argv[3]
    ov_path = sys.argv[4] if len(sys.argv) > 4 else None
    data = json.load(open(os.path.join(repo, "data", "latest.json")))
    versions = json.load(open(vfile)) if os.path.exists(vfile) else {}
    overlay = json.load(open(ov_path)) if ov_path and os.path.exists(ov_path) else {}
    os.makedirs(os.path.join(out, "docs"), exist_ok=True)
    writes = []

    def put(coll, doc_id, body):
        fp = os.path.join(out, "docs", f"{coll}__{doc_id}.json")
        with open(fp, "w") as f:
            json.dump(body, f, ensure_ascii=False, separators=(",", ":"))
        w = {"op": "set", "collection": coll, "doc_id": doc_id, "file_path": os.path.abspath(fp)}
        v = versions.get(f"{coll}/{doc_id}")
        if v:
            w["if_version"] = int(v)
        writes.append(w)

    ov_ideas = overlay.get("ideas", {})
    ids = []
    for i in range(SLOTS):
        slot = f"r{i + 1:02d}"
        if i < len(data["ideas"]):
            idea = dict(data["ideas"][i])
            idea["active"] = True
            if idea["id"] in ov_ideas:
                idea["overlay"] = ov_ideas[idea["id"]]
            ids.append(slot)
            put("ideas", slot, idea)
        elif f"ideas/{slot}" in versions:
            put("ideas", slot, {"active": False})

    state = dict(meta=data["meta"], macro=data["macro"], sectors=data["sectors"], track=data["track"],
                 slots=ids, narrative=overlay.get("narrative"), week_view=overlay.get("week_view"),
                 overlay_at=overlay.get("generated_at"))
    put("meta", "state", state)
    put("universe", "all", {"rows": data["universe"], "as_of": data["meta"]["as_of"]})

    plan = [writes[i:i + 40] for i in range(0, len(writes), 40)]
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    sizes = sorted(((os.path.getsize(w["file_path"]), w["doc_id"]) for w in writes), reverse=True)[:3]
    print(f"{len(writes)} writes in {len(plan)} batch(es); largest docs: {sizes}")


if __name__ == "__main__":
    main()
