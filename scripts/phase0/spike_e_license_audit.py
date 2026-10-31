#!/usr/bin/env python
"""Spike E (automated part): licence audit of dependencies and vendored sources.

Collects the licence declared by each installed Python dependency (core, bench and llm extras),
flags copyleft or unknown entries for human review, and records the licence file of the vendored
NoRegrets checkout if present. The manual items (NoRegrets commercial terms, decision-model
weights) cannot be automated: they are listed with status TODO and tracked in the Phase 0 report.
"""

from __future__ import annotations

import hashlib
from importlib import metadata

from common import REPO, gate, write_result

PACKAGES = {
    "core": ["pokerkit", "numpy"],
    "bench": ["eval7", "phevaluator", "treys"],
    "llm": ["jevper", "openai"],
    "dev": ["pytest", "hypothesis", "ruff"],
}
COPYLEFT = ("GPL", "AGPL", "LGPL", "SSPL", "EUPL", "MPL")  # review, not necessarily a blocker

MANUAL_ITEMS = [
    (
        "noregrets-terms",
        "Request NoRegrets commercial terms in writing (docs/phase0/noregrets_licence_email.md)",
    ),
    (
        "noregrets-own-blueprint",
        "Confirm in writing that a blueprint you train from the MIT source may be shipped",
    ),
    (
        "model-weights",
        "Record the licence of every decision-model weight you intend to ship (kev, Ollaya, CLM, any GGUF)",
    ),
    ("kev-licence", "kev (jaredpalmer/kev): licence not verified"),
    ("ollaya-licence", "Ollaya (ollaya-dev/ollaya) and its laya models: licence not verified"),
    ("clm-licence", "CLM (Contrastive-LM/CLM) and its heads: licence not verified"),
]


def declared_license(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    text = meta.get("License-Expression") or ""
    if not text:
        lic = (meta.get("License") or "").strip().splitlines()
        text = lic[0] if lic and len(lic[0]) < 80 else ""
    classifiers = [
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    return text or "; ".join(classifiers) or "UNKNOWN"


def main() -> int:
    entries, review = [], []
    for group, names in PACKAGES.items():
        for name in names:
            try:
                dist = metadata.distribution(name)
            except metadata.PackageNotFoundError:
                entries.append({"package": name, "group": group, "installed": False})
                continue
            lic = declared_license(dist)
            # Dev tools are never distributed, so a copyleft dev dependency is noted but not flagged.
            flagged = group != "dev" and (
                lic == "UNKNOWN" or any(tag in lic.upper() for tag in COPYLEFT)
            )
            entries.append(
                {
                    "package": name,
                    "group": group,
                    "version": dist.version,
                    "license": lic,
                    "needs_review": flagged,
                    "installed": True,
                }
            )
            if flagged:
                review.append(f"{name} ({lic})")

    vendored = {}
    noregrets = REPO / "vendor" / "noregrets"
    for path in sorted(noregrets.glob("LICEN[SC]E*")) if noregrets.exists() else []:
        raw = path.read_bytes()
        vendored[path.name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "first_line": raw.decode(errors="replace").splitlines()[0] if raw else "",
            "looks_like_mit": b"MIT License" in raw or b"Permission is hereby granted" in raw,
        }

    for item in entries:
        if item["installed"]:
            flag = "  <-- REVIEW" if item["needs_review"] else ""
            if item["group"] == "dev":
                flag = "  (dev only, not shipped)"
            print(f"  {item['package']:<12} {item['version']:<10} {item['license']}{flag}")
        else:
            print(f"  {item['package']:<12} not installed")
    auto_ok = not review
    detail = (
        "no copyleft or unknown licences among installed dependencies"
        if auto_ok
        else "review needed: " + ", ".join(review)
    )
    path = write_result(
        "spike_e_license_audit",
        {
            "packages": entries,
            "vendored": vendored,
            "manual_items": [{"id": i, "task": t, "status": "TODO"} for i, t in MANUAL_ITEMS],
            "note": "Package metadata is self-declared: confirm against the actual LICENSE file before shipping",
            "gates": [
                gate("E-auto", auto_ok, detail),
                gate(
                    "E-manual", None, f"{len(MANUAL_ITEMS)} manual licence tasks open (see report)"
                ),
            ],
        },
    )
    print(f"E-auto {'PASS' if auto_ok else 'REVIEW'}: {detail}\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
