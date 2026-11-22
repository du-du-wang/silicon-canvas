"""Review loop for generated images: objective checks, cold read, decision.

AI image models rarely get a technical figure right in one pass, so every
attempt goes through the same loop:

1. ``inspect_image`` measures the bitmap (size, aspect, how much of the canvas
   is used, color discipline, near-duplicate of an earlier attempt).
2. A cold read: a reader who has not seen the request transcribes the image
   (texts, arrows, groups). ``cold_read_task`` returns those instructions and
   deliberately contains no source facts.
3. ``compare`` diffs the cold read against the figure content that the
   generation prompt asked for: missing, misspelled or invented text; missing,
   reversed or extra arrows; wrong timing values.
4. ``decide`` combines technical issues, the aesthetic rubric and attempt
   history into accept / edit / regenerate / stop, with a short next prompt.

The server never sees pixels semantically; the host model does. What the
server guarantees is that the comparison is mechanical and uses the same
inventory as the prompt, so a reviewer cannot wave errors through.
"""
from __future__ import annotations

import difflib
import io
import re
import unicodedata
from typing import Any

from PIL import Image

VERSION = "LOOP1"
MAX_IMAGE_BYTES = 40 * 1024 * 1024
IMAGE_FORMATS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}
DEFAULT_MAX_ATTEMPTS = 4
MAX_EDIT_ITEMS = 8

AESTHETIC_DIMENSIONS = {
    "composition": "Canvas use and balance: the figure fills the frame, no large empty bands, nothing cramped or cut off.",
    "hierarchy": "Reading order is obvious; the main path and groups stand out, secondary detail recedes.",
    "color": "Few colors, each with one consistent meaning; no rainbow blocks, glow or decorative gradients.",
    "typography": "One typeface, consistent sizes and weights, legible at document size, math typeset properly.",
    "lines": "Thin consistent strokes, clean arrowheads that land on borders, no arrows crossing text or each other needlessly.",
    "finish": "No rendering defects: garbled or melted text, broken lines, stray marks, blur or compression artifacts.",
}


# --- objective bitmap checks -------------------------------------------------

def load_image(data: bytes) -> tuple[Image.Image, str]:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds 40 MB.")
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise ValueError("File is not a readable PNG, JPEG or WebP image.") from None
    if image.format not in IMAGE_FORMATS:
        raise ValueError("Only PNG, JPEG or WebP images can be reviewed.")
    return image, IMAGE_FORMATS[image.format]


def dhash(image: Image.Image) -> str:
    small = image.convert("L").resize((9, 8), Image.LANCZOS)
    pixels = list(small.getdata())
    bits = "".join("1" if pixels[row * 9 + col] > pixels[row * 9 + col + 1] else "0" for row in range(8) for col in range(8))
    return "%016x" % int(bits, 2)


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def inspect_image(image: Image.Image, *, expected_aspect: float | None = None, target_width_mm: float | None = None,
                  flat_figure: bool = True, previous_hashes: list[str] | None = None) -> dict[str, Any]:
    width, height = image.size
    rgb = image.convert("RGB")
    scale = 400 / max(width, height)
    small = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    sw, sh = small.size
    pixels = small.load()
    border = [pixels[x, y] for x in range(sw) for y in (0, 1, sh - 2, sh - 1)] + [pixels[x, y] for y in range(sh) for x in (0, 1, sw - 2, sw - 1)]
    background = tuple(sorted(channel)[len(channel) // 2] for channel in zip(*border))

    def ink(p):
        return max(abs(p[i] - background[i]) for i in range(3)) > 28

    rows = [sum(ink(pixels[x, y]) for x in range(sw)) for y in range(sh)]
    cols = [sum(ink(pixels[x, y]) for y in range(sh)) for x in range(sw)]
    used_rows = [i for i, v in enumerate(rows) if v > max(1, sw * 0.004)]
    used_cols = [i for i, v in enumerate(cols) if v > max(1, sh * 0.004)]
    metrics: dict[str, Any] = {"width": width, "height": height, "aspect": round(width / height, 3), "background": "#%02X%02X%02X" % background}
    issues: list[dict[str, str]] = []
    if not used_rows or not used_cols:
        metrics["occupancy"] = 0.0
        issues.append({"dimension": "composition", "severity": "major", "issue": "The image is blank or almost blank."})
    else:
        top, bottom, left, right = used_rows[0], used_rows[-1], used_cols[0], used_cols[-1]
        occupancy = ((bottom - top + 1) * (right - left + 1)) / (sw * sh)
        margins = {"top": top / sh, "bottom": (sh - 1 - bottom) / sh, "left": left / sw, "right": (sw - 1 - right) / sw}
        cells = 0
        for gy in range(4):
            for gx in range(4):
                box = [(x, y) for x in range(gx * sw // 4, (gx + 1) * sw // 4, 2) for y in range(gy * sh // 4, (gy + 1) * sh // 4, 2)]
                if box and sum(ink(pixels[x, y]) for x, y in box) / len(box) > 0.01:
                    cells += 1
        metrics.update(occupancy=round(occupancy, 3), margins={k: round(v, 3) for k, v in margins.items()}, filled_cells_of_16=cells)
        if flat_figure:
            empty = [side for side, value in margins.items() if value > 0.18]
            if occupancy < 0.5 or empty:
                issues.append({"dimension": "composition", "severity": "minor",
                               "issue": "The figure uses about %d%% of the canvas%s; enlarge it to fill the frame." % (
                                   round(occupancy * 100), (" with large empty " + "/".join(empty) + " bands") if empty else "")})
            if abs(margins["left"] - margins["right"]) > 0.12 or abs(margins["top"] - margins["bottom"]) > 0.15:
                issues.append({"dimension": "composition", "severity": "minor", "issue": "The figure sits off-center; balance the margins."})
    if flat_figure:
        hsv = small.convert("HSV")
        hues = [h * 12 // 256 for h, s, v in hsv.getdata() if s > 70 and v > 70]
        saturated = len(hues) / (sw * sh)
        families = sorted({h for h in set(hues) if hues.count(h) > max(20, len(hues) * 0.04)})
        metrics.update(saturated_fraction=round(saturated, 3), accent_hue_families=len(families))
        if len(families) >= 5 and saturated > 0.04:
            issues.append({"dimension": "color", "severity": "minor", "issue": "Uses %d distinct saturated hue families; reduce to the role palette." % len(families)})
    if expected_aspect:
        metrics["expected_aspect"] = round(expected_aspect, 3)
        if abs(width / height - expected_aspect) / expected_aspect > 0.2:
            issues.append({"dimension": "composition", "severity": "minor", "issue": "Canvas aspect %.2f differs from the planned %.2f." % (width / height, expected_aspect)})
    if target_width_mm:
        dpi = width / (target_width_mm / 25.4)
        metrics["dpi_at_document_width"] = round(dpi)
        if dpi < 240:
            metrics["print_note"] = "About %d dpi at %g mm width; upscale or regenerate larger before print use." % (dpi, target_width_mm)
    metrics["dhash"] = dhash(image)
    for index, previous in enumerate(previous_hashes or [], start=1):
        if previous and hamming(previous, metrics["dhash"]) <= 3:
            metrics["near_duplicate_of_attempt"] = index
            issues.append({"dimension": "finish", "severity": "minor", "issue": "Nearly identical to attempt %d; the last change did not take effect." % index})
    return {"metrics": metrics, "issues": issues}


# --- cold read ----------------------------------------------------------------

COLD_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One sentence: what this figure shows and what a reader would conclude."},
        "texts": {"type": "array", "items": {"type": "string"}, "description": "Every readable text exactly as drawn, misspellings included, one entry per text element."},
        "connections": {"type": "array", "items": {"type": "object", "properties": {
            "from": {"type": "string"}, "to": {"type": "string"}, "label": {"type": "string"},
            "bidirectional": {"type": "boolean"}, "dashed": {"type": "boolean"}}, "required": ["from", "to"]},
            "description": "Every arrow or connector, named by the text of the block where it starts and ends."},
        "groups": {"type": "array", "items": {"type": "object", "properties": {"label": {"type": "string"}, "members": {"type": "array", "items": {"type": "string"}}}}},
        "timing": {"type": "object", "description": "Only for timing tables: {\"columns\": [...], \"rows\": {\"signal\": [values...]}}"},
        "unclear": {"type": "array", "items": {"type": "string"}, "description": "Anything ambiguous: unreadable text, arrow ends that are hard to trace."},
        "defects": {"type": "array", "items": {"type": "string"}, "description": "Only real defects: garbled or melted text, broken lines, stray marks. Empty if none."},
    },
    "required": ["summary", "texts", "connections"],
}


def cold_read_task(family: str) -> dict[str, Any]:
    timing = family.endswith("sampled-timing")
    return {
        "blind": True,
        "instructions": "\n".join([
            "Cold read. Describe only what is visible in this image. Do not use the original request, the specification or what the image is supposed to show.",
            "Best: give only the image and these instructions to a fresh model context (a subagent or a separate call) that has not seen the request. Otherwise, write the cold read before looking at the request again.",
            "1. summary: one sentence on what the figure shows and what a reader would conclude.",
            "2. texts: transcribe every readable text exactly as drawn, including misspellings, one entry per label.",
            "3. connections: every arrow or connector, with the text of the block where it starts ('from') and ends ('to'), its label, and whether it is two-headed or dashed. Name a block inside a container by the block, not the container.",
            "4. groups: every frame or container and the labels inside it.",
            *(["5. timing: the table as {columns: [...], rows: {signal: [values]}}."] if timing else []),
            "6. unclear: anything you cannot read or trace with confidence. 7. defects: only real defects (garbled text, broken lines, stray marks); leave it empty if there are none.",
            "Report what you see, not what is likely intended. If an arrow end is ambiguous, list it under unclear instead of guessing.",
        ]),
        "schema": COLD_READ_SCHEMA,
        "aesthetic_rubric": {"scale": "1 (poor) to 5 (excellent); give one short note for any score below 4.", "dimensions": AESTHETIC_DIMENSIONS},
    }


# --- comparison ----------------------------------------------------------------

_SUPERSCRIPT_T = str.maketrans({"ᵀ": "t", "⊤": "t", "ᵗ": "t", "×": "x", "−": "-", "–": "-", "—": "-", "√": "sqrt"})


def normalize(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).translate(_SUPERSCRIPT_T).lower()
    return re.sub(r"[\s\^_{}\\$*·,.:;'\"`()\[\]\-]+", "", value)


def _pieces(text: str) -> list[str]:
    return [normalize(p) for p in re.split(r"[\s,;:|/]+|(?<=\))|(?=\()", str(text)) if normalize(p)]


def _similar(a: str, b: str) -> bool:
    """Close enough to be a misspelling: high similarity or a transposition."""
    if len(a) < 4 or len(b) < 4:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8 or (sorted(a) == sorted(b) and a != b)


class _TextIndex:
    def __init__(self, texts: list[str], planned: list[str] | None = None):
        self.raw = [str(t) for t in texts if str(t).strip()]
        self.full = [normalize(t) for t in self.raw]
        self.pieces = {p for t in self.raw for p in _pieces(t)} | set(self.full)
        self.planned = {normalize(p) for p in (planned or [])}

    def find(self, label: str) -> tuple[str, str | None]:
        """Return ('found'|'misspelled'|'missing', closest raw text)."""
        key = normalize(label)
        if not key:
            return "found", None
        if key in self.full:
            return "found", None
        # A label only counts as present inside a longer text when that text is
        # not itself another planned string (SiLU inside "H = SiLU(G) ⊙ U").
        hosts = [full for full in self.full if full not in self.planned or full == key]
        if key in {p for raw, full in zip(self.raw, self.full) if full in hosts for p in _pieces(raw)}:
            return "found", None
        if len(key) >= 4 and any(key in full for full in hosts):
            return "found", None
        best, best_ratio = None, 0.0
        for raw, full in zip(self.raw, self.full):
            if full in self.planned and full != key:
                continue
            ratio = difflib.SequenceMatcher(None, key, full).ratio() + (0.2 if sorted(key) == sorted(full) else 0)
            if ratio > best_ratio:
                best, best_ratio = raw, ratio
        if best is not None and _similar(key, normalize(best)):
            return "misspelled", best
        return "missing", None


def _resolver(content: dict[str, Any]):
    """Map a visible name to the set of element ids it can stand for."""
    entries = []
    for element in content["elements"]:
        entries.append((normalize(element["label"]), {element["id"]}))
        for child in element.get("children", []):
            entries.append((normalize(child), {element["id"]}))

    def resolve(name: str) -> set[str]:
        key = normalize(name)
        if not key:
            return set()
        exact = set().union(*[ids for label, ids in entries if label == key] or [set()])
        if exact:
            return exact
        contained = set().union(*[ids for label, ids in entries if len(label) >= 3 and (label in key or key in label)] or [set()])
        if contained:
            return contained
        close = [ids for label, ids in entries if _similar(key, label)]
        return set().union(*close) if close else set()
    return resolve


def compare(content: dict[str, Any], allowed: list[str], cold_read: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cold_read, dict) or not isinstance(cold_read.get("texts"), list) or not isinstance(cold_read.get("connections"), list):
        raise ValueError("cold_read must contain 'texts' and 'connections' lists (see the cold-read schema).")
    texts = [str(t) for t in cold_read["texts"]][:2000]
    for group in cold_read.get("groups") or []:
        if isinstance(group, dict) and group.get("label"):
            texts.append(str(group["label"]))
    index = _TextIndex(texts, allowed)
    issues: list[dict[str, Any]] = []

    def text_check(label, kind, severity, where=""):
        state, closest = index.find(label)
        if state == "misspelled":
            issues.append({"type": "misspelled_text", "severity": severity, "expected": label, "seen": closest, "where": where})
        elif state == "missing":
            issues.append({"type": "missing_" + kind, "severity": severity, "expected": label, "where": where})

    for element in content["elements"]:
        if content.get("timing"):
            break
        text_check(element["label"], "element", "major")
        for child in element.get("children", []):
            text_check(child, "element", "major", where=element["label"])
        for line in element.get("lines", []):
            text_check(line, "annotation", "minor", where=element["label"])
    for group in content["groups"]:
        text_check(group["label"], "group_label", "minor")

    # Unexpected text: transcribed strings that match nothing the prompt allowed.
    allowed_index = _TextIndex(allowed)
    allowed_full = [normalize(a) for a in allowed]
    for raw in texts:
        key = normalize(raw)
        if len(key) < 2 or key.isdigit():
            continue
        if key in allowed_index.pieces or any(len(key) >= 3 and (key in a or a in key) for a in allowed_full if len(a) >= 2):
            continue
        if any(_similar(key, a) for a in allowed_full):
            continue  # Reported as a misspelling of the planned text instead.
        if all(len(p) <= 1 or p in allowed_index.pieces for p in _pieces(raw)):
            continue
        issues.append({"type": "unexpected_text", "severity": "minor", "seen": raw})

    # Connections.
    resolve = _resolver(content)
    expected = [dict(c, matched=0, reversed=0) for c in content["connections"]]
    for obs in cold_read["connections"][:1000]:
        if not isinstance(obs, dict):
            continue
        sources, targets = resolve(obs.get("from", "")), resolve(obs.get("to", ""))
        if not sources or not targets:
            issues.append({"type": "unresolved_connection", "severity": "minor", "seen": "%s -> %s" % (obs.get("from", "?"), obs.get("to", "?")),
                           "note": "An arrow end does not match any planned block."})
            continue
        hit = False
        directions = [(sources, targets)] + ([(targets, sources)] if obs.get("bidirectional") else [])
        for src, dst in directions:
            candidates = [c for c in expected if c["source_id"] in src and c["target_id"] in dst]
            if candidates:
                chosen = min(candidates, key=lambda c: c["matched"])
                chosen["matched"] += 1
                hit = True
                if chosen.get("label") and obs.get("label") and index.find(chosen["label"])[0] == "missing" \
                        and difflib.SequenceMatcher(None, normalize(chosen["label"]), normalize(obs["label"])).ratio() < 0.8:
                    issues.append({"type": "wrong_connection_label", "severity": "minor", "connection": "%s -> %s" % (chosen["source"], chosen["target"]),
                                   "expected": chosen["label"], "seen": obs["label"]})
        if obs.get("bidirectional") and hit and not any(c.get("bidirectional") for c in expected
                                                        if c["source_id"] in sources and c["target_id"] in targets):
            issues.append({"type": "extra_arrowhead", "severity": "minor", "seen": "%s <-> %s" % (obs.get("from"), obs.get("to"))})
        if hit:
            continue
        reverse = [c for c in expected if c["source_id"] in targets and c["target_id"] in sources and not c.get("bidirectional")]
        if reverse:
            reverse[0]["reversed"] += 1
            continue
        issues.append({"type": "extra_connection", "severity": "major", "seen": "%s -> %s" % (obs.get("from"), obs.get("to"))})
    for c in expected:
        if c["matched"]:
            if c["matched"] > 1:
                issues.append({"type": "duplicate_connection", "severity": "minor", "connection": "%s -> %s" % (c["source"], c["target"])})
            continue
        if c["reversed"]:
            issues.append({"type": "reversed_connection", "severity": "major", "connection": "%s -> %s" % (c["source"], c["target"]), "label": c.get("label", "")})
        else:
            issues.append({"type": "missing_connection", "severity": "major", "connection": "%s -> %s" % (c["source"], c["target"]), "label": c.get("label", "")})

    # Timing tables.
    if content.get("timing") and isinstance(cold_read.get("timing"), dict):
        seen_rows = {normalize(k): v for k, v in (cold_read["timing"].get("rows") or {}).items()}
        for row in content["timing"]["rows"]:
            if not row["values"]:
                continue
            observed = seen_rows.get(normalize(row["label"]))
            if observed is None:
                issues.append({"type": "missing_timing_row", "severity": "major", "expected": row["label"]})
            elif [normalize(v) for v in observed] != [normalize(v) for v in row["values"]]:
                issues.append({"type": "wrong_timing_values", "severity": "major", "row": row["label"],
                               "expected": " ".join(row["values"]), "seen": " ".join(str(v) for v in observed)})
    elif content.get("timing"):
        issues.append({"type": "timing_not_read", "severity": "minor", "note": "The cold read did not transcribe the timing table."})
    for defect in (cold_read.get("defects") or [])[:20]:
        issues.append({"type": "defect", "severity": "minor", "seen": str(defect)})
    expected_count = len(content["connections"])
    return {"issues": issues,
            "counts": {"major": sum(i["severity"] == "major" for i in issues), "minor": sum(i["severity"] == "minor" for i in issues),
                       "planned_elements": len(content["elements"]), "planned_connections": expected_count,
                       "matched_connections": sum(1 for c in expected if c["matched"])},
            "cold_summary": str(cold_read.get("summary", ""))[:600],
            "unclear": [str(u) for u in (cold_read.get("unclear") or [])[:20]]}


# --- aesthetics, scoring and decision ---------------------------------------

def aesthetic_check(scores: Any) -> dict[str, Any]:
    if not isinstance(scores, dict):
        raise ValueError("aesthetic must map rubric dimensions to {score, note}.")
    result, issues = {}, []
    for dimension in AESTHETIC_DIMENSIONS:
        entry = scores.get(dimension)
        score = entry.get("score") if isinstance(entry, dict) else entry
        if not isinstance(score, (int, float)) or not 1 <= score <= 5:
            raise ValueError(f"aesthetic.{dimension}.score must be a number from 1 to 5.")
        note = str(entry.get("note", "")) if isinstance(entry, dict) else ""
        result[dimension] = {"score": score, "note": note[:300]}
        if score < 4:
            issues.append({"type": "aesthetic", "dimension": dimension, "severity": "major" if score <= 2 else "minor",
                           "score": score, "note": note[:300] or AESTHETIC_DIMENSIONS[dimension]})
    mean = sum(v["score"] for v in result.values()) / len(result)
    low = min(v["score"] for v in result.values())
    # passed: acceptable for delivery. polished: nothing left worth an edit.
    return {"scores": result, "mean": round(mean, 2), "passed": mean >= 3.8 and low >= 3, "polished": low >= 4, "issues": issues}


def attempt_score(comparison: dict[str, Any], aesthetic: dict[str, Any], objective: list[dict[str, Any]],
                  host_findings: list[dict[str, Any]] | None = None) -> float:
    technical = comparison["counts"]["major"] * 10 + comparison["counts"]["minor"] * 3
    technical += sum(10 if finding.get("technical") else 3 for finding in host_findings or []
                     if isinstance(finding, dict) and finding.get("issue"))
    return round(100 - technical - sum(max(0, 4 - v["score"]) * 5 for v in aesthetic["scores"].values()) - 2 * len(objective), 1)


def _edit_item(issue: dict[str, Any]) -> str | None:
    kind = issue["type"]
    if kind == "misspelled_text":
        return f'Change the text "{issue["seen"]}" to exactly "{issue["expected"]}".'
    if kind in {"missing_element", "missing_annotation", "missing_group_label"}:
        where = f' inside "{issue["where"]}"' if issue.get("where") and kind == "missing_element" else (f' next to "{issue["where"]}"' if issue.get("where") else "")
        noun = {"missing_element": "a block labeled", "missing_annotation": "the text", "missing_group_label": "the group heading"}[kind]
        return f'Add {noun} "{issue["expected"]}"{where}.'
    if kind == "missing_connection":
        source, target = issue["connection"].split(" -> ", 1)
        return f'Add one arrow from "{source}" to "{target}"' + (f' labeled "{issue["label"]}".' if issue.get("label") else ".")
    if kind == "reversed_connection":
        source, target = issue["connection"].split(" -> ", 1)
        return f'Reverse the arrow between "{source}" and "{target}" so it points to "{target}".'
    if kind == "extra_connection":
        source, target = issue["seen"].split(" -> ", 1)
        return f'Remove the arrow from "{source}" to "{target}".'
    if kind == "extra_arrowhead":
        return f'Make the connector {issue["seen"]} one-way: keep only the arrowhead at the planned target.'
    if kind == "duplicate_connection":
        return f'Keep only one arrow for {issue["connection"]}.'
    if kind == "wrong_connection_label":
        return f'Change the label on the arrow {issue["connection"]} from "{issue["seen"]}" to "{issue["expected"]}".'
    if kind == "unexpected_text":
        return f'Remove the text "{issue["seen"]}".'
    if kind == "wrong_timing_values":
        return f'Set row "{issue["row"]}" to: {issue["expected"]}.'
    if kind == "defect":
        return f'Fix this rendering defect: {issue["seen"].rstrip(".")}.'
    if kind == "aesthetic":
        return f'Improve {issue["dimension"]}: {issue["note"].rstrip(".")}.'
    if kind == "objective":
        return issue["issue"]
    if kind == "host_finding":
        return issue["issue"]
    return None


def decide(comparison: dict[str, Any], aesthetic: dict[str, Any], objective: list[dict[str, Any]], history: list[dict[str, Any]],
           *, generation_prompt: str, max_attempts: int = DEFAULT_MAX_ATTEMPTS, host_findings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    issues = list(comparison["issues"])
    issues += [{"type": "host_finding", "severity": "major" if f.get("technical") else "minor", "issue": str(f.get("issue", ""))[:400]}
               for f in (host_findings or []) if isinstance(f, dict) and f.get("issue")]
    objective_issues = [{"type": "objective", "severity": item["severity"], "issue": item["issue"], "dimension": item["dimension"]} for item in objective]
    all_issues = issues + aesthetic["issues"] + objective_issues
    major = [i for i in issues if i["severity"] == "major"]
    technical_minor = [i for i in issues if i["severity"] == "minor" and i["type"] not in {"defect"}]
    attempt_number = len(history) + 1
    counts = comparison["counts"]
    edge_errors = sum(i["type"] in {"missing_connection", "reversed_connection", "extra_connection"} for i in issues)
    missing_elements = sum(i["type"] == "missing_element" for i in issues)
    recent_edits = 0
    for previous in reversed(history):
        if previous.get("decision", {}).get("action") != "edit":
            break
        recent_edits += 1
    occupancy_bad = any(i["dimension"] == "composition" and "canvas" in i["issue"] for i in objective)
    blank = any("blank" in i["issue"] for i in objective)
    clean = not major and not technical_minor and not blank
    polish_rounds = sum(1 for previous in history if previous.get("decision", {}).get("technically_correct"))

    if clean and aesthetic["polished"] and not occupancy_bad:
        action, reason = "accept", "No technical differences from the plan and every aesthetic dimension scored 4 or higher."
    elif clean and aesthetic["passed"] and (polish_rounds >= 1 or attempt_number >= max_attempts):
        action, reason = "accept", "Technically correct and acceptable; one polish round was already tried, so keep the best attempt."
    elif clean and attempt_number < max_attempts:
        action, reason = "edit", "Technically correct. Polish the appearance with a targeted edit; this attempt stays the fallback if the edit makes things worse."
    elif attempt_number >= max_attempts:
        action, reason = "stop", f"Reached the {max_attempts}-attempt limit. Show the best attempt and the remaining issues to the user."
    elif (blank or missing_elements > max(1, counts["planned_elements"] // 4)
          or edge_errors > max(2, round(counts["planned_connections"] * 0.3))
          or recent_edits >= 2 or (occupancy_bad and aesthetic["scores"]["composition"]["score"] <= 2)
          or sum(1 for i in issues if _edit_item(i)) > MAX_EDIT_ITEMS):
        action = "regenerate"
        reason = ("Too many structural differences for a reliable edit" if edge_errors or missing_elements else
                  "Two edits in a row did not converge" if recent_edits >= 2 else "The composition needs a fresh layout")
        reason += "; regenerate from the plan with the problems listed."
    else:
        action, reason = "edit", "The image is close; apply targeted edits to this attempt."

    ordered = sorted(all_issues, key=lambda i: (i["severity"] != "major", i["type"] in {"aesthetic", "objective"}))
    items = [item for item in (_edit_item(i) for i in ordered) if item]
    if action == "edit":
        prompt = ("Edit Image 1. Keep the layout, colors, typography and everything not listed unchanged. Make only these changes:\n"
                  + "\n".join(f"{n}. {item}" for n, item in enumerate(items[:MAX_EDIT_ITEMS], start=1)))
    elif action == "regenerate":
        skip = ("differs from the planned", "Nearly identical")
        avoid = [item for item in items if not any(s in item for s in skip)][:6]
        prompt = generation_prompt + ("\nThe previous attempt had these problems. Avoid them:\n" + "\n".join("- " + a for a in avoid) if avoid else "")
    else:
        prompt = ""
    return {"action": action, "reason": reason, "attempt": attempt_number, "max_attempts": max_attempts, "technically_correct": clean,
            "prompt": prompt, "change_list": items, "issue_count": {"major": len(major), "minor": len(all_issues) - len(major)}}
