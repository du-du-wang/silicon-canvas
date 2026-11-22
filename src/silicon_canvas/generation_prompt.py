"""Compact Image Gen prompts built from the compiled figure plan.

The engine's full figure prompt is a detailed contract: it lists every rule the
image must satisfy and is used for review. Image models follow a long list of
negative rules poorly, so the prompt actually sent to the generator is built
here instead:

1. one shared, positively described visual system (palette roles, type, lines);
2. a canvas chosen from the figure's topology, so the content fills the frame;
3. the exact content: elements, groups, connections (and timing rows);
4. at most four hard constraints.

``figure_content`` is also the reference that the review loop compares a cold
read against, so the generator and the reviewer use the same inventory.
"""
from __future__ import annotations

import json
from typing import Any

from . import visual_style

VERSION = "GEN1"

# Ratios offered by common image APIs. GPT Image offers 3:2, 1:1 and 2:3;
# Gemini and others also offer the wider/narrower ratios.
CANVASES = [("21:9", 21 / 9), ("16:9", 16 / 9), ("3:2", 3 / 2), ("4:3", 4 / 3),
            ("1:1", 1.0), ("3:4", 3 / 4), ("2:3", 2 / 3)]
GPT_IMAGE_SIZES = [("1536x1024", 1.5), ("1024x1024", 1.0), ("1024x1536", 2 / 3)]
EXPLICIT_ASPECTS = {"16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0, "A4": 1 / 1.414}


def _q(text: str) -> str:
    return json.dumps(str(text), ensure_ascii=False)


def _family(fig: dict[str, Any]) -> str:
    if fig.get("render_brief"):
        return "render"
    if fig.get("algorithm_flow_plan"):
        return "algorithm"
    if fig.get("behavior_plan"):
        return "behavior:" + fig["behavior_plan"].get("grammar", "directed-process")
    return "diagram"


# --- content model -----------------------------------------------------------

def figure_content(fig: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Elements, groups and connections the image must show, by visible label."""
    family = _family(fig)
    content: dict[str, Any] = {"family": family, "elements": [], "groups": [], "connections": [], "timing": None}
    if family == "algorithm":
        plan = fig["algorithm_flow_plan"]
        names = {}
        for obj in plan["objects"]:
            text = obj.get("text_plan", {})
            lines = [*text.get("mathematical_lines", []), *text.get("axis_lines", [])]
            if text.get("shape_once"):
                lines.append(text["shape_once"])
            names[obj["source_node_id"]] = obj["label"]
            content["elements"].append({"id": obj["source_node_id"], "label": obj["label"], "role": obj.get("role", "operator"),
                                        "lines": [line for line in lines if line], "children": []})
        for group in plan.get("groups", []):
            members = [names[m] for m in group.get("member_node_ids", []) if m in names]
            if members and group.get("label"):
                content["groups"].append({"label": group["label"], "members": members})
        for rel in plan["relations"]:
            content["connections"].append({"source_id": rel["source"], "target_id": rel["target"],
                                           "source": names.get(rel["source"], rel["source"]), "target": names.get(rel["target"], rel["target"]),
                                           "label": rel.get("label", ""), "style": rel.get("display_stroke", "solid")})
        columns = plan.get("composition", {}).get("display_columns", [])
        content["columns"] = [[names[n] for n in column.get("node_ids", []) if n in names] for column in columns]
        return content
    if family.startswith("behavior"):
        plan = fig["behavior_plan"]
        names = {obj["id"]: obj["label"] for obj in plan.get("objects", [])}
        for obj in plan.get("objects", []):
            content["elements"].append({"id": obj["id"], "label": obj["label"], "role": obj.get("kind", "state"), "lines": [], "children": []})
        for rel in plan.get("relations", []):
            content["connections"].append({"source_id": rel["source"], "target_id": rel["target"],
                                           "source": names.get(rel["source"], rel["source"]), "target": names.get(rel["target"], rel["target"]),
                                           "label": rel.get("label", ""), "style": "dashed" if rel.get("kind") in {"control", "clock"} and plan.get("grammar") != "state-machine" else "solid"})
        if plan.get("grammar") == "sampled-timing":
            content["timing"] = {"axis": plan.get("axis", []),
                                 "rows": [{"label": row["label"], "values": row.get("values", []), "kind": row.get("kind", "")} for row in plan.get("rows", [])]}
        groups = {}
        for obj in plan.get("objects", []):
            if obj.get("source_group"):
                groups.setdefault(obj["source_group"], []).append(obj["label"])
        labels = {g["id"]: g.get("label", "") for g in spec.get("groups", [])}
        content["groups"] = [{"label": labels.get(gid) or gid, "members": members} for gid, members in groups.items() if labels.get(gid)]
        return content
    if family == "diagram":
        names = {}
        group_labels = {g["id"]: g.get("label", "") for g in spec.get("groups", [])}
        for node in fig.get("macro_nodes", []):
            children = [child.get("label", "") for child in node.get("children", []) if child.get("label")]
            names[node["id"]] = node["label"]
            for child in node.get("children", []):
                names.setdefault(child.get("id", ""), child.get("label", ""))
            content["elements"].append({"id": node["id"], "label": node["label"], "role": "container" if children else node.get("kind", "other"),
                                        "lines": [], "children": children, "kind": node.get("kind", "other"),
                                        "owner": group_labels.get(node.get("original_group", ""), "")})
        owners = {}
        for element in content["elements"]:
            if element["owner"] and element["owner"] != element["label"]:
                owners.setdefault(element["owner"], []).append(element["label"])
        content["groups"] = [{"label": label, "members": members} for label, members in owners.items()]
        for edge in fig.get("macro_edges", []):
            content["connections"].append({"source_id": edge["source"], "target_id": edge["target"],
                                           "source": names.get(edge["source"], edge["source"]), "target": names.get(edge["target"], edge["target"]),
                                           "label": edge.get("label", ""), "style": edge.get("stroke_style", "solid"),
                                           "bidirectional": bool(edge.get("bidirectional"))})
        return content
    return content


def allowed_text(content: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Every string that may legitimately appear in the image."""
    texts = []
    for element in content["elements"]:
        texts += [element["label"], *element.get("lines", []), *element.get("children", [])]
    texts += [group["label"] for group in content["groups"]]
    texts += [c["label"] for c in content["connections"] if c.get("label")]
    if content.get("timing"):
        texts += content["timing"]["axis"]
        for row in content["timing"]["rows"]:
            texts += [row["label"], *row["values"]]
    texts += [*spec.get("verbatim_labels", [])]
    return [t for t in dict.fromkeys(str(t) for t in texts) if t.strip()]


# --- canvas ------------------------------------------------------------------

def _nearest(ratio: float) -> str:
    return min(CANVASES, key=lambda item: abs(item[1] - ratio) / item[1])[0]


def choose_canvas(fig: dict[str, Any], settings: dict[str, Any], content: dict[str, Any], guide_size: tuple[int, int] | None = None) -> dict[str, Any]:
    requested = settings.get("aspect", "auto")
    auto_chosen = settings.get("automatic_visual_settings", {}).get("requested", {}).get("aspect") == "auto"
    if requested in EXPLICIT_ASPECTS and not auto_chosen:
        ratio, basis = EXPLICIT_ASPECTS[requested], "user request"
    elif guide_size:
        ratio, basis = guide_size[0] / max(1, guide_size[1]), "layout guide"
    elif content.get("columns"):
        columns = max(1, len(content["columns"]))
        lanes = max([1, *(len(c) for c in content["columns"])])
        ratio, basis = max(0.6, min(2.6, (columns * 1.25) / (lanes * 0.9 + 0.6))), "dependency columns"
    elif content.get("timing"):
        ratio, basis = max(1.3, min(2.4, (len(content["timing"]["axis"]) + 2) / (len(content["timing"]["rows"]) + 1.5))), "timing table"
    else:
        count = max(1, len(content["elements"]))
        ratio, basis = (1.5 if count <= 8 else 4 / 3), "element count"
    name = _nearest(ratio)
    gpt = min(GPT_IMAGE_SIZES, key=lambda item: abs(item[1] - ratio) / item[1])[0]
    return {"aspect": name, "content_ratio": round(ratio, 2), "basis": basis, "gpt_image_size": gpt}


# --- visual system -----------------------------------------------------------

def visual_system(settings: dict[str, Any], family: str) -> str:
    style = settings.get("style", "paper")
    colors = visual_style.palette(style)
    ink, outline = colors["ink"], visual_style.boundary_color(colors, strong=True)
    frame = visual_style.boundary_color(colors)
    if style == "monochrome":
        fills = "Grayscale only: white, light gray " + colors["secondary"] + " and very light gray " + colors["muted"] + "; roles differ by shape and line style, not color."
    elif family == "algorithm":
        fills = f"Fill tensors {colors['primary']} and operators {visual_style.role_fill('memory', colors)}; state/accumulators white with a thin outline."
    else:
        fills = (f"Fill by role and nothing else: compute {visual_style.role_fill('compute', colors)}, memory/storage {visual_style.role_fill('memory', colors)}, "
                 f"control/state {visual_style.role_fill('state', colors)}, interfaces and external parts white. Containers are unfilled with a thin {frame} outline and a small quiet heading.")
    lines = [
        "Style: a clean, flat technical figure in the quality of a top-tier computer-architecture paper. Calm, precise and spacious.",
        f"Colors: white background, text and lines in {ink}, block outlines {outline}. {fills} Same role, same color everywhere.",
        "Type: one sans-serif typeface (Inter or Helvetica style). Block names medium weight, edge labels regular and smaller. Mathematics in proper italic math typesetting with real superscripts and subscripts.",
        "Lines: thin uniform strokes, small solid arrowheads, orthogonal routes with small rounded corners. Arrows start and end exactly on block borders and never pass through text.",
        "Layout: align peers on a common grid, keep spacing even, and let the figure fill the canvas with a modest margin.",
    ]
    if family == "algorithm":
        lines.append("Matrices may be drawn as pale rectangles. If grid lines are drawn, use one cell size in every matrix and keep shared dimensions visually consistent (the inner dimension of a product matches).")
    if family.startswith("behavior:state-machine"):
        lines.append("States are rounded rectangles; a self-transition is a small loop on the state's edge; guards are short labels beside their arrows.")
    if family.startswith("behavior:sampled-timing"):
        lines.append("Draw it as a clean timing table: one column per sample, one row per signal, values centered, light shading only for the highlighted spans.")
    return "\n".join(lines)


HARD_RULES = [
    "Print only the text listed above, spelled exactly; no title, legend, footnote, watermark or badge.",
    "Draw each listed connection exactly once in the stated direction, and no other arrows.",
    "Flat 2D only: no gradients, glow, shadows, 3D blocks, photos or device icons.",
]


def _content_block(content: dict[str, Any]) -> list[str]:
    out = ["Content. Print every quoted string exactly; words in [brackets] describe a role and are never printed."]
    for element in ([] if content.get("timing") else content["elements"]):
        if element["role"] == "container":
            out.append(f"- [container] {_q(element['label'])} holding " + ", ".join(_q(c) for c in element["children"]))
            continue
        extra = element.get("lines") or []
        out.append(f"- [{element.get('role', 'block')}] {_q(element['label'])}" + (", also shows " + "; ".join(_q(x) for x in extra) if extra else ""))
    for group in content["groups"]:
        out.append(f"- Group {_q(group['label'])} encloses: " + ", ".join(_q(m) for m in group["members"]))
    if content.get("timing"):
        timing = content["timing"]
        out.append("Timing table columns: " + ", ".join(timing["axis"]))
        for row in timing["rows"]:
            out.append(f"- row {_q(row['label'])}: " + (" ".join(row["values"]) if row["values"] else "sample markers only"))
    if content["connections"]:
        out.append("Connections (from -> to: label):")
        for c in content["connections"]:
            arrow = "<->" if c.get("bidirectional") else "->"
            style = " (dashed)" if c.get("style") == "dashed" else ""
            label = f": {_q(c['label'])}" if c.get("label") else ""
            out.append(f"- {_q(c['source'])} {arrow} {_q(c['target'])}{label}{style}")
    return out


def compose(fig: dict[str, Any], spec: dict[str, Any], settings: dict[str, Any], *, guide_size: tuple[int, int] | None = None,
            user_notes: str = "", feedback: str = "") -> dict[str, Any]:
    family = _family(fig)
    content = figure_content(fig, spec)
    if family == "render":
        # Product renders keep their art-direction brief; it already reads as a
        # positive description and is the best-performing path.
        return {"version": VERSION, "family": family, "prompt": fig["prompt"], "canvas": None, "content": content}
    canvas = choose_canvas(fig, settings, content, guide_size)
    orientation = "landscape" if canvas["content_ratio"] > 1.05 else "portrait" if canvas["content_ratio"] < 0.95 else "square"
    subject = {"algorithm": "an algorithm dataflow figure", "diagram": "an architecture block diagram"}.get(family)
    if not subject:
        subject = {"state-machine": "a state-machine diagram", "sampled-timing": "a sampled timing table",
                   "logical-topology": "a logical topology diagram"}.get(family.split(":", 1)[1], "a directed process diagram")
    parts = [f"Create {subject} with Image Gen: {spec.get('title', '').strip() or fig.get('title', '')}."]
    if fig.get("requires_layout_guide"):
        parts.append("Image 1 is a layout sketch. Keep its arrangement, containment, label positions and arrow routes, and redraw it as a finished figure in the style below.")
    elif content.get("columns"):
        bands = [" + ".join(_q(x) for x in column) for column in content["columns"] if column]
        parts.append("Reading order left to right: " + " | ".join(bands) + ". Independent inputs sit on separate lanes; operands enter their operator from separate sides.")
    gpt_ratio = {"1536x1024": 1.5, "1024x1024": 1.0, "1024x1536": 2 / 3}[canvas["gpt_image_size"]]
    fit = ("" if abs(gpt_ratio - canvas["content_ratio"]) / canvas["content_ratio"] < 0.15 else
           f" If the generator only offers {canvas['gpt_image_size']}, use it and fill that frame with larger blocks and text instead of leaving empty bands.")
    parts.append(f"Canvas: {canvas['aspect']} {orientation}. The figure spans about 90% of the width with balanced margins and no large empty bands.{fit}")
    parts.append(visual_system(settings, family))
    parts += _content_block(content)
    if user_notes.strip():
        parts.append("User appearance request (takes precedence over the style above): " + user_notes.strip())
    if feedback.strip():
        parts.append("Revision request: " + feedback.strip())
    rules = list(HARD_RULES)
    if not content["connections"]:
        rules[1] = "Draw no arrows or connectors other than the listed table structure and sample markers."
    parts.append("Rules:\n" + "\n".join("- " + rule for rule in rules))
    return {"version": VERSION, "family": family, "prompt": "\n".join(parts), "canvas": canvas, "content": content}
