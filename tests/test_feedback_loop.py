"""Review loop: compact prompt, objective image checks, cold-read diff, decision."""
import asyncio
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw
from mcp.client import Client

from silicon_canvas import engine, feedback_loop, generation_prompt
from silicon_canvas.mcp_server import create_server

SPEC = {
    "title": "SwiGLU Gated MLP", "category": "dataflow", "summary": "Gate and up projections combine elementwise.", "groups": [],
    "nodes": [
        {"id": "x", "label": "X", "kind": "io", "group": "", "detail": "N × d", "diagram_role": "tensor", "tensor_shape": "N × d", "evidence": "e", "origin": "given"},
        {"id": "g", "label": "G = XW_g", "kind": "compute", "group": "", "detail": "gate", "diagram_role": "operator", "evidence": "e", "origin": "given"},
        {"id": "silu", "label": "SiLU(·)", "kind": "compute", "group": "", "detail": "act", "diagram_role": "operator", "evidence": "e", "origin": "given"},
        {"id": "u", "label": "U = XW_u", "kind": "compute", "group": "", "detail": "up", "diagram_role": "operator", "evidence": "e", "origin": "given"},
        {"id": "h", "label": "H = SiLU(G) ⊙ U", "kind": "compute", "group": "", "detail": "N × h", "diagram_role": "operator", "tensor_shape": "N × h", "evidence": "e", "origin": "given"},
        {"id": "o", "label": "Y = HW_o", "kind": "compute", "group": "", "detail": "out", "diagram_role": "operator", "evidence": "e", "origin": "given"},
        {"id": "y", "label": "Y", "kind": "io", "group": "", "detail": "N × d", "diagram_role": "tensor", "tensor_shape": "N × d", "evidence": "e", "origin": "given"}],
    "edges": [
        {"source": "x", "target": "g", "kind": "data", "label": "", "evidence": "e", "origin": "given"},
        {"source": "x", "target": "u", "kind": "data", "label": "", "evidence": "e", "origin": "given"},
        {"source": "g", "target": "silu", "kind": "data", "label": "G", "evidence": "e", "origin": "given"},
        {"source": "silu", "target": "h", "kind": "data", "label": "", "evidence": "e", "origin": "given"},
        {"source": "u", "target": "h", "kind": "data", "label": "U", "evidence": "e", "origin": "given"},
        {"source": "h", "target": "o", "kind": "data", "label": "H", "evidence": "e", "origin": "given"},
        {"source": "o", "target": "y", "kind": "data", "label": "", "evidence": "e", "origin": "given"}]}

GOOD_READ = {
    "summary": "A gated MLP: X splits into a gate and an up projection that combine into H, then Y.",
    "texts": ["X", "N × d", "G = XW_g", "G", "SiLU(·)", "U = XW_u", "U", "H = SiLU(G) ⊙ U", "N × h", "H", "Y = HW_o", "Y", "N × d"],
    "connections": [{"from": "X", "to": "G = XW_g"}, {"from": "X", "to": "U = XW_u"}, {"from": "G = XW_g", "to": "SiLU(·)", "label": "G"},
                    {"from": "SiLU(·)", "to": "H = SiLU(G) ⊙ U"}, {"from": "U = XW_u", "to": "H = SiLU(G) ⊙ U", "label": "U"},
                    {"from": "H = SiLU(G) ⊙ U", "to": "Y = HW_o", "label": "H"}, {"from": "Y = HW_o", "to": "Y"}],
    "groups": [], "unclear": [], "defects": []}
GOOD_LOOKS = {name: {"score": 4, "note": ""} for name in feedback_loop.AESTHETIC_DIMENSIONS}


def compiled():
    return engine.compile_manual({"spec": copy.deepcopy(SPEC), "settings": {"representation": "diagram"}})


def figure_png(fill=0.9, size=(1536, 1024), color=(220, 234, 255)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    w, h = size
    mx, my = w * (1 - fill) / 2, h * (1 - fill) / 2
    for i in range(5):
        x0 = mx + i * (w - 2 * mx) / 5
        draw.rectangle([x0 + 10, my, x0 + (w - 2 * mx) / 5 - 10, h - my], fill=color, outline=(23, 50, 77), width=3)
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


class GenerationPromptTests(unittest.TestCase):
    def test_generation_prompt_is_compact_positive_and_complete(self):
        result = compiled()
        figure = result["figures"][0]
        prompt = figure["generation_prompt"]
        self.assertLess(len(prompt), len(figure["prompt"]) / 2)
        self.assertLess(len(prompt), 3000)
        for node in SPEC["nodes"]:
            self.assertIn(json.dumps(node["label"], ensure_ascii=False), prompt)
        self.assertIn('"G = XW_g" -> "SiLU(·)": "G"', prompt)
        self.assertLessEqual(prompt.count("\n- ", prompt.index("Rules:")), 4)
        self.assertIn("Fill tensors", prompt)

    def test_canvas_follows_topology_unless_user_sets_it(self):
        result = compiled()
        self.assertEqual(result["figures"][0]["generation_canvas"]["basis"], "dependency columns")
        self.assertIn(result["figures"][0]["generation_canvas"]["aspect"], {"21:9", "16:9"})
        fixed = engine.compile_manual({"spec": copy.deepcopy(SPEC), "settings": {"representation": "diagram", "aspect": "1:1"}})
        self.assertEqual(fixed["figures"][0]["generation_canvas"]["aspect"], "1:1")
        self.assertEqual(fixed["figures"][0]["generation_canvas"]["basis"], "user request")


class ObjectiveCheckTests(unittest.TestCase):
    def test_well_filled_image_has_no_composition_issue(self):
        image, _ = feedback_loop.load_image(figure_png(0.9))
        checks = feedback_loop.inspect_image(image)
        self.assertGreater(checks["metrics"]["occupancy"], 0.75)
        self.assertEqual([i for i in checks["issues"] if i["dimension"] == "composition"], [])

    def test_small_figure_in_large_canvas_is_reported(self):
        image, _ = feedback_loop.load_image(figure_png(0.4))
        issues = feedback_loop.inspect_image(image)["issues"]
        self.assertTrue(any("canvas" in i["issue"] for i in issues))

    def test_rainbow_palette_and_duplicates_are_reported(self):
        image = Image.new("RGB", (1200, 800), "white")
        draw = ImageDraw.Draw(image)
        for i, color in enumerate([(230, 40, 40), (40, 200, 40), (40, 80, 230), (240, 200, 20), (170, 40, 220), (20, 200, 210)]):
            draw.rectangle([60 + i * 185, 100, 220 + i * 185, 700], fill=color)
        first = feedback_loop.inspect_image(image)
        self.assertTrue(any(i["dimension"] == "color" for i in first["issues"]))
        again = feedback_loop.inspect_image(image, previous_hashes=[first["metrics"]["dhash"]])
        self.assertEqual(again["metrics"]["near_duplicate_of_attempt"], 1)

    def test_non_images_are_rejected(self):
        with self.assertRaises(ValueError):
            feedback_loop.load_image(b"%PDF-1.4 not an image")


class CompareTests(unittest.TestCase):
    def setUp(self):
        result = compiled()
        self.figure = result["figures"][0]
        self.content = generation_prompt.figure_content(self.figure, result["spec"])
        self.allowed = generation_prompt.allowed_text(self.content, result["spec"])

    def test_faithful_cold_read_has_no_differences(self):
        comparison = feedback_loop.compare(self.content, self.allowed, GOOD_READ)
        self.assertEqual(comparison["issues"], [])
        self.assertEqual(comparison["counts"]["matched_connections"], 7)

    def test_misspelling_reversal_extra_and_missing_are_found(self):
        read = copy.deepcopy(GOOD_READ)
        read["texts"] = [t.replace("SiLU(·)", "SiUL(·)") for t in read["texts"]] + ["Gate"]
        read["connections"][3] = {"from": "H = SiLU(G) ⊙ U", "to": "SiUL(·)"}
        read["connections"].append({"from": "X", "to": "Y"})
        del read["connections"][0]
        kinds = {i["type"] for i in feedback_loop.compare(self.content, self.allowed, read)["issues"]}
        self.assertTrue({"misspelled_text", "reversed_connection", "extra_connection", "missing_connection", "unexpected_text"} <= kinds)

    def test_label_inside_another_equation_does_not_count_as_present(self):
        read = copy.deepcopy(GOOD_READ)
        read["texts"].remove("SiLU(·)")
        issues = feedback_loop.compare(self.content, self.allowed, read)["issues"]
        self.assertIn({"type": "missing_element", "severity": "major", "expected": "SiLU(·)", "where": ""}, issues)

    def test_container_children_resolve_to_their_container(self):
        content = {"elements": [{"id": "c", "label": "Compute", "role": "container", "children": ["CPU", "NPU"], "lines": []},
                                {"id": "m", "label": "Memory", "role": "memory", "children": [], "lines": []}],
                   "groups": [], "connections": [{"source_id": "c", "target_id": "m", "source": "Compute", "target": "Memory", "label": ""}]}
        read = {"texts": ["Compute", "CPU", "NPU", "Memory"], "connections": [{"from": "NPU", "to": "Memory"}]}
        self.assertEqual(feedback_loop.compare(content, ["Compute", "CPU", "NPU", "Memory"], read)["issues"], [])


class DecisionTests(unittest.TestCase):
    def setUp(self):
        result = compiled()
        self.content = generation_prompt.figure_content(result["figures"][0], result["spec"])
        self.allowed = generation_prompt.allowed_text(self.content, result["spec"])

    def decide(self, read, looks, objective=(), history=()):
        comparison = feedback_loop.compare(self.content, self.allowed, read)
        return feedback_loop.decide(comparison, feedback_loop.aesthetic_check(looks), list(objective), list(history), generation_prompt="GEN")

    def test_clean_and_polished_is_accepted(self):
        self.assertEqual(self.decide(GOOD_READ, GOOD_LOOKS)["action"], "accept")

    def test_correct_but_unpolished_gets_one_polish_edit_then_accepts(self):
        looks = dict(GOOD_LOOKS, composition={"score": 3, "note": "Large empty bands above and below."})
        first = self.decide(GOOD_READ, looks)
        self.assertEqual(first["action"], "edit")
        self.assertTrue(first["technically_correct"])
        self.assertIn("Edit Image 1", first["prompt"])
        self.assertIn("empty bands", first["prompt"])
        second = self.decide(GOOD_READ, looks, history=[{"decision": first}])
        self.assertEqual(second["action"], "accept")

    def test_small_technical_errors_are_edited_with_a_change_list(self):
        read = copy.deepcopy(GOOD_READ)
        read["texts"] = [t.replace("U = XW_u", "U = XW_v") for t in read["texts"]]
        decision = self.decide(read, GOOD_LOOKS)
        self.assertEqual(decision["action"], "edit")
        self.assertIn('Change the text "U = XW_v" to exactly "U = XW_u".', decision["prompt"])
        self.assertLess(len(decision["prompt"]), 800)

    def test_structural_errors_regenerate_from_the_generation_prompt(self):
        read = copy.deepcopy(GOOD_READ)
        read["connections"] = read["connections"][:3] + [{"from": "Y", "to": "X"}, {"from": "G = XW_g", "to": "Y"}]
        decision = self.decide(read, GOOD_LOOKS)
        self.assertEqual(decision["action"], "regenerate")
        self.assertTrue(decision["prompt"].startswith("GEN"))
        self.assertIn("Avoid them", decision["prompt"])

    def test_attempt_limit_stops(self):
        read = copy.deepcopy(GOOD_READ)
        read["texts"].remove("SiLU(·)")
        history = [{"decision": {"action": "regenerate"}}] * 3
        self.assertEqual(self.decide(read, GOOD_LOOKS, history=history)["action"], "stop")


class McpLoopTests(unittest.TestCase):
    def test_host_reported_geometry_error_does_not_outrank_accepted_image(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "attempt.png"
                async with Client(create_server(Path(directory) / "out")) as client:
                    data = (await client.call_tool("silicon_compile", {"spec": copy.deepcopy(SPEC), "settings": {"representation": "diagram"}})).structured_content
                    artifact = {"artifact_id": data["artifact_id"]}
                    base_read = dict(artifact, cold_read=GOOD_READ, aesthetic=GOOD_LOOKS)
                    image.write_bytes(figure_png(0.9))
                    await client.call_tool("silicon_review_image", dict(artifact, image_path=str(image)))
                    first = (await client.call_tool("silicon_submit_review", dict(
                        base_read, attempt=1, host_findings=[{"technical": True, "issue": "The operand is outside its required boundary."}],
                    ))).structured_content
                    self.assertEqual(first["decision"]["action"], "edit")
                    self.assertEqual(first["differences"], [])
                    self.assertLess(first["score"], 100)
                    image.write_bytes(figure_png(0.85))
                    await client.call_tool("silicon_review_image", dict(artifact, image_path=str(image)))
                    second = (await client.call_tool("silicon_submit_review", dict(base_read, attempt=2))).structured_content
                    self.assertEqual(second["decision"]["action"], "accept")
                    self.assertEqual(second["best_attempt"]["attempt"], 2)
        asyncio.run(run())

    def test_review_tools_round_trip_and_keep_best_attempt(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "attempt.png"
                image.write_bytes(figure_png(0.9))
                async with Client(create_server(Path(directory) / "out")) as client:
                    data = (await client.call_tool("silicon_compile", {"spec": copy.deepcopy(SPEC), "settings": {"representation": "diagram"}})).structured_content
                    self.assertEqual(data["prompt"], compiled()["figures"][0]["generation_prompt"])
                    contract = await client.read_resource(data["contract_uri"])
                    self.assertIn("Closed outgoing-connection contract", contract.contents[0].text)
                    started = (await client.call_tool("silicon_review_image", {"artifact_id": data["artifact_id"], "image_path": str(image)})).structured_content
                    self.assertEqual(started["attempt"], 1)
                    self.assertTrue(started["cold_read"]["blind"])
                    task_text = json.dumps(started["cold_read"], ensure_ascii=False)
                    self.assertNotIn("SiLU", task_text)
                    self.assertNotIn("XW_g", task_text)
                    bad = copy.deepcopy(GOOD_READ)
                    bad["connections"] = bad["connections"][:2]
                    first = (await client.call_tool("silicon_submit_review", {"artifact_id": data["artifact_id"], "attempt": 1, "cold_read": bad, "aesthetic": GOOD_LOOKS})).structured_content
                    self.assertEqual(first["decision"]["action"], "regenerate")
                    self.assertEqual(first["review_status"], "in_review")
                    await client.call_tool("silicon_review_image", {"artifact_id": data["artifact_id"], "image_path": str(image)})
                    second = (await client.call_tool("silicon_submit_review", {"artifact_id": data["artifact_id"], "attempt": 2, "cold_read": GOOD_READ, "aesthetic": GOOD_LOOKS})).structured_content
                    self.assertEqual(second["decision"]["action"], "accept")
                    self.assertEqual(second["review_status"], "accepted")
                    self.assertEqual(second["best_attempt"]["attempt"], 2)
                    rejected = await client.call_tool("silicon_review_image", {"artifact_id": data["artifact_id"], "image_path": str(Path(directory) / "notes.txt")})
                    self.assertTrue(rejected.is_error)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
