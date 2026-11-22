"""Silicon Canvas local stdio MCP: direct calls into the shared compiler."""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations

from . import __version__, bundle, engine, feedback_loop, generation_prompt, layout_guide, provider, reading_plan, service
from .artifact_store import ArtifactStore

MAX_INPUT_BYTES = 2 * 1024 * 1024


def _bounded(value: Any) -> None:
    try:
        size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        raise ValueError("Input must contain JSON values.") from None
    if size > MAX_INPUT_BYTES:
        raise ValueError("Input exceeds 2 MiB. Split the system into separate diagrams.")


def create_server(directory: Path | None = None) -> MCPServer:
    store = ArtifactStore(directory)
    mcp = MCPServer(
        "Silicon Canvas",
        version=__version__,
        instructions=(
            "Create chip architecture, datapath and control Image Gen briefs. "
            "Default workflow: silicon_prepare(description), use the returned semantic messages to create the JSON spec with your own model, then silicon_compile(spec, preparation_id). "
            "No extra API key is needed for that workflow. Choose the medium and explanatory scope from the user's intent: material product rendering, functional/spatial illustration, or exact engineering notation. Rendered functional regions, schematic external resources and interfaces can explain architecture without asserting a measured physical layout. A cover hero may prioritize exterior; a system explanation must convey useful structure. Diagrams use the exact prompt and target guide. "
            "New artwork uses English when language is omitted or auto, including for Chinese descriptions; source evidence keeps its original language. Explicit saved zh settings remain supported for historical projects and their revisions. "
            "silicon_generate optionally uses a separately configured semantic provider. "
            "After Image Gen, run the review loop: silicon_review_image(artifact_id, image_path) measures the image and returns a blind cold-read task; "
            "do the cold read without consulting the request (ideally in a fresh context), then silicon_submit_review returns the differences and an accept/edit/regenerate/stop decision with the next prompt. "
            "Repeat until accept or stop. silicon_prepare_review returns the full spec-aware review checklist. "
            "Use artifact handles returned by tools; this server never reads arbitrary user-supplied paths."
        ),
    )

    def figure(result: dict[str, Any], figure_id: str = "") -> dict[str, Any]:
        requested = figure_id or result.get("selected_figure_id", "")
        for item in result.get("figures", []):
            if item["id"] == requested:
                return item
        raise ValueError("Figure does not exist. Use a figure_id listed by silicon_compile.")

    def save_result(result: dict[str, Any]) -> str:
        artifact_id = store.create("compiled", result)
        store.write(artifact_id, "result.json", json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
        for item in result.get("figures", []):
            figure_id = item["id"]
            guide = layout_guide.render(result, item)
            if guide:
                store.write(artifact_id, store.figure_name(figure_id, "guide.png"), guide)
            store.write(artifact_id, store.figure_name(figure_id, "prompt.txt"), item.get("generation_prompt", item["prompt"]).encode("utf-8"))
            store.write(artifact_id, store.figure_name(figure_id, "contract.txt"), item["prompt"].encode("utf-8"))
            store.write(artifact_id, store.figure_name(figure_id, "caption.txt"), item["caption"].encode("utf-8"))
            store.write(artifact_id, store.figure_name(figure_id, "review.json"), json.dumps(item.get("review_task", {}), ensure_ascii=False).encode("utf-8"))
        return artifact_id

    def compiled_response(artifact_id: str, result: dict[str, Any]) -> CallToolResult:
        selected = figure(result)
        selected_id = selected["id"]
        guide_path = store.path(artifact_id, store.figure_name(selected_id, "guide.png"))
        guide_uri = store.uri(artifact_id, "figures/" + selected_id + "/guide.png") if guide_path.is_file() else None
        data = {
            "artifact_id": artifact_id,
            "source_hash": reading_plan.source_hash(result["spec"]),
            "representation": selected.get("representation", "diagram"),
            "representation_profile": result.get("representation_profile"),
            "render_brief": selected.get("render_brief"),
            "algorithm_flow_plan": selected.get("algorithm_flow_plan"),
            "diagram_design": selected.get("diagram_design"),
            "visual_contract": selected.get("visual_contract"),
            "behavior_plan": selected.get("behavior_plan"),
            "figure_strategy": selected.get("figure_strategy"),
            "explanation_plan": selected.get("render_brief", {}).get("explanation_plan"),
            "generation_steps": selected.get("generation_steps", []),
            "guide_role": None if selected.get("algorithm_flow_plan") else "composition_reference" if selected.get("representation") in {"product_render", "system_illustration"} else "target_structure",
            "guide_optional": bool(selected.get("algorithm_flow_plan")) or selected.get("representation") in {"product_render", "system_illustration"},
            "selected_figure_id": selected_id,
            "figures": [{"id": item["id"], "title": item["title"], "primary": item.get("primary", False), "technical_annex": item.get("technical_annex", False), "brief_uri": store.uri(artifact_id, "figures/" + item["id"] + "/brief.json")} for item in result.get("figures", [])],
            "prompt": selected.get("generation_prompt", selected["prompt"]),
            "canvas": selected.get("generation_canvas"),
            "contract_uri": store.uri(artifact_id, "figures/" + selected_id + "/contract.txt"),
            "caption": selected["caption"],
            "warnings": result.get("warnings", []),
            "preflight": {
                "layout_strategy": selected.get("layout_strategy"),
                "quality": selected.get("layout_scene", {}).get("quality", {}),
                "diagnostics": selected.get("composition_diagnostics", []),
                "meaning": "Compiler geometry and typography checks only; the actual Image Gen bitmap has not been reviewed.",
            },
            "guide_uri": guide_uri,
            "guide_path": str(guide_path) if guide_uri else None,
            "result_uri": store.uri(artifact_id, "result.json"),
            "review_uri": store.uri(artifact_id, "figures/" + selected_id + "/review.json"),
            "review_status": "not_reviewed",
            "next_step": "Use Image Gen with the selected prompt and target guide. Review the actual image using silicon_prepare_review; preflight does not certify it.",
        }
        if selected.get("algorithm_flow_plan"):
            data["preflight"]["meaning"] = "Source-backed mathematical roles and dependency placement only. This algorithm view intentionally has no hardware target guide; the actual Image Gen bitmap has not been reviewed."
            data["next_step"] = "Use Image Gen with the selected algorithm prompt alone. Follow algorithm_flow_plan for distinct operands, tensor/operator shapes, branches, source cycles and residency. No hardware target guide is required. Review the actual bitmap using silicon_prepare_review."
        elif data["guide_optional"]:
            data["preflight"]["meaning"] = "Art-direction and source/assumption contract only. The optional framing guide is not CAD geometry or proof of a manufactured product. Review the actual Image Gen render."
            data["next_step"] = "Use the complete render prompt with Image Gen, following explanation_plan for functional regions, context and relationships. The optional composition reference only suggests framing. Review clarity, source-backed visible structure and aesthetic quality in the actual image."
            if selected.get("render_brief", {}).get("workflow", {}).get("recommended"):
                data["next_step"] = "Use generation_steps: first generate the complete informative illustration, then optionally refine that actual Image Gen output using the edit step. Preserve its planned functions, ownership and relationships during refinement. The edit target is the real first output, never the framing reference. Review the finished image."
        elif not guide_uri:
            data["next_step"] = "This engineering view has no available controlled target guide. Inspect figure_strategy.limitations and preflight diagnostics, recompose the scope, then compile again before using the new prompt and guide with Image Gen. The current compilation has not certified an actual image."
        data["next_step"] += (" Then run the review loop: silicon_review_image(artifact_id, image_path) -> blind cold read -> "
                              "silicon_submit_review(...), and follow its decision until it returns accept or stop.")
        content = [TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]
        if guide_uri:
            content.append(ImageContent(type="image", data=base64.b64encode(guide_path.read_bytes()).decode("ascii"), mime_type="image/png"))
        return CallToolResult(content=content, structured_content=data)

    closed_read = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    closed_write = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)

    @mcp.tool(annotations=closed_write)
    def silicon_prepare(description: str, settings: dict[str, Any] | None = None, previous_artifact_id: str = "", feedback: str = "", lock_structure: bool = True) -> dict[str, Any]:
        """Prepare the technical source and visual intent without an API key or external request. representation defaults to auto: product_render for commercial product CGI, system_illustration for spatial explanations, diagram for exact engineering figures. scene independently selects concept/system/flow/detail scope; style selects accent palette. Natural-language notes control camera, material, lighting, background and composition. Keep logical capabilities separate from supplied physical facts and illustrative appearance."""
        try:
            _bounded([description, settings, feedback])
            payload = {"scene": "auto", "representation": "auto", **(settings or {}), "description": description, "feedback": feedback, "lock_structure": lock_structure}
            if previous_artifact_id:
                payload["previous"] = store.read(previous_artifact_id, "compiled")
            messages, context = engine.prepare_request(payload)
            preparation_id = store.create("prepared", {"context": context})
            return {"preparation_id": preparation_id, "messages": messages, "next_tool": "silicon_compile", "instructions": "Create one JSON spec obeying these messages, then call silicon_compile(spec, preparation_id). Preserve assumptions and exact technical facts. No additional API key is needed."}
        except ValueError as error:
            raise ToolError(str(error)) from None
        except OSError:
            raise ToolError("Cannot save local preparation. Check SILICON_CANVAS_OUTPUT_DIR permissions.") from None

    @mcp.tool(annotations=closed_write)
    def silicon_compile(spec: dict[str, Any], preparation_id: str = "", settings: dict[str, Any] | None = None, description: str = "", figure_id: str = "") -> CallToolResult:
        """Compile a semantic spec into an Image Gen brief and source-preserving artifacts. Prefer the preparation_id from silicon_prepare. Algorithm dataflow uses the prompt alone and returns a source-backed tensor/operator/dependency plan, without a hardware target guide. Product renders/spatial illustrations work from the prompt alone; their returned composition reference is optional framing, never physical geometry. Engineering diagrams use the exact target guide. Without preparation_id, legacy specs lacking representation keep diagram behavior. Check guide_role and guide_optional before using any returned image."""
        try:
            _bounded([spec, settings, description])
            if preparation_id:
                if settings or description:
                    raise ValueError("When preparation_id is supplied, its description and settings are authoritative; omit separate settings and description.")
                context = copy.deepcopy(store.read(preparation_id, "prepared")["context"])
                if len(figure_id) > 80:
                    raise ValueError("Figure ID exceeds 80 characters.")
                context["figure_id"] = figure_id.strip()
                result = engine.finish_request(spec, context)
            else:
                result = engine.compile_manual({"spec": spec, "settings": settings or {}, "description": description, "figure_id": figure_id})
            if figure_id:
                figure(result, figure_id)
            artifact_id = save_result(result)
            return compiled_response(artifact_id, result)
        except ValueError as error:
            raise ToolError(str(error)) from None
        except OSError:
            raise ToolError("Cannot save compiled artifacts. Check SILICON_CANVAS_OUTPUT_DIR permissions.") from None

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True))
    def silicon_generate(description: str, settings: dict[str, Any] | None = None, previous_artifact_id: str = "", feedback: str = "", lock_structure: bool = True) -> CallToolResult:
        """Optional semantic-provider workflow: use locally configured DEEPSEEK_API_KEY/BASE_URL/MODEL to create and compile a spec. This makes a paid external API request and returns prompt/guide, not the final generated image. Use silicon_prepare + silicon_compile to use the host model without a separate key."""
        try:
            _bounded([description, settings, feedback])
            payload = {"scene": "auto", "representation": "auto", **(settings or {}), "description": description, "feedback": feedback, "lock_structure": lock_structure}
            if previous_artifact_id:
                payload["previous"] = store.read(previous_artifact_id, "compiled")
            result = service.generate_result(payload, engine)
            artifact_id = save_result(result)
            return compiled_response(artifact_id, result)
        except (ValueError, provider.ProviderError, service.GenerationError) as error:
            raise ToolError(str(error)) from None
        except OSError:
            raise ToolError("Cannot save generated artifacts. Check SILICON_CANVAS_OUTPUT_DIR permissions.") from None

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False))
    def silicon_export(artifact_id: str, figure_id: str = "") -> dict[str, Any]:
        """Export a saved compilation as a ZIP containing exact prompts, guides, captions, full source and review instructions. figure_id selects one view; omitted exports all views. Writes only into the owned artifact directory and returns a local download path and binary MCP resource."""
        try:
            result = store.read(artifact_id, "compiled")
            if figure_id:
                figure(result, figure_id)
            name = "export.zip" if not figure_id else figure_id + ".export.zip"
            if figure_id:
                store.figure_name(figure_id, "guide.png")  # Validate before using it as a filename.
            data = bundle.build(result, figure_id)
            path = store.write(artifact_id, name, data)
            return {"artifact_id": artifact_id, "figure_id": figure_id, "path": str(path), "bytes": len(data), "uri": store.uri(artifact_id, "exports/" + name), "mime_type": "application/zip"}
        except ValueError as error:
            raise ToolError(str(error)) from None
        except OSError:
            raise ToolError("Cannot export local artifacts. Check output directory permissions.") from None

    @mcp.tool(annotations=closed_read)
    def silicon_prepare_review(artifact_id: str, figure_id: str = "") -> dict[str, Any]:
        """Prepare a source-grounded review handoff for an actual Image Gen bitmap. Render review covers subject, perspective, materials, lighting, claim fidelity and appearance assumptions; diagram review covers ownership, endpoints, directions and labels. This tool does not inspect a bitmap or accept it. Ask the host to inspect the actual image using the returned form-specific manifest."""
        try:
            result = store.read(artifact_id, "compiled")
            selected = figure(result, figure_id)
            review = copy.deepcopy(selected.get("review_task", {}))
            return {"artifact_id": artifact_id, "figure_id": selected["id"], "status": "not_reviewed", "review_task": review, "instructions": "Inspect the actual generated bitmap with your host's image capability and this source-grounded manifest. Report technical correctness, reading structure and document-size appearance separately. This tool has prepared evidence only."}
        except ValueError as error:
            raise ToolError(str(error)) from None

    def loop_record(artifact_id: str, figure_id: str) -> dict[str, Any]:
        path = store.path(artifact_id, store.figure_name(figure_id, "loop.json"))
        if not path.is_file():
            return {"version": feedback_loop.VERSION, "figure_id": figure_id, "attempts": [], "status": "not_reviewed"}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_loop(artifact_id: str, figure_id: str, record: dict[str, Any]) -> None:
        store.write(artifact_id, store.figure_name(figure_id, "loop.json"), json.dumps(record, ensure_ascii=False, indent=1).encode("utf-8"))

    def read_image_input(image_path: str, image_base64: str) -> bytes:
        if bool(image_path) == bool(image_base64):
            raise ValueError("Provide exactly one of image_path or image_base64.")
        if image_base64:
            try:
                data = base64.b64decode(image_base64, validate=True)
            except (ValueError, TypeError):
                raise ValueError("image_base64 is not valid base64.") from None
            return data
        path = Path(image_path).expanduser()
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("image_path must be a .png, .jpg, .jpeg or .webp file.")
        try:
            if not path.is_file():
                raise ValueError("image_path does not point to a file.")
            if path.stat().st_size > feedback_loop.MAX_IMAGE_BYTES:
                raise ValueError("Image exceeds 40 MB.")
            return path.read_bytes()
        except OSError:
            raise ValueError("Cannot read image_path.") from None

    @mcp.tool(annotations=closed_write)
    def silicon_review_image(artifact_id: str, image_path: str = "", image_base64: str = "", figure_id: str = "") -> dict[str, Any]:
        """Start reviewing one generated image. Reads only the given PNG/JPEG/WebP file (or base64), stores a copy as the next attempt, measures it (canvas use, margins, aspect, palette, near-duplicate) and returns a BLIND cold-read task with no source facts plus the aesthetic rubric. Do the cold read before re-reading the request, ideally in a fresh context, then call silicon_submit_review."""
        try:
            result = store.read(artifact_id, "compiled")
            selected = figure(result, figure_id)
            fid = selected["id"]
            data = read_image_input(image_path, image_base64)
            image, extension = feedback_loop.load_image(data)
            record = loop_record(artifact_id, fid)
            content = generation_prompt.figure_content(selected, result["spec"])
            canvas = selected.get("generation_canvas") or {}
            planned = dict(generation_prompt.CANVASES).get(canvas.get("aspect", ""))
            gpt = {"1536x1024": 1.5, "1024x1024": 1.0, "1024x1536": 2 / 3}.get(canvas.get("gpt_image_size", ""))
            actual = image.size[0] / image.size[1]
            expected = None if gpt and abs(actual - gpt) / gpt < 0.03 else planned
            width_mm = selected.get("diagram_design", {}).get("typography", {}).get("target_width_mm") or selected.get("design_contract", {}).get("target_width_mm")
            checks = feedback_loop.inspect_image(image, expected_aspect=expected, target_width_mm=width_mm,
                                                 flat_figure=content["family"] != "render",
                                                 previous_hashes=[a.get("metrics", {}).get("dhash", "") for a in record["attempts"]])
            number = len(record["attempts"]) + 1
            name = f"{fid}.attempt-{number}.{extension}"
            stored = store.write(artifact_id, name, data)
            record["attempts"].append({"attempt": number, "image": name, "metrics": checks["metrics"], "objective_issues": checks["issues"], "status": "awaiting_cold_read"})
            record["status"] = "in_review"
            save_loop(artifact_id, fid, record)
            task = feedback_loop.cold_read_task(content["family"])
            return {"artifact_id": artifact_id, "figure_id": fid, "attempt": number, "stored_image_path": str(stored),
                    "objective": checks, "cold_read": task,
                    "next_tool": "silicon_submit_review",
                    "instructions": "Look at the image and do the cold read now, without consulting the request or plan. Score the aesthetic rubric from the image. Then call silicon_submit_review(artifact_id, attempt, cold_read, aesthetic). You may add spec-aware host_findings after the cold read."}
        except ValueError as error:
            raise ToolError(str(error)) from None
        except OSError:
            raise ToolError("Cannot save the review attempt. Check output directory permissions.") from None

    @mcp.tool(annotations=closed_write)
    def silicon_submit_review(artifact_id: str, attempt: int, cold_read: dict[str, Any], aesthetic: dict[str, Any], figure_id: str = "",
                              host_findings: list[dict[str, Any]] | None = None, max_attempts: int = feedback_loop.DEFAULT_MAX_ATTEMPTS) -> dict[str, Any]:
        """Compare a blind cold read with the figure plan and decide the next step. Returns differences (missing/misspelled/extra text, missing/reversed/extra arrows, timing values), the aesthetic result, and a decision: accept, edit (with a short edit prompt for this attempt's image), regenerate (generation prompt plus the problems to avoid) or stop (attempt limit; returns the best attempt). host_findings: optional [{issue, technical}] the host noticed after the cold read."""
        try:
            _bounded([cold_read, aesthetic, host_findings])
            if not 1 <= max_attempts <= 10:
                raise ValueError("max_attempts must be between 1 and 10.")
            result = store.read(artifact_id, "compiled")
            selected = figure(result, figure_id)
            fid = selected["id"]
            record = loop_record(artifact_id, fid)
            if not 1 <= attempt <= len(record["attempts"]):
                raise ValueError("Unknown attempt. Call silicon_review_image first.")
            entry = record["attempts"][attempt - 1]
            content = generation_prompt.figure_content(selected, result["spec"])
            comparison = feedback_loop.compare(content, generation_prompt.allowed_text(content, result["spec"]), cold_read)
            rubric = feedback_loop.aesthetic_check(aesthetic)
            history = [a for a in record["attempts"][:attempt - 1] if a.get("decision")]
            decision = feedback_loop.decide(comparison, rubric, entry["objective_issues"], history,
                                            generation_prompt=selected.get("generation_prompt", selected["prompt"]),
                                            max_attempts=max_attempts, host_findings=host_findings)
            score = feedback_loop.attempt_score(comparison, rubric, entry["objective_issues"], host_findings)
            entry.update(status="reviewed", comparison=comparison, aesthetic=rubric, decision=decision, score=score,
                         host_findings=host_findings or [])
            reviewed = [a for a in record["attempts"] if a.get("score") is not None]
            best = max(reviewed, key=lambda a: (a.get("decision", {}).get("technically_correct", False), a["score"]))
            record["best_attempt"] = best["attempt"]
            record["status"] = {"accept": "accepted", "stop": "stopped"}.get(decision["action"], "in_review")
            save_loop(artifact_id, fid, record)
            inputs = {"edit": "Image 1 = this attempt's image: " + str(store.path(artifact_id, entry["image"])),
                      "regenerate": ("Image 1 = the layout guide: " + str(store.path(artifact_id, store.figure_name(fid, "guide.png"))))
                      if selected.get("requires_layout_guide") else "No input image."}
            return {"artifact_id": artifact_id, "figure_id": fid, "attempt": attempt, "score": score,
                    "decision": decision, "image_inputs": inputs.get(decision["action"]),
                    "differences": comparison["issues"], "counts": comparison["counts"],
                    "reading_check": {"cold_summary": comparison["cold_summary"],
                                      "intended_question": selected.get("diagram_design", {}).get("reading_question") or selected.get("title", ""),
                                      "note": "If the cold summary misses the intended point, treat it as a hierarchy problem and add it as a host finding next time."},
                    "aesthetic": rubric,
                    "best_attempt": {"attempt": best["attempt"], "score": best["score"], "image_path": str(store.path(artifact_id, best["image"]))},
                    "review_status": record["status"],
                    "next_step": {"accept": "Show the image to the user, with any minor polish notes.",
                                  "edit": "Use Image Gen in edit mode on this attempt's image with decision.prompt, then call silicon_review_image with the new file.",
                                  "regenerate": "Generate a new image from decision.prompt (with the guide if listed), then call silicon_review_image with the new file.",
                                  "stop": "Show the best attempt to the user with the remaining differences and ask how to proceed."}[decision["action"]]}
        except ValueError as error:
            raise ToolError(str(error)) from None
        except OSError:
            raise ToolError("Cannot save the review result. Check output directory permissions.") from None

    @mcp.resource("silicon://catalog", mime_type="application/json")
    def catalog() -> dict[str, Any]:
        """Styles and diagram grammar supported by the shared compiler."""
        return engine.CATALOG

    @mcp.resource("silicon://artifacts/{artifact_id}/result.json", mime_type="application/json")
    def result_resource(artifact_id: str) -> dict[str, Any]:
        """Complete canonical source, projection mappings, prompts and preflight evidence."""
        return store.read(artifact_id, "compiled")

    @mcp.resource("silicon://artifacts/{artifact_id}/figures/{figure_id}/guide.png", mime_type="image/png")
    def guide_resource(artifact_id: str, figure_id: str) -> bytes:
        """Guide PNG. Diagrams use it as target structure; renders use it only as an optional framing reference. The figure brief records its role."""
        result = store.read(artifact_id, "compiled")
        figure(result, figure_id)
        path = store.path(artifact_id, store.figure_name(figure_id, "guide.png"))
        if not path.is_file():
            raise ValueError("This view has no controlled target guide.")
        return path.read_bytes()

    @mcp.resource("silicon://artifacts/{artifact_id}/figures/{figure_id}/brief.json", mime_type="application/json")
    def figure_resource(artifact_id: str, figure_id: str) -> dict[str, Any]:
        """One saved view's prompt, caption and target guide, without recompiling or loading every view."""
        result = store.read(artifact_id, "compiled")
        selected = figure(result, figure_id)
        guide_path = store.path(artifact_id, store.figure_name(figure_id, "guide.png"))
        return {
            "artifact_id": artifact_id, "figure_id": figure_id,
            "title": selected["title"], "primary": selected.get("primary", False),
            "prompt": selected.get("generation_prompt", selected["prompt"]), "caption": selected["caption"],
            "canvas": selected.get("generation_canvas"),
            "contract_uri": store.uri(artifact_id, "figures/" + figure_id + "/contract.txt"),
            "representation": selected.get("representation", "diagram"),
            "visual_contract": selected.get("visual_contract"),
            "render_brief": selected.get("render_brief"),
            "algorithm_flow_plan": selected.get("algorithm_flow_plan"),
            "diagram_design": selected.get("diagram_design"),
            "behavior_plan": selected.get("behavior_plan"),
            "figure_strategy": selected.get("figure_strategy"),
            "generation_steps": selected.get("generation_steps", []),
            "guide_role": None if selected.get("algorithm_flow_plan") else "composition_reference" if selected.get("render_brief") else "target_structure",
            "guide_optional": bool(selected.get("render_brief") or selected.get("algorithm_flow_plan")),
            "guide_path": str(guide_path) if guide_path.is_file() else None,
            "guide_uri": store.uri(artifact_id, "figures/" + figure_id + "/guide.png") if guide_path.is_file() else None,
            "review_uri": store.uri(artifact_id, "figures/" + figure_id + "/review.json"),
            "review_status": "not_reviewed",
        }

    @mcp.resource("silicon://artifacts/{artifact_id}/figures/{figure_id}/contract.txt", mime_type="text/plain")
    def contract_resource(artifact_id: str, figure_id: str) -> str:
        """Full figure contract: every rule the image must satisfy. Use it for spec-aware review, not as the generation prompt."""
        result = store.read(artifact_id, "compiled")
        return figure(result, figure_id)["prompt"]

    @mcp.resource("silicon://artifacts/{artifact_id}/figures/{figure_id}/review.json", mime_type="application/json")
    def review_resource(artifact_id: str, figure_id: str) -> dict[str, Any]:
        """Prepared review evidence; status remains not_reviewed until the host inspects a bitmap."""
        result = store.read(artifact_id, "compiled")
        return figure(result, figure_id).get("review_task", {})

    @mcp.resource("silicon://artifacts/{artifact_id}/exports/{filename}", mime_type="application/zip")
    def export_resource(artifact_id: str, filename: str) -> bytes:
        """Previously exported generation ZIP, addressed by the tool-returned URI."""
        result = store.read(artifact_id, "compiled")
        allowed = {"export.zip", *(item["id"] + ".export.zip" for item in result.get("figures", []))}
        if filename not in allowed:
            raise ValueError("Export does not exist. Call silicon_export first.")
        path = store.path(artifact_id, filename)
        if not path.is_file():
            raise ValueError("Export does not exist. Call silicon_export first.")
        return path.read_bytes()

    return mcp


def main() -> None:
    """Console script used only as the local MCP process entry."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
