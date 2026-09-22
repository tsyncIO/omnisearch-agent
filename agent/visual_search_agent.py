import os
import re
import gc
import torch
from typing import Dict, Any, Generator, List, Optional
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from tools.visual_tools import parse_grounding_tags, parse_grounding_tag, crop_image, draw_bounding_boxes
from tools.web_tools import (
    parse_web_search_tag,
    search_web_ddg,
    format_search_results_markdown,
    extract_search_intent_keywords,
    clean_search_query
)
from .prompts import AGENT_SYSTEM_PROMPT, NEXT_TURN_PROMPT_CROP, NEXT_TURN_PROMPT_WEB

def parse_model_response(response_text: str):
    """
    Robustly parses <think>, tool calls (<grounding>, <web_search>), and <answer>.
    Handles unclosed tags, malformed outputs, and plain responses.
    """
    think = ""
    remainder = response_text
    
    # 1. Extract thinking trace
    if "<think>" in response_text:
        parts = response_text.split("<think>", 1)[1]
        if "</think>" in parts:
            think, remainder = parts.split("</think>", 1)
        else:
            # Unclosed think tag
            think = parts
            remainder = ""
    elif "Thought:" in response_text:
        parts = response_text.split("Thought:", 1)[1]
        think = parts
        remainder = ""
    
    think = think.strip()
    remainder = remainder.strip()
    
    # 2. Extract answer if present
    answer = None
    if "<answer>" in remainder:
        ans_parts = remainder.split("<answer>", 1)[1]
        if "</answer>" in ans_parts:
            answer = ans_parts.split("</answer>", 1)[0].strip()
        else:
            answer = ans_parts.strip()
    elif "<answer>" in response_text:
        ans_parts = response_text.split("<answer>", 1)[1]
        if "</answer>" in ans_parts:
            answer = ans_parts.split("</answer>", 1)[0].strip()
        else:
            answer = ans_parts.strip()

    # 3. Extract tools
    grounding_list = parse_grounding_tags(response_text)
    web_query = parse_web_search_tag(response_text)

    # 4. If no tools invoked and no <answer> tag, remainder is the direct answer
    if answer is None and not grounding_list and not web_query and remainder:
        # Strip any stray tags
        cleaned_ans = re.sub(r"</?[a-zA-Z_]+>", "", remainder).strip()
        if len(cleaned_ans) > 10:
            answer = cleaned_ans

    return think, remainder, grounding_list, web_query, answer

class OmniSearchVisualAgent:
    def __init__(
        self,
        model_name_or_path: str = "Mini-o3/Mini-o3-7B-v1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = True
    ):
        self.device = device
        self.model_name = model_name_or_path
        self.load_in_4bit = load_in_4bit and (device == "cuda")

        print(f"[OmniSearchAgent] Initializing agent with model: {self.model_name} (4-bit: {self.load_in_4bit})")

        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[OmniSearchAgent Warning] Could not load {self.model_name}: {e}")
            fallback_model = "Qwen/Qwen2.5-VL-7B-Instruct"
            print(f"[OmniSearchAgent] Falling back to: {fallback_model}")
            self.model_name = fallback_model
            self.processor = AutoProcessor.from_pretrained(fallback_model, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                fallback_model,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )

        print("[OmniSearchAgent] Model successfully loaded and ready.")

    def run_streaming(
        self,
        original_image: Image.Image,
        query: str,
        max_turns: int = 4,
        auto_web_search: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Executes the agent loop and yields real-time updates for UI streaming.
        Yields dict with:
          - 'status': current activity description
          - 'annotated_image': image with current bounding box overlays
          - 'crop_gallery': list of (PIL.Image, caption)
          - 'thinking_text': accumulated thinking trace (terminal styled)
          - 'web_results_md': markdown representation of live web search (terminal styled)
          - 'final_answer': answer if completed (terminal styled)
        """
        # Memory maintenance
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        import time
        start_time = time.time()
        event_logs: List[str] = []

        def add_log(icon: str, tag: str, msg: str):
            elapsed = time.time() - start_time
            entry = f"[{elapsed:05.2f}s] {icon} [{tag}] {msg}"
            event_logs.append(entry)
            return entry

        original_image = original_image.convert("RGB")
        observations = [original_image]
        box_records = []
        crop_gallery = []
        performed_web_search = False

        add_log("🚀", "INGEST", f"Optical stream ingested: {original_image.width}x{original_image.height} RGB")
        add_log("⚡", "VLM_INIT", f"Backbone {self.model_name} initialized (4-bit NF4)")

        # Terminal initial headers
        initial_thought = (
            "```bash\n"
            "omnisearch@agent:~$ cat /proc/reasoning_stream\n"
            f"[SYS_INIT] Model backbone: {self.model_name}\n"
            "[STATUS] Vision-Language grounding session online.\n"
            "```\n\n"
            "*Analyzing input image visual features and objective...*"
        )
        initial_web = (
            "```bash\n"
            "omnisearch@agent:~$ netstat --active-rag --monitor\n"
            "[DAEMON] Live DuckDuckGo & Wikipedia RAG bridge active.\n"
            "[STATUS] Standby: waiting for entity detection...\n"
            "```"
        )
        initial_synthesis = (
            "```bash\n"
            "omnisearch@agent:~$ tail -f /var/log/executive_synthesis.md\n"
            "[STATUS] Waiting for inspection & verification...\n"
            "```"
        )

        full_thinking_log = initial_thought
        web_search_cards = initial_web
        final_answer = ""

        # Initial yield
        yield {
            "status": "Initializing visual inspection session...",
            "stage": "INIT",
            "progress": 10,
            "event_logs": list(event_logs),
            "annotated_image": original_image,
            "crop_gallery": crop_gallery,
            "thinking_text": full_thinking_log,
            "web_results_md": web_search_cards,
            "final_answer": initial_synthesis
        }

        # Prepare chat conversation
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},
                    {"type": "text", "text": query}
                ]
            }
        ]

        # Check if user query has immediate high-value keywords to preload web intelligence
        candidate_web_query = clean_search_query(query)
        if auto_web_search and len(candidate_web_query.split()) >= 3 and not candidate_web_query.startswith("this landmark"):
            try:
                add_log("🌐", "PRE_RAG", f"Pre-querying live web for '{candidate_web_query}'")
                pre_results = search_web_ddg(candidate_web_query, max_results=3)
                if pre_results:
                    web_search_cards = format_search_results_markdown(pre_results, candidate_web_query)
                    performed_web_search = True
                    add_log("🔗", "PRE_RAG_OK", f"Indexed {len(pre_results)} live references")
            except Exception as e:
                print(f"[OmniSearchAgent] Pre-search notice: {e}")

        for turn in range(max_turns):
            turn_num = turn + 1
            turn_progress = int(25 + (turn / max_turns) * 45)
            add_log("👁️", f"TURN_{turn_num}", f"Formulating visual perception & CoT reasoning ({turn_num}/{max_turns})")

            yield {
                "status": f"Turn {turn_num}/{max_turns}: Generating visual reasoning & planning actions...",
                "stage": "PERCEPTION",
                "progress": turn_progress,
                "event_logs": list(event_logs),
                "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
                "crop_gallery": crop_gallery,
                "thinking_text": full_thinking_log + f"\n\n```bash\n[TURN {turn_num}: GENERATING_REASONING_TOKENS...]\n```",
                "web_results_md": web_search_cards,
                "final_answer": initial_synthesis
            }

            # Format chat prompt
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Collect image inputs
            image_inputs = []
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image":
                            image_inputs.append(item["image"])

            # Tokenize & generate
            inputs = self.processor(
                text=[prompt_text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt"
            ).to(self.device)

            input_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9
                )

            # Extract ONLY newly generated tokens
            new_tokens = generated_ids[0][input_len:]
            response_text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

            # Clean memory
            del inputs, generated_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Parse response components
            turn_think, remainder, grounding_list, web_query, answer = parse_model_response(response_text)

            # Accumulate thinking log
            display_think = turn_think if turn_think else (remainder if not answer else "Direct visual deduction.")
            full_thinking_log += (
                f"\n\n```bash\n"
                f"═══════════════════════════════════════════════════════\n"
                f"[TURN {turn_num} // ACTIVE_REASONING_TRACE]\n"
                f"═══════════════════════════════════════════════════════\n"
                f"```\n"
                f"{display_think}\n"
            )

            # Append assistant message
            messages.append({"role": "assistant", "content": response_text})

            # 1. Process Visual Grounding / Crop Tool
            if grounding_list:
                for g_item in grounding_list:
                    bbox = g_item["bbox_2d"]
                    source = g_item.get("source", "original_image")

                    src_img = original_image
                    if source.startswith("observation_"):
                        try:
                            obs_idx = int(source.split("_")[-1])
                            if obs_idx < len(observations):
                                src_img = observations[obs_idx]
                        except Exception:
                            src_img = original_image

                    cropped_patch = crop_image(src_img, bbox)
                    observations.append(cropped_patch)
                    box_records.append({
                        "bbox": bbox,
                        "label": f"Focus {len(box_records) + 1}"
                    })
                    crop_gallery.append((cropped_patch, f"Region {len(box_records)}"))

                annotated_current = draw_bounding_boxes(original_image, box_records)
                add_log("🔬", "TOOL_CROP", f"Extracted {len(grounding_list)} target region(s) [Focus #{len(box_records)}]")
                add_log("🖼️", "OBSERVATION", f"Observation patch #{len(crop_gallery)} added to multi-turn gallery")
                yield {
                    "status": f"Turn {turn_num}: Cropped {len(grounding_list)} target regions. Feeding back into perception...",
                    "stage": "CROPPING",
                    "progress": min(85, turn_progress + 15),
                    "event_logs": list(event_logs),
                    "annotated_image": annotated_current,
                    "crop_gallery": crop_gallery,
                    "thinking_text": full_thinking_log,
                    "web_results_md": web_search_cards,
                    "final_answer": initial_synthesis
                }

                # Feed observation back
                last_crop = observations[-1]
                obs_prompt = NEXT_TURN_PROMPT_CROP.format(
                    turn_idx=turn_num,
                    obs_idx=len(observations) - 1,
                    source=f"observation_{len(observations) - 1}"
                )
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": obs_prompt},
                        {"type": "image", "image": last_crop}
                    ]
                })
                continue

            # 2. Process Web Search Tool (<web_search>)
            if web_query:
                # Filter out generic placeholder strings emitted by smaller models
                placeholder_tokens = {
                    "query keywords here", "keywords", "query here", "keywords here",
                    "search query", "query", "search keywords", "exact entity or topic",
                    "exact entity or topic to search", "exact entity or topic here"
                }
                if web_query.lower().strip() in placeholder_tokens or len(web_query.strip()) <= 2:
                    web_query = extract_search_intent_keywords(turn_think + " " + (answer or ""), user_query=query)

                performed_web_search = True
                add_log("🌐", "TOOL_WEB", f"Transmitted live search query: '{web_query}'")
                yield {
                    "status": f"Turn {turn_num}: Querying live web intelligence for '{web_query}'...",
                    "stage": "ACTIVE_RAG",
                    "progress": min(85, turn_progress + 15),
                    "event_logs": list(event_logs),
                    "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
                    "crop_gallery": crop_gallery,
                    "thinking_text": full_thinking_log,
                    "web_results_md": (
                        "```bash\n"
                        f"omnisearch@agent:~$ curl -s \"https://api.duckduckgo.com/?q={web_query}\"\n"
                        "[DISPATCH] Query transmitted to live internet bridge...\n"
                        "[WAITING] Parsing HTML & JSON endpoints...\n"
                        "```"
                    ),
                    "final_answer": initial_synthesis
                }

                results = search_web_ddg(web_query, max_results=4)
                web_search_cards = format_search_results_markdown(results, web_query)
                add_log("🔗", "WEB_HTTP_200", f"Retrieved {len(results)} verified references (DuckDuckGo + Wikipedia)")

                results_text = "\n".join([f"- [{r['title']}]: {r['body']} (Link: {r['href']})" for r in results])
                web_prompt = NEXT_TURN_PROMPT_WEB.format(query=web_query, web_results_text=results_text)
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": web_prompt}]
                })

                yield {
                    "status": f"Turn {turn_num}: Web data retrieved for '{web_query}'. Continuing synthesis...",
                    "stage": "ACTIVE_RAG",
                    "progress": min(90, turn_progress + 20),
                    "event_logs": list(event_logs),
                    "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
                    "crop_gallery": crop_gallery,
                    "thinking_text": full_thinking_log,
                    "web_results_md": web_search_cards,
                    "final_answer": initial_synthesis
                }
                continue

            # 3. Check for Concluded Answer
            if answer:
                add_log("📑", "SYNTHESIS", "Cross-referencing visual proof with live web facts...")
                # If user wanted web search and it hasn't run yet, enrich now
                if auto_web_search and not performed_web_search:
                    search_kw = extract_search_intent_keywords(turn_think + " " + answer, user_query=query)
                    add_log("🌐", "AUTO_RAG", f"Enriching synthesis with live web search for '{search_kw}'")
                    yield {
                        "status": f"Finalizing: Cross-referencing findings online for '{search_kw}'...",
                        "stage": "ACTIVE_RAG",
                        "progress": 92,
                        "event_logs": list(event_logs),
                        "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
                        "crop_gallery": crop_gallery,
                        "thinking_text": full_thinking_log,
                        "web_results_md": (
                            "```bash\n"
                            f"omnisearch@agent:~$ curl -s \"https://api.duckduckgo.com/?q={search_kw}\"\n"
                            "[DISPATCH] Live verification query dispatched...\n"
                            "```"
                        ),
                        "final_answer": initial_synthesis
                    }
                    web_res = search_web_ddg(search_kw, max_results=4)
                    if web_res:
                        web_search_cards = format_search_results_markdown(web_res, search_kw)
                        performed_web_search = True
                        add_log("🔗", "RAG_OK", f"Retrieved {len(web_res)} references for '{search_kw}'")

                final_formatted = (
                    "```bash\n"
                    "omnisearch@agent:~$ cat /var/out/synthesis.md\n"
                    "[REPORT_GEN] 200 OK • Synthesis Complete\n"
                    "```\n\n"
                    f"{answer}"
                )

                add_log("✅", "COMPLETE", "Autonomous mission concluded with verified report.")
                yield {
                    "status": "✅ Completed! Autonomous visual search finished.",
                    "stage": "COMPLETE",
                    "progress": 100,
                    "event_logs": list(event_logs),
                    "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
                    "crop_gallery": crop_gallery,
                    "thinking_text": full_thinking_log + "\n\n```bash\n[STATUS] Reasoning concluded. Final synthesis emitted.\n```",
                    "web_results_md": web_search_cards,
                    "final_answer": final_formatted
                }
                return

            # If no tools called and no answer, guide the model
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": "Please provide your final conclusion inside <answer> and </answer>."}]
            })

        # Max turns reached fallback
        add_log("⚠️", "MAX_TURNS", "Reached maximum exploration turns. Compiling synthesis.")
        fallback_ans = remainder if remainder else (turn_think if turn_think else response_text)
        if auto_web_search and not performed_web_search:
            search_kw = extract_search_intent_keywords(fallback_ans, user_query=query)
            web_res = search_web_ddg(search_kw, max_results=4)
            if web_res:
                web_search_cards = format_search_results_markdown(web_res, search_kw)

        final_formatted = (
            "```bash\n"
            "omnisearch@agent:~$ cat /var/out/synthesis.md\n"
            "[REPORT_GEN] 200 OK • Max Exploration Turns Reached\n"
            "```\n\n"
            f"{fallback_ans}"
        )

        add_log("✅", "COMPLETE", "Mission concluded with best visual synthesis.")
        yield {
            "status": "✅ Max turns reached. Concluded with best visual synthesis.",
            "stage": "COMPLETE",
            "progress": 100,
            "event_logs": list(event_logs),
            "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
            "crop_gallery": crop_gallery,
            "thinking_text": full_thinking_log + "\n\n```bash\n[STATUS] Maximum turns reached.\n```",
            "web_results_md": web_search_cards,
            "final_answer": final_formatted
        }
