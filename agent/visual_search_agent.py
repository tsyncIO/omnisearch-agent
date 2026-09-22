import os
import re
import torch
from typing import Dict, Any, Generator, List
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from tools.visual_tools import parse_grounding_tag, crop_image, draw_bounding_boxes
from tools.web_tools import parse_web_search_tag, search_web_ddg, format_search_results_markdown
from .prompts import AGENT_SYSTEM_PROMPT, NEXT_TURN_PROMPT_CROP, NEXT_TURN_PROMPT_WEB

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

        # Fallback to base model if weights aren't yet downloaded or specified
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
        max_turns: int = 5,
        auto_web_search: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Executes the agent loop and yields real-time updates for UI streaming.
        Yields dict with:
          - 'status': current activity description
          - 'annotated_image': image with current bounding box overlays
          - 'crop_gallery': list of (PIL.Image, caption)
          - 'thinking_text': accumulated thinking trace
          - 'web_results_md': markdown representation of live web search
          - 'final_answer': answer if completed
        """
        # Ensure image is RGB
        original_image = original_image.convert("RGB")
        observations = [original_image]
        box_records = []
        crop_gallery = []
        full_thinking_log = ""
        web_search_cards = ""
        performed_web_search = False

        # Prepare initial messages
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

        yield {
            "status": "Thinking & inspecting visual scene...",
            "annotated_image": original_image,
            "crop_gallery": crop_gallery,
            "thinking_text": "Agent initialized. Analyzing input image...",
            "web_results_md": "*No web search executed yet.*",
            "final_answer": ""
        }

        for turn in range(max_turns):
            yield {
                "status": f"Turn {turn + 1}: Generating reasoning & tool actions...",
                "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
                "crop_gallery": crop_gallery,
                "thinking_text": full_thinking_log + f"\n\n*[Turn {turn + 1} Thinking...]*",
                "web_results_md": web_search_cards or "*No web search executed yet.*",
                "final_answer": ""
            }

            # Format chat prompt
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Collect all image inputs from conversation history
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

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.9
                )

            # Extract generated response tokens
            new_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
            response_text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

            # Parse think tag
            think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
            turn_thinking = think_match.group(1).strip() if think_match else response_text
            full_thinking_log += f"\n\n### 🧠 Turn {turn + 1} Reasoning:\n{turn_thinking}"

            # Append assistant response to chat history
            messages.append({"role": "assistant", "content": response_text})

            # Check if final answer is provided
            if "<answer>" in response_text:
                answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
                final_answer = answer_match.group(1).strip() if answer_match else response_text

                # Optional: If user requested web info and no web search happened yet, enrich with web search
                if auto_web_search and not performed_web_search:
                    search_query = query.replace("identify", "").replace("search", "").strip()
                    if len(search_query) > 3:
                        web_results = search_web_ddg(search_query, max_results=3)
                        if web_results:
                            web_search_cards = format_search_results_markdown(web_results, search_query)

                yield {
                    "status": "Completed! Final answer ready.",
                    "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
                    "crop_gallery": crop_gallery,
                    "thinking_text": full_thinking_log,
                    "web_results_md": web_search_cards or "*No web search required for this query.*",
                    "final_answer": final_answer
                }
                return

            # Check for visual tool call (<grounding>)
            grounding_data = parse_grounding_tag(response_text)
            if grounding_data:
                bbox = grounding_data["bbox_2d"]
                source = grounding_data["source"]

                # Resolve source image
                src_img = original_image
                if source.startswith("observation_"):
                    try:
                        obs_idx = int(source.split("_")[-1])
                        if obs_idx < len(observations):
                            src_img = observations[obs_idx]
                    except Exception:
                        src_img = original_image

                # Execute crop tool
                cropped_patch = crop_image(src_img, bbox, relative=True)
                observations.append(cropped_patch)
                crop_label = f"Turn {turn + 1} Inspection (from {source})"
                crop_gallery.append((cropped_patch, crop_label))

                # Add bounding box to canvas
                box_records.append({
                    "bbox": bbox,
                    "label": f"Turn {turn + 1} Focus"
                })

                # Yield updated view with crop
                yield {
                    "status": f"Turn {turn + 1}: Cropped region [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]. Feeding back into reasoning...",
                    "annotated_image": draw_bounding_boxes(original_image, box_records),
                    "crop_gallery": crop_gallery,
                    "thinking_text": full_thinking_log,
                    "web_results_md": web_search_cards or "*No web search executed yet.*",
                    "final_answer": ""
                }

                # Feed observation back to conversation
                obs_prompt = NEXT_TURN_PROMPT_CROP.format(
                    turn_idx=turn + 1,
                    obs_idx=len(observations) - 1,
                    source=f"observation_{len(observations) - 1}"
                )
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": obs_prompt},
                        {"type": "image", "image": cropped_patch}
                    ]
                })
                continue

            # Check for web search tool call (<web_search>)
            web_query = parse_web_search_tag(response_text)
            if web_query:
                performed_web_search = True
                yield {
                    "status": f"Querying live web for: '{web_query}'...",
                    "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
                    "crop_gallery": crop_gallery,
                    "thinking_text": full_thinking_log,
                    "web_results_md": f"⏳ *Fetching live web search results for `{web_query}`...*",
                    "final_answer": ""
                }

                results = search_web_ddg(web_query, max_results=4)
                web_search_cards = format_search_results_markdown(results, web_query)

                # Format text results for model
                results_text = "\n".join([f"- [{r['title']}]: {r['body']} (Link: {r['href']})" for r in results])
                web_prompt = NEXT_TURN_PROMPT_WEB.format(query=web_query, web_results_text=results_text)

                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": web_prompt}]
                })

                yield {
                    "status": f"Web results received for '{web_query}'. Synthesizing answer...",
                    "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
                    "crop_gallery": crop_gallery,
                    "thinking_text": full_thinking_log,
                    "web_results_md": web_search_cards,
                    "final_answer": ""
                }
                continue

            # If no recognized tool tag and no answer tag, prompt the model to finalize
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": "Please provide your final conclusion inside <answer> and </answer>."}]
            })

        # Max turns reached
        yield {
            "status": "Max turns reached. Outputting best synthesis.",
            "annotated_image": draw_bounding_boxes(original_image, box_records) if box_records else original_image,
            "crop_gallery": crop_gallery,
            "thinking_text": full_thinking_log,
            "web_results_md": web_search_cards or "*No web search executed.*",
            "final_answer": response_text
        }
