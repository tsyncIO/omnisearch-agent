import os
import time
from PIL import Image
from agent.visual_search_agent import OmniSearchVisualAgent

def main():
    output_dir = "demo_outputs"
    os.makedirs(output_dir, exist_ok=True)

    test_image_path = "sample_images/fig_demo_crop.jpg"
    print(f"[Demo] Loading test image from: {test_image_path}")
    image = Image.open(test_image_path)
    print(f"[Demo] Image loaded: format={image.format}, size={image.size}")

    query = "Find and identify the target object or prominent item in this scene, zoom in to inspect details, and fetch real-time web info."
    print(f"[Demo] Query: {query}\n")

    print("[Demo] Initializing OmniSearchVisualAgent (Qwen2.5-VL-3B-Instruct in 4-bit NF4)...")
    agent = OmniSearchVisualAgent(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        load_in_4bit=True
    )

    print("\n" + "="*60)
    print("STARTING AUTONOMOUS MULTI-TURN SEARCH DEMO")
    print("="*60 + "\n")

    step_idx = 0
    final_state = None

    for state in agent.run_streaming(image, query, max_turns=3, auto_web_search=True):
        step_idx += 1
        print(f"\n--- [Update {step_idx}] Status: {state['status']} ---")
        if state["final_answer"]:
            final_state = state
            break

    if final_state:
        print("\n" + "="*60)
        print("DEMO RESULTS")
        print("="*60)
        print("\n[AGENT_THINKING]:")
        print(final_state["thinking_text"])

        print("\n[WEB_RESULTS]:")
        print(final_state["web_results_md"])

        print("\n[FINAL_SYNTHESIS]:")
        print(final_state["final_answer"])

        # Save outputs
        annotated_path = os.path.join(output_dir, "annotated_grounding.jpg")
        final_state["annotated_image"].save(annotated_path)
        print(f"\nSaved annotated canvas to: {annotated_path}")

        for i, (crop_img, label) in enumerate(final_state["crop_gallery"]):
            crop_path = os.path.join(output_dir, f"crop_turn_{i+1}.jpg")
            crop_img.save(crop_path)
            print(f"Saved {label} to: {crop_path}")

        print("\nDemo execution completed successfully.")

if __name__ == "__main__":
    main()
