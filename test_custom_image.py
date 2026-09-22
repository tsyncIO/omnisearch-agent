import os
from PIL import Image
from agent.visual_search_agent import OmniSearchVisualAgent

def main():
    image_path = "/home/tmdt-admin/tanvir/omnisearch-agent/sample_images/test.jpg"
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")

    query = (
        "Identify each waste bin in this image by its color and German label, "
        "read the city name mentioned on the bins, and search the web to explain "
        "the waste separation system used here."
    )
    print(f"User Query: {query}\n")

    print("Initializing OmniSearchVisualAgent (Qwen2.5-VL-3B-Instruct, 4-bit)...")
    agent = OmniSearchVisualAgent(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        load_in_4bit=True
    )

    output_dir = "demo_outputs_test"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("RUNNING VISUAL SEARCH + WEB RAG ON test.jpg")
    print("="*60 + "\n")

    final_state = None
    for state in agent.run_streaming(image, query, max_turns=3, auto_web_search=True):
        print(f"Status: {state['status']}")
        if state["final_answer"]:
            final_state = state
            break

    if final_state:
        print("\n" + "="*60)
        print("AGENT REASONING:")
        print(final_state["thinking_text"])
        print("="*60)

        print("\n" + "="*60)
        print("LIVE WEB SEARCH RESULTS:")
        print(final_state["web_results_md"])
        print("="*60)

        print("\n" + "="*60)
        print("FINAL SYNTHESIZED REPORT:")
        print(final_state["final_answer"])
        print("="*60)

        # Save annotated image
        annotated_path = os.path.join(output_dir, "annotated_bins.jpg")
        final_state["annotated_image"].save(annotated_path)
        print(f"\nSaved annotated image to: {annotated_path}")

        for i, (crop, label) in enumerate(final_state["crop_gallery"]):
            p = os.path.join(output_dir, f"crop_{i+1}.jpg")
            crop.save(p)
            print(f"Saved {label} to: {p}")

if __name__ == "__main__":
    main()
