import os
import argparse
import gradio as gr
from PIL import Image
from agent.visual_search_agent import OmniSearchVisualAgent

# Global lazy agent instance
agent_instance = None

def get_agent(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", load_in_4bit: bool = True):
    global agent_instance
    if agent_instance is None or agent_instance.model_name != model_name:
        agent_instance = OmniSearchVisualAgent(
            model_name_or_path=model_name,
            load_in_4bit=load_in_4bit
        )
    return agent_instance

def run_agent_interface(
    input_image: Image.Image,
    query: str,
    max_turns: int,
    auto_web: bool,
    model_choice: str
):
    if input_image is None:
        yield (
            "⚠️ **Please upload an image first.**",
            None,
            None,
            [],
            [],
            "### ⚠️ No Image Provided\nPlease upload an image to begin visual search.",
            "*No web search executed.*",
            "*No reasoning log.*"
        )
        return

    if not query.strip():
        query = "Identify the key item or text in this image, zoom in if needed, and find information about it."

    yield (
        "🚀 **Initializing Agent & Loading Weights...**",
        input_image,
        input_image,
        [],
        [],
        "*Agent is starting visual inspection...*",
        "*Fetching web sources...*",
        "*Initializing chain-of-thought...*"
    )

    agent = get_agent(model_name=model_choice, load_in_4bit=True)

    for state in agent.run_streaming(
        original_image=input_image,
        query=query,
        max_turns=max_turns,
        auto_web_search=auto_web
    ):
        status = f"⚡ **Activity**: {state['status']}"
        annotated_img = state["annotated_image"]
        gallery = state["crop_gallery"]
        final_answer = state["final_answer"]
        web_md = state["web_results_md"]
        thinking_md = state["thinking_text"]

        answer_display = (
            f"### 🎯 Agent Synthesized Report\n\n{final_answer}"
            if final_answer
            else "*Agent is actively reasoning and investigating...*"
        )

        yield (
            status,
            annotated_img,      # Canvas Full View
            annotated_img,      # Canvas Side-by-Side
            gallery,            # Gallery Lightbox View
            gallery,            # Gallery Side-by-Side
            answer_display,
            web_md,
            thinking_md
        )

# Modern, polished CSS styling
custom_css = """
/* App Header */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
}
.app-header-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #f8fafc;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.app-header-desc {
    color: #94a3b8;
    font-size: 1rem;
    margin: 8px 0 0 0;
}
.badge-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 10px;
}

/* Card Containers */
.card-container {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Status Pill */
.status-pill {
    background: linear-gradient(90deg, rgba(6, 182, 212, 0.12) 0%, rgba(59, 130, 246, 0.12) 100%);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 500;
    color: #0284c7;
    margin-bottom: 12px;
}

/* Quick prompt chips */
.prompt-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 8px 0 12px 0;
}
.prompt-chip-btn {
    font-size: 0.8rem !important;
    padding: 4px 10px !important;
    border-radius: 8px !important;
}

/* Gallery & Image Containers */
.visual-studio-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    overflow: hidden;
}
"""

with gr.Blocks(title="OmniSearch Studio") as demo:
    gr.HTML("""
    <div class="app-header">
        <h1 class="app-header-title">🦅 OmniSearch Studio</h1>
        <p class="app-header-desc">Autonomous Multi-Turn Visual Discovery & Live Web Intelligence — Powered by Mini-o3 & Qwen2.5-VL</p>
        <div>
            <span class="badge-chip">● NVIDIA RTX A4000 16GB (4-bit NF4)</span>
            <span class="badge-chip" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(52, 211, 153, 0.3);">⚡ Live Active RAG</span>
            <span class="badge-chip" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border-color: rgba(192, 132, 252, 0.3);">🔬 Multi-Turn Zoom</span>
        </div>
    </div>
    """)

    with gr.Row():
        # Left: Controls, Upload & Settings
        with gr.Column(scale=4):
            with gr.Group():
                image_input = gr.Image(
                    type="pil",
                    label="📸 Input Image (High-Resolution)",
                    sources=["upload", "clipboard"],
                    height=360
                )

                query_input = gr.Textbox(
                    label="💬 Agent Task & Search Objective",
                    placeholder="e.g., Identify the objects, zoom in to read small text/labels, and search the web for details...",
                    lines=3
                )

                # Quick-fill Prompt Pills
                gr.Markdown("<small style='color: var(--body-text-color-subdued);'>⚡ Quick Prompts:</small>")
                with gr.Row():
                    p1_btn = gr.Button("🇩🇪 Sort German Waste Bins", size="sm", variant="secondary")
                    p2_btn = gr.Button("🎬 Identify Venue & Brand", size="sm", variant="secondary")
                    p3_btn = gr.Button("🏛️ Landmark & History", size="sm", variant="secondary")
                    p4_btn = gr.Button("🔬 Deep Inspect Details", size="sm", variant="secondary")

            with gr.Accordion("⚙️ Agent & Model Settings", open=False):
                model_selector = gr.Dropdown(
                    choices=[
                        "Qwen/Qwen2.5-VL-3B-Instruct",
                        "Mini-o3/Mini-o3-7B-v1",
                        "Qwen/Qwen2.5-VL-7B-Instruct"
                    ],
                    value="Qwen/Qwen2.5-VL-3B-Instruct",
                    label="Backbone Vision Model"
                )
                max_turns_slider = gr.Slider(
                    minimum=1, maximum=6, value=3, step=1,
                    label="Max Exploration Turns"
                )
                auto_web_checkbox = gr.Checkbox(
                    value=True,
                    label="Auto-enrich with Live Web Search (DuckDuckGo)"
                )

            with gr.Row():
                submit_btn = gr.Button("🔍 Run Autonomous Search", variant="primary", scale=3, size="lg")
                clear_btn = gr.ClearButton([image_input, query_input], value="↺ Reset", scale=1, size="lg")

            # Built-in Examples
            gr.Markdown("### 📂 Preloaded Test Cases")
            gr.Examples(
                examples=[
                    [
                        "sample_images/landmark_colosseum.jpg",
                        "Identify this landmark, inspect its architectural features, and search the web for its history, original name, and significance.",
                        3,
                        True,
                        "Qwen/Qwen2.5-VL-3B-Instruct"
                    ],
                    [
                        "sample_images/test.jpg",
                        "Identify each waste bin by its color and German label, read the city name mentioned on the bins, and search the web to explain the waste separation system used here.",
                        3,
                        True,
                        "Qwen/Qwen2.5-VL-3B-Instruct"
                    ],
                    [
                        "sample_images/fig_demo_crop.jpg",
                        "Identify the brand and text in this image, zoom in if needed, and search the web to explain what kind of venue or company this is.",
                        3,
                        True,
                        "Qwen/Qwen2.5-VL-3B-Instruct"
                    ]
                ],
                inputs=[image_input, query_input, max_turns_slider, auto_web_checkbox, model_selector]
            )

        # Right: Visual Studio & Research Workspace
        with gr.Column(scale=6):
            status_output = gr.Markdown(
                "⚡ **Activity**: Ready. Select an example or upload an image and click **Run Autonomous Search**.",
                elem_classes=["status-pill"]
            )

            # Visual Inspection Studio (Full tabs with natural fit and lightbox preview)
            with gr.Tabs():
                with gr.Tab("🎯 Grounding Canvas (Full View)"):
                    canvas_output_full = gr.Image(
                        label="Visual Grounding & Detected Bounding Boxes",
                        interactive=False,
                        height=480
                    )

                with gr.Tab("🔎 Zoomed Inspection Crops (Click to Enlarge)"):
                    crops_output_gallery = gr.Gallery(
                        label="Multi-Turn Zoomed Patches",
                        columns=[2, 3],
                        rows=[1, 2],
                        height=480,
                        preview=True,
                        allow_preview=True,
                        object_fit="contain",
                        show_label=True
                    )

                with gr.Tab("🔄 Side-by-Side View"):
                    with gr.Row():
                        canvas_output_split = gr.Image(
                            label="Focus Canvas",
                            interactive=False,
                            height=380
                        )
                        crops_output_split = gr.Gallery(
                            label="Inspection Crops",
                            columns=1,
                            height=380,
                            preview=True,
                            allow_preview=True,
                            object_fit="contain"
                        )

            # Intelligence & Report Studio
            with gr.Tabs():
                with gr.Tab("📋 Final Synthesis"):
                    answer_output = gr.Markdown(
                        "*Your synthesized report will appear here once the agent concludes its investigation.*"
                    )

                with gr.Tab("🌐 Live Web Sources"):
                    web_output = gr.Markdown(
                        "*Live web search citations and links will appear here.*"
                    )

                with gr.Tab("🧠 Deep Reasoning Trace (<think>)"):
                    thinking_output = gr.Markdown(
                        "*Agent internal chain-of-thought tokens will stream here.*"
                    )

    # Quick prompt button click events
    p1_btn.click(
        fn=lambda: "Identify each waste bin by its color and German label, read the city name mentioned on the bins, and search the web to explain the waste separation system used here.",
        outputs=[query_input]
    )
    p2_btn.click(
        fn=lambda: "Identify the brand and text in this image, zoom in if needed, and search the web to explain what kind of venue or company this is.",
        outputs=[query_input]
    )
    p3_btn.click(
        fn=lambda: "Identify this landmark, inspect its architectural features, and search the web for its history, original name, and significance.",
        outputs=[query_input]
    )
    p4_btn.click(
        fn=lambda: "Examine this image with deep inspection, zoom in on any fine details, numbers or labels, and explain everything visible.",
        outputs=[query_input]
    )

    # Submit event
    submit_btn.click(
        fn=run_agent_interface,
        inputs=[
            image_input,
            query_input,
            max_turns_slider,
            auto_web_checkbox,
            model_selector
        ],
        outputs=[
            status_output,
            canvas_output_full,
            canvas_output_split,
            crops_output_gallery,
            crops_output_split,
            answer_output,
            web_output,
            thinking_output
        ]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    args = parser.parse_args()

    print(f"[OmniSearch Agent] Starting Gradio UI on port {args.port} (share={args.share})...")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(
            primary_hue="cyan",
            font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"]
        ),
        css=custom_css
    )
