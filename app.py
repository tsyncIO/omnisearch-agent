import os
import argparse
import gradio as gr
from PIL import Image
from agent.visual_search_agent import OmniSearchVisualAgent

# Lazy agent loader so Gradio boots fast
agent_instance = None

def get_agent(model_name: str = "Mini-o3/Mini-o3-7B-v1", load_in_4bit: bool = True):
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
            "⚠️ Please upload an image first.",
            None,
            [],
            "### ⚠️ No Image Provided\nPlease upload an image to begin visual search.",
            "",
            ""
        )
        return

    if not query.strip():
        query = "Identify the key item or text in this image, zoom in if needed, and find information about it."

    yield (
        "🚀 Loading agent & weights (first run may download checkpoints)...",
        input_image,
        [],
        "*Initializing visual search agent...*",
        "",
        ""
    )

    agent = get_agent(model_name=model_choice, load_in_4bit=True)

    for state in agent.run_streaming(
        original_image=input_image,
        query=query,
        max_turns=max_turns,
        auto_web_search=auto_web
    ):
        status = f"⚡ **Status**: {state['status']}"
        annotated_img = state["annotated_image"]
        gallery = state["crop_gallery"]
        final_answer = state["final_answer"]
        web_md = state["web_results_md"]
        thinking_md = state["thinking_text"]

        answer_display = (
            f"### 🎯 Agent Final Conclusion:\n\n{final_answer}"
            if final_answer
            else "*Agent is actively reasoning and investigating...*"
        )

        yield (
            status,
            annotated_img,
            gallery,
            answer_display,
            web_md,
            thinking_md
        )

# Gradio Dashboard Layout
custom_css = """
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 24px;
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.main-header h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
}
.main-header p {
    margin-top: 8px;
    color: #94a3b8;
    font-size: 1.05rem;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan"), css=custom_css, title="OmniSearch Agent") as demo:
    gr.HTML("""
    <div class="main-header">
        <h1>🦅 OmniSearch Agent</h1>
        <p>Autonomous Multi-Turn Visual Discovery & Live Web Intelligence — Powered by Mini-o3 & Qwen2.5-VL</p>
    </div>
    """)

    with gr.Row():
        # Left: Controls & Inputs
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📸 Input Image (High-Resolution)", height=340)
            query_input = gr.Textbox(
                label="💬 Agent Task / Query",
                placeholder="e.g., Identify this specific watch model, zoom in on the dials and markings, and search the web for current market pricing.",
                lines=3
            )

            with gr.Accordion("⚙️ Agent Settings", open=False):
                model_selector = gr.Dropdown(
                    choices=["Mini-o3/Mini-o3-7B-v1", "Qwen/Qwen2.5-VL-7B-Instruct"],
                    value="Mini-o3/Mini-o3-7B-v1",
                    label="Backbone Model"
                )
                max_turns_slider = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Max Exploration Turns")
                auto_web_checkbox = gr.Checkbox(value=True, label="Auto-enrich with Live Web Search")

            with gr.Row():
                submit_btn = gr.Button("🔍 Run Autonomous Search", variant="primary", scale=2)
                clear_btn = gr.ClearButton([image_input, query_input], value="Clear", scale=1)

            gr.Markdown("""
            ---
            **Key Features**:
            - 🔬 **Active Visual Zooming**: Inspects regions across multiple turns.
            - 🌐 **Live Web Intelligence**: Automatically fetches web sources via DuckDuckGo.
            - ⚡ **Optimized for 16GB VRAM**: 4-bit NF4 quantization runs easily on a single GPU.
            """)

        # Right: Outputs & Interactive Displays
        with gr.Column(scale=2):
            status_output = gr.Markdown("⚡ **Status**: Ready for input.")

            with gr.Row():
                canvas_output = gr.Image(label="🎯 Visual Grounding & Focus Bounding Boxes", height=320)
                crops_output = gr.Gallery(label="🔎 Multi-Turn Zoomed Inspection Crops", columns=2, height=320)

            with gr.Tabs():
                with gr.Tab("📋 Final Synthesis"):
                    answer_output = gr.Markdown("*Your synthesized report will appear here once the agent concludes.*")

                with gr.Tab("🌐 Live Web Sources"):
                    web_output = gr.Markdown("*Live web search citations and links will appear here.*")

                with gr.Tab("🧠 Step-by-Step Thinking (<think>)"):
                    thinking_output = gr.Markdown("*Agent internal reasoning logs will stream here.*")

    submit_btn.click(
        fn=run_agent_interface,
        inputs=[image_input, query_input, max_turns_slider, auto_web_checkbox, model_selector],
        outputs=[status_output, canvas_output, crops_output, answer_output, web_output, thinking_output]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    args = parser.parse_args()

    print(f"[OmniSearch Agent] Starting Gradio UI on port {args.port}...")
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
