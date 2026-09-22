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
        init_term_reasoning = """```bash
omnisearch@agent:~$ cat /proc/reasoning_stream
[SYS_WARN] No image input detected in stdin.
[STATUS] Please upload an image or click a preloaded test case.
```"""
        init_term_web = """```bash
omnisearch@agent:~$ netstat --active-rag --monitor
[SYS_HALT] Waiting for visual stream before dispatching queries.
```"""
        init_term_synthesis = """```bash
omnisearch@agent:~$ cat /var/out/synthesis.md
[STATUS] Idle. Awaiting visual inspection session.
```"""
        yield (
            "⚠️ **Please upload an image first.**",
            None,
            None,
            [],
            [],
            init_term_reasoning,
            init_term_web,
            init_term_synthesis
        )
        return

    if not query.strip():
        query = "Identify the key item or landmark in this image, zoom in if needed, and search the web for details."

    # Immediate responsive yield
    boot_reasoning = """```bash
omnisearch@agent:~$ cat /proc/reasoning_stream
[SYS_INIT] Vision-Language grounding session starting...
[CUDA] Initializing NVIDIA RTX A4000 16GB (4-bit NF4)
[STATUS] Tokenizing image and formulating initial chain-of-thought...
```"""
    boot_web = """```bash
omnisearch@agent:~$ netstat --active-rag --monitor
[DAEMON] Live DuckDuckGo & Wikipedia RAG bridge active.
[STATUS] Standby: monitoring entity recognition stream...
```"""
    boot_synthesis = """```bash
omnisearch@agent:~$ tail -f /var/log/executive_synthesis.md
[REPORT_GEN] Executive synthesis pipeline online.
[STATUS] Awaiting visual inspection & web verification...
```"""

    yield (
        "🚀 **Initializing Agent & Loading Weights...**",
        input_image,
        input_image,
        [],
        [],
        boot_reasoning,
        boot_web,
        boot_synthesis
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

        yield (
            status,
            annotated_img,      # Canvas Full View
            annotated_img,      # Canvas Side-by-Side
            gallery,            # Gallery Lightbox View
            gallery,            # Gallery Side-by-Side
            thinking_md,        # Terminal 1: Deep Reasoning Trace
            web_md,             # Terminal 2: Live Web Intelligence
            final_answer        # Terminal 3: Final Synthesis
        )

# Modern, polished Cyber/Hacker CSS styling with authentic Terminal windows
custom_css = """
/* App Header */
.app-header {
    background: linear-gradient(135deg, #0b0f19 0%, #161e2e 50%, #0b0f19 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
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
    font-size: 0.98rem;
    margin: 6px 0 0 0;
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
    margin-top: 8px;
}

/* Status Pill */
.status-pill {
    background: linear-gradient(90deg, rgba(6, 182, 212, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
    border: 1px solid rgba(6, 182, 212, 0.35);
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 12px;
}

/* Section Header for Parallel Terminals */
.terminal-section-banner {
    background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 18px;
    margin: 18px 0 12px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.terminal-section-title {
    font-family: ui-monospace, 'SFMono-Regular', 'JetBrains Mono', Menlo, Consolas, monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #58a6ff;
    display: flex;
    align-items: center;
    gap: 8px;
}
.terminal-section-subtitle {
    font-size: 0.82rem;
    color: #8b949e;
}

/* Authentic Terminal Window Container */
.terminal-container {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45) !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Terminal Title Bar */
.terminal-top-bar {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 9px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    user-select: none;
}
.terminal-dots {
    display: flex;
    align-items: center;
    gap: 6px;
}
.terminal-dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    display: inline-block;
}
.terminal-dot.dot-red { background: #ff5f56; }
.terminal-dot.dot-yellow { background: #ffbd2e; }
.terminal-dot.dot-green { background: #27c93f; }

.terminal-title-text {
    font-family: ui-monospace, 'SFMono-Regular', 'JetBrains Mono', Menlo, monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #c9d1d9;
    text-transform: uppercase;
}

.terminal-badge {
    font-family: ui-monospace, 'SFMono-Regular', 'JetBrains Mono', Menlo, monospace;
    font-size: 0.72rem;
    padding: 2px 7px;
    border-radius: 4px;
    font-weight: 700;
}
.badge-reasoning { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
.badge-web { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
.badge-synthesis { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }

/* Terminal Scrollable Body */
.terminal-window-body {
    background: #0d1117 !important;
    height: 480px !important;
    max-height: 480px !important;
    overflow-y: auto !important;
    padding: 14px 16px !important;
    font-family: ui-monospace, 'SFMono-Regular', 'JetBrains Mono', Menlo, Consolas, monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
    color: #c9d1d9 !important;
}

/* Custom Scrollbar for Terminal */
.terminal-window-body::-webkit-scrollbar {
    width: 6px;
}
.terminal-window-body::-webkit-scrollbar-track {
    background: #0d1117;
}
.terminal-window-body::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 3px;
}
.terminal-window-body::-webkit-scrollbar-thumb:hover {
    background: #8b949e;
}

/* Terminal Content Formatting */
.terminal-window-body pre, .terminal-window-body code {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #58a6ff !important;
    border-radius: 6px !important;
    font-family: inherit !important;
}
.terminal-window-body a {
    color: #58a6ff !important;
    text-decoration: underline !important;
}
.terminal-window-body strong {
    color: #f0f6fc !important;
}
.terminal-window-body blockquote {
    border-left: 3px solid #38bdf8 !important;
    margin: 8px 0 !important;
    padding: 4px 12px !important;
    background: rgba(56, 189, 248, 0.05) !important;
    color: #8b949e !important;
}
"""

with gr.Blocks(title="OmniSearch Studio") as demo:
    gr.HTML("""
    <div class="app-header">
        <h1 class="app-header-title">🦅 OmniSearch Studio</h1>
        <p class="app-header-desc">Autonomous Multi-Turn Visual Discovery & Live Web Intelligence — Powered by Mini-o3 & Qwen2.5-VL</p>
        <div>
            <span class="badge-chip">● NVIDIA RTX A4000 16GB (4-bit NF4)</span>
            <span class="badge-chip" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(52, 211, 153, 0.3);">⚡ Live Active RAG (DDG + Wikipedia)</span>
            <span class="badge-chip" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border-color: rgba(192, 132, 252, 0.3);">🔬 Multi-Turn Zoom Grounding</span>
        </div>
    </div>
    """)

    # Top Section: Controls (Left) & Visual Grounding Canvas (Right)
    with gr.Row():
        # Left: Controls, Upload & Settings
        with gr.Column(scale=5):
            with gr.Group():
                image_input = gr.Image(
                    type="pil",
                    label="📸 Input Image (High-Resolution)",
                    sources=["upload", "clipboard"],
                    height=340
                )

                query_input = gr.Textbox(
                    label="💬 Agent Task & Search Objective",
                    placeholder="e.g., Identify the objects, zoom in to read small text/labels, and search the web for details...",
                    lines=3
                )

                # Quick-fill Prompt Pills
                gr.Markdown("<small style='color: var(--body-text-color-subdued); font-weight:600;'>⚡ Quick Prompts:</small>")
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
                    minimum=1, maximum=5, value=3, step=1,
                    label="Max Exploration Turns"
                )
                auto_web_checkbox = gr.Checkbox(
                    value=True,
                    label="Auto-enrich with Live Web Search (DuckDuckGo + Wikipedia)"
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

        # Right: Visual Inspection Studio
        with gr.Column(scale=7):
            status_output = gr.Markdown(
                "⚡ **Activity**: Ready. Select an example or upload an image and click **Run Autonomous Search**.",
                elem_classes=["status-pill"]
            )

            # Visual Inspection Studio Tabs
            with gr.Tabs():
                with gr.Tab("🎯 Grounding Canvas (Full View)"):
                    canvas_output_full = gr.Image(
                        label="Visual Grounding & Detected Bounding Boxes",
                        interactive=False,
                        height=460
                    )

                with gr.Tab("🔎 Zoomed Inspection Crops (Click to Enlarge)"):
                    crops_output_gallery = gr.Gallery(
                        label="Multi-Turn Zoomed Patches",
                        columns=[2, 3],
                        rows=[1, 2],
                        height=460,
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

    # Section Banner for Parallel Side-by-Side Terminals
    gr.HTML("""
    <div class="terminal-section-banner">
        <div class="terminal-section-title">
            <span>⚡ AUTONOMOUS AGENT PARALLEL TERMINALS</span>
        </div>
        <div class="terminal-section-subtitle">
            Concurrent live streaming: Deep Reasoning Trace (<think>) ⏐ Live Web Sources (Active RAG) ⏐ Final Synthesis
        </div>
    </div>
    """)

    # Bottom Section: 3 Authentic Parallel Side-by-Side Terminals
    with gr.Row(elem_classes=["terminals-row"]):
        # Terminal 1: Deep Reasoning Trace (<think>)
        with gr.Column(scale=1, min_width=320):
            with gr.Group(elem_classes=["terminal-container"]):
                gr.HTML("""
                <div class="terminal-top-bar">
                    <div class="terminal-dots">
                        <span class="terminal-dot dot-red"></span>
                        <span class="terminal-dot dot-yellow"></span>
                        <span class="terminal-dot dot-green"></span>
                    </div>
                    <span class="terminal-title-text">TERMINAL 01 // REASONING_TRACE</span>
                    <span class="terminal-badge badge-reasoning">&lt;THINK&gt;</span>
                </div>
                """)
                thinking_output = gr.Markdown(
                    """```bash
omnisearch@agent:~$ cat /proc/reasoning_stream
[SYS_INIT] Qwen2.5-VL CoT streaming daemon active.
[STATUS] Awaiting visual input & task prompt...
```""",
                    elem_classes=["terminal-window-body"]
                )

        # Terminal 2: Live Web Sources (Active RAG)
        with gr.Column(scale=1, min_width=320):
            with gr.Group(elem_classes=["terminal-container"]):
                gr.HTML("""
                <div class="terminal-top-bar">
                    <div class="terminal-dots">
                        <span class="terminal-dot dot-red"></span>
                        <span class="terminal-dot dot-yellow"></span>
                        <span class="terminal-dot dot-green"></span>
                    </div>
                    <span class="terminal-title-text">TERMINAL 02 // LIVE_WEB_INTEL</span>
                    <span class="terminal-badge badge-web">ACTIVE_RAG</span>
                </div>
                """)
                web_output = gr.Markdown(
                    """```bash
omnisearch@agent:~$ netstat --active-rag --monitor
[DAEMON] DuckDuckGo & Wikipedia search bridges online.
[STATUS] Standby: monitoring entity recognition stream...
```""",
                    elem_classes=["terminal-window-body"]
                )

        # Terminal 3: Final Synthesis
        with gr.Column(scale=1, min_width=320):
            with gr.Group(elem_classes=["terminal-container"]):
                gr.HTML("""
                <div class="terminal-top-bar">
                    <div class="terminal-dots">
                        <span class="terminal-dot dot-red"></span>
                        <span class="terminal-dot dot-yellow"></span>
                        <span class="terminal-dot dot-green"></span>
                    </div>
                    <span class="terminal-title-text">TERMINAL 03 // FINAL_SYNTHESIS</span>
                    <span class="terminal-badge badge-synthesis">REPORT</span>
                </div>
                """)
                answer_output = gr.Markdown(
                    """```bash
omnisearch@agent:~$ tail -f /var/log/executive_synthesis.md
[REPORT_GEN] Synthesis pipeline online.
[STATUS] Waiting for inspection & verification...
```""",
                    elem_classes=["terminal-window-body"]
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
            thinking_output,
            web_output,
            answer_output
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
