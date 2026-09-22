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
[SYS_WARN] No optical stream detected in stdin.
[STATUS] Please drop an image or click a preloaded test case.
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
            "⚠️ **[SYS_ALERT]**: Please upload or select an image to initiate autonomous mission.",
            None,
            [],
            init_term_reasoning,
            init_term_web,
            init_term_synthesis
        )
        return

    if not query.strip():
        query = "Identify the key item or landmark in this image, zoom in if needed, and search the web for details."

    boot_reasoning = """```bash
omnisearch@agent:~$ cat /proc/reasoning_stream
[SYS_INIT] Optical grounding matrix online.
[CUDA] NVIDIA RTX A4000 16GB (4-bit NF4) active.
[STATUS] Tokenizing visual features & compiling chain-of-thought...
```"""
    boot_web = """```bash
omnisearch@agent:~$ netstat --active-rag --monitor
[DAEMON] Live DuckDuckGo & Wikipedia RAG bridge active.
[STATUS] Monitoring perception stream for entity detection...
```"""
    boot_synthesis = """```bash
omnisearch@agent:~$ tail -f /var/log/executive_synthesis.md
[REPORT_GEN] Executive synthesis pipeline online.
[STATUS] Awaiting visual inspection & web verification...
```"""

    yield (
        "🚀 **[INITIALIZING]**: Loading weights into RTX A4000 & parsing mission prompt...",
        input_image,
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
        status = f"⚡ **[ACTIVITY]**: {state['status']}"
        annotated_img = state["annotated_image"]
        gallery = state["crop_gallery"]
        final_answer = state["final_answer"]
        web_md = state["web_results_md"]
        thinking_md = state["thinking_text"]

        yield (
            status,
            annotated_img,
            gallery,
            thinking_md,
            web_md,
            final_answer
        )

# Modern Cyber/Terminal CSS that fits completely in a single window
custom_css = """
/* GLOBAL RESET & VIEWPORT LOCK */
* {
    box-sizing: border-box !important;
}
html, body {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    background-color: #06090f !important;
    color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, Menlo, Consolas, monospace !important;
}

/* GRADIO CONTAINER LOCK */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 6px 12px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    background-color: #06090f !important;
}

/* TERMINAL SYSTEM NAVBAR */
.terminal-navbar {
    background: linear-gradient(90deg, #0d131f 0%, #111a2e 50%, #0d131f 100%);
    border: 1px solid #1c2738;
    border-radius: 6px;
    padding: 6px 12px;
    height: 38px;
    min-height: 38px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
    flex: 0 0 auto;
}
.terminal-nav-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.terminal-dots {
    display: flex;
    align-items: center;
    gap: 5px;
}
.terminal-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}
.terminal-dot.dot-red { background: #ff5f56; box-shadow: 0 0 5px rgba(255, 95, 86, 0.5); }
.terminal-dot.dot-yellow { background: #ffbd2e; box-shadow: 0 0 5px rgba(255, 189, 46, 0.5); }
.terminal-dot.dot-green { background: #27c93f; box-shadow: 0 0 5px rgba(39, 201, 63, 0.5); }

.terminal-logo-tag {
    font-size: 0.88rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #00f0ff;
}
.terminal-version-tag {
    font-size: 0.72rem;
    color: #64748b;
    border-left: 1px solid #1e293b;
    padding-left: 8px;
}
.terminal-nav-center {
    display: flex;
    align-items: center;
    gap: 14px;
}
.telemetry-item {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 600;
}
.telemetry-val {
    color: #38bdf8;
}
.terminal-nav-right {
    display: flex;
    align-items: center;
    gap: 6px;
}
.status-live-dot {
    width: 7px;
    height: 7px;
    background: #00ff9d;
    border-radius: 50%;
    box-shadow: 0 0 6px #00ff9d;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 0.4; }
    50% { opacity: 1; }
    100% { opacity: 0.4; }
}
.status-live-text {
    font-size: 0.72rem;
    font-weight: 700;
    color: #00ff9d;
    letter-spacing: 0.05em;
}

/* MAIN COCKPIT ROW (Fills remaining screen height) */
.main-cockpit-row {
    flex: 1 1 auto !important;
    height: calc(100vh - 54px) !important;
    max-height: calc(100vh - 54px) !important;
    overflow: hidden !important;
    display: flex !important;
    gap: 8px !important;
    margin: 0 !important;
}

/* LEFT COLUMN: CONTROL CONSOLE */
.control-console-col {
    height: 100% !important;
    max-height: 100% !important;
    overflow-y: auto !important;
    background: #090e17 !important;
    border: 1px solid #182232 !important;
    border-radius: 6px !important;
    padding: 8px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
}
.control-console-col::-webkit-scrollbar {
    width: 4px;
}
.control-console-col::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 2px;
}

/* RIGHT COLUMN: DISPLAY MATRIX */
.display-matrix-col {
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
    flex: 1 1 auto !important;
}

/* TELEMETRY STATUS PILL */
.telemetry-status-bar {
    height: 30px !important;
    min-height: 30px !important;
    max-height: 30px !important;
    background: #0d131f !important;
    border: 1px solid #1c2738 !important;
    border-radius: 5px !important;
    padding: 4px 10px !important;
    font-size: 0.78rem !important;
    color: #38bdf8 !important;
    display: flex !important;
    align-items: center !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

/* PANEL HEADER BAR */
.panel-header-bar {
    background: #0f1726;
    border-bottom: 1px solid #1c2738;
    padding: 4px 8px;
    height: 24px;
    min-height: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    user-select: none;
}
.panel-header-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #94a3b8;
    text-transform: uppercase;
}
.panel-header-badge {
    font-size: 0.65rem;
    padding: 1px 5px;
    border-radius: 3px;
    font-weight: 700;
}
.badge-reasoning { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
.badge-web { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
.badge-synthesis { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }

/* UPPER ROW: OPTICAL VIEWPORTS */
.optical-viewports-row {
    flex: 0 0 auto !important;
    height: 240px !important;
    max-height: 240px !important;
    overflow: hidden !important;
    display: flex !important;
    gap: 8px !important;
    margin: 0 !important;
}
.viewport-card {
    height: 100% !important;
    max-height: 100% !important;
    background: #090e17 !important;
    border: 1px solid #182232 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}
.viewport-card .viewport-content {
    flex: 1 1 auto !important;
    height: 216px !important;
    max-height: 216px !important;
    overflow: hidden !important;
    background: #06090f !important;
}

/* LOWER ROW: 3 PARALLEL TERMINALS */
.terminals-matrix-row {
    flex: 1 1 auto !important;
    height: calc(100% - 280px) !important;
    min-height: 200px !important;
    overflow: hidden !important;
    display: flex !important;
    gap: 8px !important;
    margin: 0 !important;
}
.cockpit-terminal-card {
    height: 100% !important;
    max-height: 100% !important;
    background: #090e17 !important;
    border: 1px solid #182232 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}
.cockpit-terminal-body {
    flex: 1 1 auto !important;
    height: calc(100% - 24px) !important;
    overflow-y: auto !important;
    padding: 8px 10px !important;
    font-size: 0.78rem !important;
    line-height: 1.45 !important;
    color: #cbd5e1 !important;
    background: #06090f !important;
}
.cockpit-terminal-body::-webkit-scrollbar {
    width: 4px;
}
.cockpit-terminal-body::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 2px;
}

/* TERMINAL TEXT FORMATTING */
.cockpit-terminal-body pre, .cockpit-terminal-body code {
    background: #0d131f !important;
    border: 1px solid #1c2738 !important;
    color: #38bdf8 !important;
    border-radius: 4px !important;
    font-size: 0.75rem !important;
}
.cockpit-terminal-body a {
    color: #38bdf8 !important;
    text-decoration: underline !important;
}
.cockpit-terminal-body strong {
    color: #f1f5f9 !important;
}
.cockpit-terminal-body blockquote {
    border-left: 2px solid #00f0ff !important;
    margin: 4px 0 !important;
    padding: 2px 8px !important;
    background: rgba(0, 240, 255, 0.05) !important;
    color: #94a3b8 !important;
}

/* TERMINAL BUTTONS */
.term-btn-primary {
    background: rgba(0, 240, 255, 0.12) !important;
    border: 1px solid #00f0ff !important;
    color: #00f0ff !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    font-size: 0.8rem !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
.term-btn-primary:hover {
    background: #00f0ff !important;
    color: #06090f !important;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.4) !important;
}
.term-btn-secondary {
    background: #0d131f !important;
    border: 1px solid #1c2738 !important;
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    border-radius: 4px !important;
}
.term-btn-pill {
    background: #0d131f !important;
    border: 1px solid #1c2738 !important;
    color: #cbd5e1 !important;
    font-size: 0.72rem !important;
    padding: 3px 6px !important;
    border-radius: 3px !important;
}
.term-btn-pill:hover {
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
}
"""

with gr.Blocks(title="OmniSearch Cockpit") as demo:
    # 1. Compact Terminal Top Navbar
    gr.HTML("""
    <div class="terminal-navbar">
        <div class="terminal-nav-left">
            <div class="terminal-dots">
                <span class="terminal-dot dot-red"></span>
                <span class="terminal-dot dot-yellow"></span>
                <span class="terminal-dot dot-green"></span>
            </div>
            <span class="terminal-logo-tag">OMNISEARCH_OS</span>
            <span class="terminal-version-tag">v2.5 // CYBER-COCKPIT</span>
        </div>
        <div class="terminal-nav-center">
            <span class="telemetry-item">⚡ HOST: <span class="telemetry-val">local-linux</span></span>
            <span class="telemetry-item">🎮 GPU: <span class="telemetry-val">RTX A4000 16GB (NF4)</span></span>
            <span class="telemetry-item">🧠 CORE: <span class="telemetry-val">Mini-o3 / Qwen2.5-VL</span></span>
            <span class="telemetry-item">🌐 RAG: <span class="telemetry-val">DDG + Wiki [ONLINE]</span></span>
        </div>
        <div class="terminal-nav-right">
            <span class="status-live-dot"></span>
            <span class="status-live-text">KERNEL ACTIVE</span>
        </div>
    </div>
    """)

    # 2. Main Full-Window Row
    with gr.Row(elem_classes=["main-cockpit-row"]):
        # Left: Control Console
        with gr.Column(scale=3, min_width=290, elem_classes=["control-console-col"]):
            gr.HTML("""
            <div class="panel-header-bar">
                <span class="panel-header-title">INPUT_CONSOLE // MISSION_CONTROL</span>
                <span class="panel-header-badge" style="background:#1e293b; color:#94a3b8;">STDIN</span>
            </div>
            """)

            image_input = gr.Image(
                type="pil",
                label="📸 Optical Stream Input",
                sources=["upload", "clipboard"],
                height=180
            )

            query_input = gr.Textbox(
                label="💬 Mission Objective Prompt",
                placeholder="$ Enter autonomous visual objective...",
                lines=2
            )

            # Quick Prompt Pills
            with gr.Row():
                p1_btn = gr.Button("🇩🇪 Bins", size="sm", elem_classes=["term-btn-pill"])
                p2_btn = gr.Button("🎬 Venue", size="sm", elem_classes=["term-btn-pill"])
                p3_btn = gr.Button("🏛️ Landmark", size="sm", elem_classes=["term-btn-pill"])
                p4_btn = gr.Button("🔬 Inspect", size="sm", elem_classes=["term-btn-pill"])

            # Settings Accordion
            with gr.Accordion("⚙️ Engine Parameters", open=False):
                model_selector = gr.Dropdown(
                    choices=[
                        "Qwen/Qwen2.5-VL-3B-Instruct",
                        "Mini-o3/Mini-o3-7B-v1",
                        "Qwen/Qwen2.5-VL-7B-Instruct"
                    ],
                    value="Qwen/Qwen2.5-VL-3B-Instruct",
                    label="Backbone VLM"
                )
                max_turns_slider = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Max Reasoning Turns"
                )
                auto_web_checkbox = gr.Checkbox(
                    value=True,
                    label="Active Web Grounding (DDG + Wiki)"
                )

            with gr.Row():
                submit_btn = gr.Button("⚡ [ EXECUTE ]", variant="primary", scale=3, elem_classes=["term-btn-primary"])
                clear_btn = gr.ClearButton([image_input, query_input], value="↺ Reset", scale=1, elem_classes=["term-btn-secondary"])

            # Preloaded Test Cases
            with gr.Accordion("📂 Preloaded Test Cases", open=False):
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

        # Right: Display Matrix (Upper Viewports + Lower Terminals)
        with gr.Column(scale=9, elem_classes=["display-matrix-col"]):
            # Status telemetry line
            status_output = gr.Markdown(
                "⚡ **[TELEMETRY]**: Ready. Select an example or drop an image and click **[ EXECUTE ]**.",
                elem_classes=["telemetry-status-bar"]
            )

            # Upper Row: Optical Grounding Matrix (Canvas + Crops side-by-side)
            with gr.Row(elem_classes=["optical-viewports-row"]):
                # Viewport 1: Live Grounding Canvas
                with gr.Column(scale=1, elem_classes=["viewport-card"]):
                    gr.HTML("""
                    <div class="panel-header-bar">
                        <div class="terminal-dots">
                            <span class="terminal-dot dot-red"></span>
                            <span class="terminal-dot dot-yellow"></span>
                            <span class="terminal-dot dot-green"></span>
                        </div>
                        <span class="panel-header-title">VIEWPORT_01 // OPTICAL_BOUNDING_MATRIX</span>
                        <span class="panel-header-badge" style="background:#0284c7; color:#f0f9ff;">CANVAS</span>
                    </div>
                    """)
                    canvas_output = gr.Image(
                        interactive=False,
                        height=214,
                        show_label=False,
                        elem_classes=["viewport-content"]
                    )

                # Viewport 2: Zoomed Inspection Patches
                with gr.Column(scale=1, elem_classes=["viewport-card"]):
                    gr.HTML("""
                    <div class="panel-header-bar">
                        <div class="terminal-dots">
                            <span class="terminal-dot dot-red"></span>
                            <span class="terminal-dot dot-yellow"></span>
                            <span class="terminal-dot dot-green"></span>
                        </div>
                        <span class="panel-header-title">TARGET_LOCK // MULTI_TURN_CROPS</span>
                        <span class="panel-header-badge" style="background:#7c3aed; color:#f5f3ff;">GALLERY</span>
                    </div>
                    """)
                    crops_gallery = gr.Gallery(
                        columns=3,
                        rows=1,
                        height=214,
                        preview=True,
                        allow_preview=True,
                        object_fit="contain",
                        show_label=False,
                        elem_classes=["viewport-content"]
                    )

            # Lower Row: 3 Parallel Side-by-Side Terminals
            with gr.Row(elem_classes=["terminals-matrix-row"]):
                # Terminal 1: Deep Reasoning Trace (<think>)
                with gr.Column(scale=1, min_width=240, elem_classes=["cockpit-terminal-card"]):
                    gr.HTML("""
                    <div class="panel-header-bar">
                        <div class="terminal-dots">
                            <span class="terminal-dot dot-red"></span>
                            <span class="terminal-dot dot-yellow"></span>
                            <span class="terminal-dot dot-green"></span>
                        </div>
                        <span class="panel-header-title">TERMINAL 01 // REASONING_TRACE</span>
                        <span class="panel-header-badge badge-reasoning">&lt;THINK&gt;</span>
                    </div>
                    """)
                    thinking_output = gr.Markdown(
                        """```bash
omnisearch@agent:~$ cat /proc/reasoning_stream
[SYS_INIT] Qwen2.5-VL CoT daemon active.
[STATUS] Awaiting visual input & objective...
```""",
                        elem_classes=["cockpit-terminal-body"]
                    )

                # Terminal 2: Live Web Sources (Active RAG)
                with gr.Column(scale=1, min_width=240, elem_classes=["cockpit-terminal-card"]):
                    gr.HTML("""
                    <div class="panel-header-bar">
                        <div class="terminal-dots">
                            <span class="terminal-dot dot-red"></span>
                            <span class="terminal-dot dot-yellow"></span>
                            <span class="terminal-dot dot-green"></span>
                        </div>
                        <span class="panel-header-title">TERMINAL 02 // LIVE_WEB_INTEL</span>
                        <span class="panel-header-badge badge-web">ACTIVE_RAG</span>
                    </div>
                    """)
                    web_output = gr.Markdown(
                        """```bash
omnisearch@agent:~$ netstat --active-rag --monitor
[DAEMON] DuckDuckGo & Wikipedia bridge online.
[STATUS] Standby: monitoring visual stream...
```""",
                        elem_classes=["cockpit-terminal-body"]
                    )

                # Terminal 3: Final Synthesis
                with gr.Column(scale=1, min_width=240, elem_classes=["cockpit-terminal-card"]):
                    gr.HTML("""
                    <div class="panel-header-bar">
                        <div class="terminal-dots">
                            <span class="terminal-dot dot-red"></span>
                            <span class="terminal-dot dot-yellow"></span>
                            <span class="terminal-dot dot-green"></span>
                        </div>
                        <span class="panel-header-title">TERMINAL 03 // FINAL_SYNTHESIS</span>
                        <span class="panel-header-badge badge-synthesis">REPORT</span>
                    </div>
                    """)
                    answer_output = gr.Markdown(
                        """```bash
omnisearch@agent:~$ tail -f /var/log/synthesis.md
[REPORT_GEN] Synthesis pipeline online.
[STATUS] Waiting for inspection & verification...
```""",
                        elem_classes=["cockpit-terminal-body"]
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
            canvas_output,
            crops_gallery,
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
            font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"]
        ),
        css=custom_css
    )
