import os
import argparse
import gradio as gr
from PIL import Image
from typing import List
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

def format_hud_stepper(stage: str, progress: int, status_text: str) -> str:
    stage_order = ["INIT", "PERCEPTION", "CROPPING", "ACTIVE_RAG", "SYNTHESIS", "COMPLETE"]
    current_idx = stage_order.index(stage) if stage in stage_order else 0

    stages_def = [
        ("INIT", "1. INGEST"),
        ("PERCEPTION", "2. PERCEIVE"),
        ("CROPPING", "3. ZOOM CROP"),
        ("ACTIVE_RAG", "4. WEB RAG"),
        ("SYNTHESIS", "5. REPORT"),
    ]

    chips_html = ""
    for i, (st_key, st_label) in enumerate(stages_def):
        if i < current_idx or stage == "COMPLETE":
            cls = "step-chip done"
            lbl = f"[OK] {st_label}"
        elif i == current_idx:
            cls = "step-chip active"
            lbl = f"[RUN] {st_label}"
        else:
            cls = "step-chip pending"
            lbl = st_label

        chips_html += f'<span class="{cls}">{lbl}</span>'
        if i < len(stages_def) - 1:
            conn_cls = "conn-done" if (i < current_idx or stage == "COMPLETE") else "conn-pending"
            chips_html += f'<span class="step-connector {conn_cls}">&gt;</span>'

    filled = int(progress / 5)
    empty = 20 - filled
    ascii_bar = "█" * filled + "░" * empty

    report_banner = ""
    if stage == "COMPLETE":
        report_banner = '<span class="hud-report-alert">[REPORT READY]</span>'

    return f"""
    <div class="hud-stepper-box">
        <div class="hud-top-line">
            <div class="hud-stages-pills">{chips_html}</div>
            <div class="hud-metric-box">
                {report_banner}
                <span class="hud-ascii-bar">[{ascii_bar}]</span>
                <span class="hud-percent-pill">{progress}%</span>
            </div>
        </div>
        <div class="hud-progress-line">
            <div class="hud-progress-track">
                <div class="hud-progress-fill" style="width: {progress}%;"></div>
            </div>
        </div>
        <div class="hud-status-line">
            <span class="hud-activity-label">[{stage}]:</span>
            <span class="hud-activity-text">{status_text}</span>
        </div>
    </div>
    """

def format_event_logs(logs: List[str]) -> str:
    if not logs:
        return """```bash
[00.00s] [INF] [STANDBY] Event logger daemon active.
[00.00s] [INF] [READY] Select an example or upload an image and click [EXECUTE].
```"""
    recent = logs[-14:]
    return "```bash\n" + "\n".join(recent) + "\n```"

def run_agent_interface(
    input_image: Image.Image,
    query: str,
    max_turns: int,
    auto_web: bool,
    model_choice: str
):
    if input_image is None:
        init_hud = format_hud_stepper("INIT", 0, "[ALERT]: Please drop an image or select a test case.")
        init_logs = """```bash
[00.00s] [WRN] [ALERT] No optical feed detected.
[00.00s] [INF] [STANDBY] Awaiting image upload...
```"""
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
omnisearch@agent:~$ cat /var/out/EXECUTIVE_REPORT.md
+-------------------------------------------------------+
|  EXECUTIVE MISSION SYNTHESIS & FINDINGS REPORT        |
+-------------------------------------------------------+
[STATUS] Idle. Waiting for visual perception & web grounding...
```"""
        yield (
            init_hud,
            init_logs,
            None,
            [],
            init_term_reasoning,
            init_term_web,
            init_term_synthesis,
            init_term_synthesis
        )
        return

    if not query.strip():
        query = "Identify the key item or landmark in this image, zoom in if needed, and search the web for details."

    boot_hud = format_hud_stepper("INIT", 12, "Loading neural weights into RTX A4000 VRAM...")
    boot_logs = """```bash
[00.00s] [INF] [START] Ingested optical stream.
[00.15s] [INF] [GPU_ALLOC] Allocated 8.5 GB VRAM on NVIDIA RTX A4000 (NF4).
[00.30s] [INF] [VLM_LOAD] Compiling Qwen2.5-VL vision-language encoder...
```"""
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
omnisearch@agent:~$ tail -f /var/log/EXECUTIVE_REPORT.md
+-------------------------------------------------------+
|  EXECUTIVE MISSION SYNTHESIS & FINDINGS REPORT        |
+-------------------------------------------------------+
[STATUS] Awaiting visual inspection & web verification...
```"""

    yield (
        boot_hud,
        boot_logs,
        input_image,
        [],
        boot_reasoning,
        boot_web,
        boot_synthesis,
        boot_synthesis
    )

    agent = get_agent(model_name=model_choice, load_in_4bit=True)

    for state in agent.run_streaming(
        original_image=input_image,
        query=query,
        max_turns=max_turns,
        auto_web_search=auto_web
    ):
        stage = state.get("stage", "PERCEPTION")
        progress = state.get("progress", 50)
        status_text = state["status"]
        event_logs = state.get("event_logs", [])

        if stage == "COMPLETE":
            status_text = "Mission complete. Executive report ready in Terminal 03 and Report view."

        hud_html = format_hud_stepper(stage, progress, status_text)
        logs_md = format_event_logs(event_logs)

        annotated_img = state["annotated_image"]
        gallery = state["crop_gallery"]
        final_answer = state["final_answer"]
        web_md = state["web_results_md"]
        thinking_md = state["thinking_text"]

        yield (
            hud_html,
            logs_md,
            annotated_img,
            gallery,
            thinking_md,
            web_md,
            final_answer,        # Terminal 03 (Parallel View)
            final_answer         # Full Report View (Dedicated Tab)
        )

# Modern Cyber/Terminal CSS with Fixed Left Sidebar & Zero Window Scrolling
custom_css = """
/* GLOBAL RESET & SCREEN LOCK */
* {
    box-sizing: border-box !important;
}
html, body {
    height: 100vh !important;
    max-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    background-color: #06090f !important;
    color: #f8fafc !important;
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, Menlo, Consolas, monospace !important;
    overflow: hidden !important;
}

/* DISABLE GRADIO'S INTRUSIVE GREY LOADING OVERLAY */
.wrap.default.full.translucent,
.wrap.default.full,
.loading-status,
.progress-bar-wrap {
    display: none !important;
}

/* GRADIO CONTAINER: FIT EXACTLY IN 100VH WITHOUT WINDOW SCROLL */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 4px 8px !important;
    display: flex !important;
    flex-direction: column !important;
    background-color: #06090f !important;
    overflow: hidden !important;
}

/* TERMINAL SYSTEM NAVBAR (Compact 28px) */
.terminal-navbar {
    background: linear-gradient(90deg, #0d131f 0%, #111a2e 50%, #0d131f 100%);
    border: 1px solid #1c2738;
    border-radius: 5px;
    padding: 2px 10px;
    height: 28px;
    min-height: 28px;
    max-height: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
    flex: 0 0 28px;
}
.terminal-nav-left {
    display: flex;
    align-items: center;
    gap: 8px;
}
.terminal-dots {
    display: flex;
    align-items: center;
    gap: 4px;
}
.terminal-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.terminal-dot.dot-red { background: #ff5f56; box-shadow: 0 0 5px rgba(255, 95, 86, 0.5); }
.terminal-dot.dot-yellow { background: #ffbd2e; box-shadow: 0 0 5px rgba(255, 189, 46, 0.5); }
.terminal-dot.dot-green { background: #27c93f; box-shadow: 0 0 5px rgba(39, 201, 63, 0.5); }

.terminal-logo-tag {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #00f0ff;
}
.terminal-version-tag {
    font-size: 0.68rem;
    color: #64748b;
    border-left: 1px solid #1e293b;
    padding-left: 6px;
}
.terminal-nav-center {
    display: flex;
    align-items: center;
    gap: 12px;
}
.telemetry-item {
    font-size: 0.68rem;
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
    width: 6px;
    height: 6px;
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
    font-size: 0.68rem;
    font-weight: 700;
    color: #00ff9d;
    letter-spacing: 0.05em;
}

/* MAIN COCKPIT ROW: FIXED HORIZONTAL SPLIT (NEVER VERTICAL STACK, NEVER WRAP) */
#main_cockpit_row,
.main-cockpit-row,
.gradio-container .main-cockpit-row,
div.main-cockpit-row {
    flex: 1 1 0% !important;
    height: calc(100vh - 38px) !important;
    max-height: calc(100vh - 38px) !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    width: 100% !important;
    gap: 8px !important;
    margin: 0 !important;
    overflow: hidden !important;
}

/* LEFT SIDEBAR: RIGIDLY PINNED ON THE LEFT (NEVER SPANS FULL SCREEN) */
#control_console_col,
.control-console-col,
.gradio-container .control-console-col {
    flex: 0 0 310px !important;
    width: 310px !important;
    max-width: 310px !important;
    min-width: 280px !important;
    height: 100% !important;
    max-height: 100% !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    background: #090e17 !important;
    border: 1px solid #182232 !important;
    border-radius: 6px !important;
    padding: 6px 8px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
}
.control-console-col::-webkit-scrollbar {
    width: 4px;
}
.control-console-col::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 2px;
}

/* COMPACT INPUTS IN LEFT SIDEBAR */
.control-console-col .block {
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
}
.control-console-col label {
    margin-bottom: 2px !important;
}
.control-console-col label span {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    color: #94a3b8 !important;
}
.control-console-col .image-container,
.control-console-col [data-testid="image"] {
    max-height: 125px !important;
    min-height: 105px !important;
    height: 120px !important;
}
.control-console-col [data-testid="image"] img {
    object-fit: contain !important;
    max-height: 115px !important;
}
.control-console-col textarea {
    font-size: 0.74rem !important;
    line-height: 1.3 !important;
    padding: 4px 6px !important;
    min-height: 44px !important;
    max-height: 52px !important;
    background: #06090f !important;
    border: 1px solid #1e293b !important;
    color: #f8fafc !important;
    border-radius: 4px !important;
}

/* ACTION BUTTON ROW */
.action-btn-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 4px !important;
    margin: 2px 0 !important;
    flex: 0 0 30px !important;
    height: 30px !important;
    min-height: 30px !important;
}
.term-btn-primary {
    background: rgba(0, 240, 255, 0.15) !important;
    border: 1px solid #00f0ff !important;
    color: #00f0ff !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em !important;
    font-size: 0.76rem !important;
    border-radius: 4px !important;
    height: 30px !important;
    min-height: 30px !important;
    padding: 0 6px !important;
    transition: all 0.2s ease !important;
}
.term-btn-primary:hover {
    background: #00f0ff !important;
    color: #06090f !important;
    box-shadow: 0 0 12px rgba(0, 240, 255, 0.5) !important;
}
.term-btn-secondary {
    background: #0d131f !important;
    border: 1px solid #1c2738 !important;
    color: #94a3b8 !important;
    font-size: 0.72rem !important;
    border-radius: 4px !important;
    height: 30px !important;
    min-height: 30px !important;
    padding: 0 6px !important;
}

/* LIVE EVENT LOG WIDGET IN SIDEBAR */
.event-logger-box {
    background: #06090f !important;
    border: 1px solid #182232 !important;
    border-radius: 5px !important;
    overflow: hidden !important;
    flex: 0 0 auto !important;
    margin-top: 2px !important;
}
.event-logger-body {
    height: 100px !important;
    max-height: 100px !important;
    overflow-y: auto !important;
    padding: 3px 6px !important;
    font-size: 0.66rem !important;
    line-height: 1.3 !important;
    background: #06090f !important;
    color: #38bdf8 !important;
}
.event-logger-body::-webkit-scrollbar {
    width: 3px;
}
.event-logger-body::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 2px;
}

/* ACCORDIONS IN SIDEBAR */
.sidebar-accordion {
    background: #090e17 !important;
    border: 1px solid #182232 !important;
    border-radius: 4px !important;
    margin: 2px 0 !important;
    flex: 0 0 auto !important;
}
.sidebar-accordion summary {
    padding: 3px 6px !important;
    font-size: 0.68rem !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    cursor: pointer !important;
}

/* RIGHT COLUMN: DISPLAY MATRIX */
#display_matrix_col,
.display-matrix-col,
.gradio-container .display-matrix-col {
    flex: 1 1 0% !important;
    width: 0 !important;
    min-width: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
}

/* DYNAMIC STEPPER & HUD BOX */
.hud-stepper-box {
    background: linear-gradient(90deg, #0d131f 0%, #111a2e 100%);
    border: 1px solid #1c2738;
    border-radius: 5px;
    padding: 3px 8px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 0 0 auto;
}
.hud-top-line {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hud-stages-pills {
    display: flex;
    align-items: center;
    gap: 3px;
}
.step-chip {
    font-size: 0.64rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.step-chip.done {
    background: rgba(0, 255, 157, 0.12);
    color: #00ff9d;
    border: 1px solid rgba(0, 255, 157, 0.3);
}
.step-chip.active {
    background: rgba(0, 240, 255, 0.2);
    color: #00f0ff;
    border: 1px solid #00f0ff;
    box-shadow: 0 0 8px rgba(0, 240, 255, 0.3);
    animation: activeGlow 1.5s infinite alternate;
}
@keyframes activeGlow {
    0% { border-color: rgba(0, 240, 255, 0.5); }
    100% { border-color: #00f0ff; }
}
.step-chip.pending {
    background: rgba(30, 41, 59, 0.6);
    color: #64748b;
    border: 1px solid #1e293b;
}
.step-connector {
    font-size: 0.62rem;
}
.step-connector.conn-done { color: #00ff9d; }
.step-connector.conn-pending { color: #334155; }

.hud-metric-box {
    display: flex;
    align-items: center;
    gap: 6px;
}
.hud-report-alert {
    background: rgba(0, 255, 157, 0.2);
    color: #00ff9d;
    border: 1px solid #00ff9d;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 0.64rem;
    font-weight: 800;
    animation: pulse 1.5s infinite;
}
.hud-ascii-bar {
    font-size: 0.66rem;
    color: #38bdf8;
    letter-spacing: 0.04em;
}
.hud-percent-pill {
    font-size: 0.66rem;
    font-weight: 800;
    background: #1e293b;
    color: #00f0ff;
    padding: 0px 5px;
    border-radius: 3px;
    border: 1px solid #00f0ff;
}

.hud-progress-line {
    width: 100%;
}
.hud-progress-track {
    width: 100%;
    height: 3px;
    background: #162032;
    border-radius: 2px;
    overflow: hidden;
}
.hud-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #00f0ff 0%, #00ff9d 100%);
    box-shadow: 0 0 6px #00f0ff;
    transition: width 0.3s ease;
}

.hud-status-line {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.70rem;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
.hud-activity-label {
    font-weight: 800;
    color: #00f0ff;
}
.hud-activity-text {
    color: #94a3b8;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* PANEL HEADER BAR */
.panel-header-bar {
    background: #0f1726;
    border-bottom: 1px solid #1c2738;
    padding: 3px 8px;
    height: 22px;
    min-height: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    user-select: none;
    flex: 0 0 22px;
}
.panel-header-title {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #94a3b8;
    text-transform: uppercase;
}
.panel-header-badge {
    font-size: 0.60rem;
    padding: 1px 4px;
    border-radius: 3px;
    font-weight: 700;
}
.badge-reasoning { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
.badge-web { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
.badge-synthesis { background: #00ff9d; color: #06090f; font-weight: 900; }

/* UPPER ROW: OPTICAL VIEWPORTS (CANVAS + CROPS) */
.optical-viewports-row {
    flex: 0 0 160px !important;
    height: 160px !important;
    min-height: 160px !important;
    max-height: 160px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 6px !important;
    margin: 0 !important;
}
.viewport-card {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    height: 100% !important;
    background: #090e17 !important;
    border: 1px solid #182232 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}
.viewport-card .viewport-content {
    flex: 1 1 auto !important;
    height: 138px !important;
    max-height: 138px !important;
    overflow: hidden !important;
    background: #06090f !important;
}
.viewport-card .viewport-content img {
    object-fit: contain !important;
    height: 135px !important;
}

/* LOWER DECK: TABS CONTAINER */
.cockpit-deck-tabs {
    flex: 1 1 0% !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
}
.cockpit-deck-tabs > .tab-nav {
    flex: 0 0 24px !important;
    height: 24px !important;
    min-height: 24px !important;
    border-bottom: 1px solid #1c2738 !important;
    margin-bottom: 3px !important;
    background: transparent !important;
}
.cockpit-deck-tabs > .tab-nav button {
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    padding: 2px 10px !important;
    color: #94a3b8 !important;
    border-radius: 4px 4px 0 0 !important;
    border: 1px solid transparent !important;
}
.cockpit-deck-tabs > .tab-nav button.selected {
    color: #00ff9d !important;
    background: #090e17 !important;
    border-color: #1c2738 #1c2738 transparent #1c2738 !important;
}
.cockpit-deck-tabs > .tabitem,
.cockpit-deck-tabs > div[role="tabpanel"],
.cockpit-deck-tabs [role="tabpanel"] {
    flex: 1 1 0% !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    padding: 0 !important;
}

/* LOWER ROW: 3 PARALLEL TERMINALS */
.terminals-matrix-row {
    flex: 1 1 0% !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 6px !important;
    margin: 0 !important;
    overflow: hidden !important;
}
.cockpit-terminal-card {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    background: #090e17 !important;
    border: 1px solid #182232 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}
.cockpit-terminal-card-report {
    flex: 1.35 1 0% !important;
    border: 2px solid #00ff9d !important;
    box-shadow: 0 0 16px rgba(0, 255, 157, 0.22) !important;
}
.cockpit-terminal-body {
    flex: 1 1 0% !important;
    height: calc(100% - 22px) !important;
    max-height: calc(100% - 22px) !important;
    overflow-y: auto !important;
    padding: 6px 8px !important;
    font-size: 0.74rem !important;
    line-height: 1.45 !important;
    color: #f8fafc !important;
    background: #06090f !important;
}
.cockpit-terminal-body::-webkit-scrollbar {
    width: 4px;
}
.cockpit-terminal-body::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 2px;
}

/* FULL REPORT TAB STYLING */
.full-report-container {
    flex: 1 1 0% !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    background: #090e17 !important;
    border: 2px solid #00ff9d !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    box-shadow: 0 0 20px rgba(0, 255, 157, 0.2) !important;
}
.full-report-body {
    flex: 1 1 0% !important;
    height: calc(100% - 22px) !important;
    max-height: calc(100% - 22px) !important;
    overflow-y: auto !important;
    padding: 10px 14px !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
    color: #f8fafc !important;
    background: #06090f !important;
}
.full-report-body::-webkit-scrollbar {
    width: 5px;
}
.full-report-body::-webkit-scrollbar-thumb {
    background: #00ff9d;
    border-radius: 3px;
}

/* HIGH CONTRAST REPORT TEXT STYLING (FIX GREYED-OUT TEXT) */
.cockpit-terminal-body, .cockpit-terminal-body *,
.full-report-body, .full-report-body * {
    opacity: 1 !important;
}
.cockpit-terminal-body p, .cockpit-terminal-body li, .cockpit-terminal-body span,
.full-report-body p, .full-report-body li, .full-report-body span {
    color: #f8fafc !important; /* Bright crisp white */
    font-size: 0.78rem !important;
    line-height: 1.55 !important;
}
.cockpit-terminal-body h1, .cockpit-terminal-body h2, .cockpit-terminal-body h3, .cockpit-terminal-body h4,
.full-report-body h1, .full-report-body h2, .full-report-body h3, .full-report-body h4 {
    color: #00ff9d !important; /* Vivid neon emerald */
    font-weight: 800 !important;
    margin-top: 8px !important;
    margin-bottom: 4px !important;
}
/* Override Gradio prose default dimming */
.prose p, .prose li, .prose span {
    color: #f8fafc !important;
}
.prose h1, .prose h2, .prose h3 {
    color: #00ff9d !important;
}

/* TERMINAL TEXT FORMATTING */
.cockpit-terminal-body pre, .cockpit-terminal-body code, .event-logger-body pre, .event-logger-body code, .full-report-body pre, .full-report-body code {
    background: #0d131f !important;
    border: 1px solid #1c2738 !important;
    color: #38bdf8 !important;
    border-radius: 3px !important;
    font-size: 0.72rem !important;
}
.cockpit-terminal-body a, .full-report-body a {
    color: #38bdf8 !important;
    text-decoration: underline !important;
    font-weight: 600 !important;
}
.cockpit-terminal-body strong, .full-report-body strong {
    color: #00ff9d !important;
    font-weight: 700 !important;
}
.cockpit-terminal-body blockquote, .full-report-body blockquote {
    border-left: 2px solid #00f0ff !important;
    margin: 4px 0 !important;
    padding: 2px 8px !important;
    background: rgba(0, 240, 255, 0.05) !important;
    color: #cbd5e1 !important;
}

/* HARD OVERRIDE AGAINST RESPONSIVE COLLAPSE / WRAPPING AT ANY RESOLUTION */
@media (max-width: 3840px), (max-width: 2560px), (max-width: 1920px), (max-width: 1440px), (max-width: 1200px), (max-width: 900px), (max-width: 768px) {
    #main_cockpit_row,
    .main-cockpit-row {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
    }
    #control_console_col,
    .control-console-col {
        flex: 0 0 310px !important;
        width: 310px !important;
        max-width: 310px !important;
        min-width: 280px !important;
    }
    #display_matrix_col,
    .display-matrix-col {
        flex: 1 1 0% !important;
        width: 0 !important;
        min-width: 0 !important;
    }
    .optical-viewports-row {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
    }
    .terminals-matrix-row {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
    }
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
            <span class="telemetry-item">HOST: <span class="telemetry-val">local-linux</span></span>
            <span class="telemetry-item">GPU: <span class="telemetry-val">RTX A4000 16GB (NF4)</span></span>
            <span class="telemetry-item">CORE: <span class="telemetry-val">Mini-o3 / Qwen2.5-VL</span></span>
            <span class="telemetry-item">RAG: <span class="telemetry-val">DDG + Wiki [ONLINE]</span></span>
        </div>
        <div class="terminal-nav-right">
            <span class="status-live-dot"></span>
            <span class="status-live-text">KERNEL ACTIVE</span>
        </div>
    </div>
    """)

    # 2. Main Cockpit Row
    with gr.Row(elem_id="main_cockpit_row", elem_classes=["main-cockpit-row"]):
        # Left: Clean Control Console & Live Event Logger (Pinned left sidebar)
        with gr.Column(scale=1, min_width=280, elem_id="control_console_col", elem_classes=["control-console-col"]):
            gr.HTML("""
            <div class="panel-header-bar">
                <span class="panel-header-title">INPUT_CONSOLE // MISSION_CONTROL</span>
                <span class="panel-header-badge" style="background:#1e293b; color:#94a3b8;">STDIN</span>
            </div>
            """)

            image_input = gr.Image(
                type="pil",
                label="Optical Stream Input",
                sources=["upload", "clipboard"],
                height=120
            )

            query_input = gr.Textbox(
                label="Mission Objective Prompt",
                placeholder="$ Enter autonomous visual objective or question...",
                lines=2,
                max_lines=2
            )

            with gr.Row(elem_classes=["action-btn-row"]):
                submit_btn = gr.Button("[ EXECUTE MISSION ]", variant="primary", scale=3, elem_classes=["term-btn-primary"])
                clear_btn = gr.ClearButton([image_input, query_input], value="Reset", scale=1, elem_classes=["term-btn-secondary"])

            # Live Event Logger Widget
            with gr.Group(elem_classes=["event-logger-box"]):
                gr.HTML("""
                <div class="panel-header-bar">
                    <div class="terminal-dots">
                        <span class="terminal-dot dot-red"></span>
                        <span class="terminal-dot dot-yellow"></span>
                        <span class="terminal-dot dot-green"></span>
                    </div>
                    <span class="panel-header-title">SYS_LOG // REALTIME_TRACE</span>
                    <span class="panel-header-badge" style="background:#0284c7; color:#fff;">STREAM</span>
                </div>
                """)
                event_logs_output = gr.Markdown(
                    format_event_logs([]),
                    elem_classes=["event-logger-body"]
                )

            # Settings Accordion
            with gr.Accordion("Engine Parameters", open=False, elem_classes=["sidebar-accordion"]):
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

            # Preloaded Test Cases
            with gr.Accordion("Preloaded Test Cases", open=False, elem_classes=["sidebar-accordion"]):
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

        # Right: Display Matrix (Upper Viewports + Lower Terminals & Full Report)
        with gr.Column(scale=4, min_width=0, elem_id="display_matrix_col", elem_classes=["display-matrix-col"]):
            # Dynamic Stepper & HUD Box
            hud_stepper_output = gr.HTML(
                format_hud_stepper("INIT", 0, "Ready. Select an example or drop an image and click [EXECUTE MISSION].")
            )

            # Upper Row: Optical Grounding Matrix (Canvas + Crops side-by-side)
            with gr.Row(elem_classes=["optical-viewports-row"]):
                # Viewport 1: Live Grounding Canvas
                with gr.Column(scale=1, min_width=0, elem_classes=["viewport-card"]):
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
                        height=138,
                        show_label=False,
                        elem_classes=["viewport-content"]
                    )

                # Viewport 2: Zoomed Inspection Patches
                with gr.Column(scale=1, min_width=0, elem_classes=["viewport-card"]):
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
                        height=138,
                        preview=True,
                        allow_preview=True,
                        object_fit="contain",
                        show_label=False,
                        elem_classes=["viewport-content"]
                    )

            # Lower Row: Tabs for Parallel Cockpit vs. Full Report View
            with gr.Tabs(elem_classes=["cockpit-deck-tabs"]):
                with gr.Tab("PARALLEL COCKPIT (Reasoning + Web + Report Side-by-Side)"):
                    with gr.Row(elem_classes=["terminals-matrix-row"]):
                        # Terminal 1: Deep Reasoning Trace (<think>)
                        with gr.Column(scale=1, min_width=0, elem_classes=["cockpit-terminal-card"]):
                            gr.HTML("""
                            <div class="panel-header-bar">
                                <div class="terminal-dots">
                                    <span class="terminal-dot dot-red"></span>
                                    <span class="terminal-dot dot-yellow"></span>
                                    <span class="terminal-dot dot-green"></span>
                                </div>
                                <span class="panel-header-title">TERMINAL 01 // REASONING</span>
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
                        with gr.Column(scale=1, min_width=0, elem_classes=["cockpit-terminal-card"]):
                            gr.HTML("""
                            <div class="panel-header-bar">
                                <div class="terminal-dots">
                                    <span class="terminal-dot dot-red"></span>
                                    <span class="terminal-dot dot-yellow"></span>
                                    <span class="terminal-dot dot-green"></span>
                                </div>
                                <span class="panel-header-title">TERMINAL 02 // WEB_INTEL</span>
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

                        # Terminal 3: Final Executive Synthesis Report
                        with gr.Column(scale=1, min_width=0, elem_classes=["cockpit-terminal-card", "cockpit-terminal-card-report"]):
                            gr.HTML("""
                            <div class="panel-header-bar" style="background:#091b15; border-bottom: 2px solid #00ff9d;">
                                <div class="terminal-dots">
                                    <span class="terminal-dot dot-red"></span>
                                    <span class="terminal-dot dot-yellow"></span>
                                    <span class="terminal-dot dot-green"></span>
                                </div>
                                <span class="panel-header-title" style="color:#00ff9d; font-weight:900;">TERMINAL 03 // FINAL_REPORT</span>
                                <span class="panel-header-badge badge-synthesis" style="background:#00ff9d; color:#06090f; font-weight:900;">REPORT</span>
                            </div>
                            """)
                            answer_output = gr.Markdown(
                                """```bash
omnisearch@agent:~$ cat /var/out/EXECUTIVE_REPORT.md
+-------------------------------------------------------+
|  EXECUTIVE MISSION SYNTHESIS & FINDINGS REPORT        |
+-------------------------------------------------------+
[STATUS] Waiting for inspection & verification...
```""",
                                elem_classes=["cockpit-terminal-body"]
                            )

                with gr.Tab("FULL REPORT VIEW (Executive Summary & Findings)"):
                    with gr.Group(elem_classes=["full-report-container"]):
                        gr.HTML("""
                        <div class="panel-header-bar" style="background:#091b15; border-bottom: 2px solid #00ff9d;">
                            <div class="terminal-dots">
                                <span class="terminal-dot dot-red"></span>
                                <span class="terminal-dot dot-yellow"></span>
                                <span class="terminal-dot dot-green"></span>
                            </div>
                            <span class="panel-header-title" style="color:#00ff9d; font-weight:900;">EXECUTIVE_REPORT // VERIFIED_FINDINGS_DECK</span>
                            <span class="panel-header-badge badge-synthesis" style="background:#00ff9d; color:#06090f; font-weight:900;">REPORT VIEW</span>
                        </div>
                        """)
                        full_report_output = gr.Markdown(
                            """```bash
omnisearch@agent:~$ cat /var/out/EXECUTIVE_REPORT.md
+-------------------------------------------------------+
|  EXECUTIVE MISSION SYNTHESIS & FINDINGS REPORT        |
+-------------------------------------------------------+
[STATUS] Waiting for visual perception & web verification...
```""",
                            elem_classes=["full-report-body"]
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
            hud_stepper_output,
            event_logs_output,
            canvas_output,
            crops_gallery,
            thinking_output,
            web_output,
            answer_output,
            full_report_output
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
