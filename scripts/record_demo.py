#!/usr/bin/env python3
import os
import time
import shutil
import subprocess
from playwright.sync_api import sync_playwright

def main():
    frames_dir = os.path.abspath("frames")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    img_path = os.path.abspath("sample_images/landmark_colosseum.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")

    print(f"[INFO] Starting demo capture with Playwright...")
    print(f"[INFO] Image: {img_path}")

    frame_idx = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
        # Use 1366x768 crisp standard viewport
        page = browser.new_page(viewport={"width": 1366, "height": 768})

        print("[INFO] Navigating to http://localhost:7860...")
        page.goto("http://localhost:7860")
        page.wait_for_timeout(4000)

        # 1. Fill input image
        print("[INFO] Uploading test image...")
        file_input = page.locator("input[type=file]").first
        file_input.set_input_files(img_path)
        page.wait_for_timeout(1500)

        # 2. Fill query prompt
        print("[INFO] Setting query prompt...")
        textarea = page.locator("textarea").first
        textarea.fill("what this structure is used for?")
        page.wait_for_timeout(1000)

        # Capture 3 frames of initial ready state
        for _ in range(3):
            page.screenshot(path=f"{frames_dir}/frame_{frame_idx:04d}.png")
            frame_idx += 1
            time.sleep(0.3)

        # 3. Click execute
        print("[INFO] Dispatching [EXECUTE MISSION]...")
        exec_btn = page.locator(".term-btn-primary").first
        exec_btn.click()

        # 4. Recording loop
        start_time = time.time()
        max_duration = 45.0  # max 45s safety limit
        is_complete = False
        consecutive_complete_count = 0

        print("[INFO] Recording mission frames...")
        while time.time() - start_time < max_duration:
            page.screenshot(path=f"{frames_dir}/frame_{frame_idx:04d}.png")
            frame_idx += 1

            # Check if mission has completed
            hud_text = page.locator(".hud-stepper-box").inner_text() if page.locator(".hud-stepper-box").count() > 0 else ""
            if "COMPLETE" in hud_text or "100%" in hud_text or "REPORT READY" in hud_text:
                consecutive_complete_count += 1
                if consecutive_complete_count >= 5:  # ensure it stays complete and rendered
                    print(f"[INFO] Mission concluded. Frame count: {frame_idx}")
                    is_complete = True
                    break
            else:
                consecutive_complete_count = 0

            time.sleep(0.4)

        # Capture 6 frames of final parallel view
        print("[INFO] Capturing final cockpit view...")
        for _ in range(6):
            page.screenshot(path=f"{frames_dir}/frame_{frame_idx:04d}.png")
            frame_idx += 1
            time.sleep(0.3)

        # Switch to FULL REPORT tab to showcase executive deck
        print("[INFO] Capturing FULL REPORT VIEW tab...")
        try:
            full_tab = page.get_by_role("tab", name="FULL REPORT VIEW")
            if full_tab.count() > 0:
                full_tab.click()
                time.sleep(0.8)
                for _ in range(6):
                    page.screenshot(path=f"{frames_dir}/frame_{frame_idx:04d}.png")
                    frame_idx += 1
                    time.sleep(0.3)
        except Exception as e:
            print(f"Tab switch note: {e}")

        browser.close()

    print(f"[INFO] Total frames captured: {frame_idx}.")

    # 5. Convert frames to GIF using ffmpeg
    gif_path = os.path.abspath("assets/demo.gif")
    mp4_path = os.path.abspath("assets/demo.mp4")
    print(f"[INFO] Encoding GIF: {gif_path}...")

    # High quality palette-optimized GIF with smooth framerate and scaled resolution
    cmd_gif = [
        "ffmpeg", "-y",
        "-framerate", "3",
        "-i", f"{frames_dir}/frame_%04d.png",
        "-vf", "fps=3,scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0",
        gif_path
    ]
    subprocess.run(cmd_gif, check=True)

    # Also generate MP4 for video preview
    print(f"[INFO] Encoding MP4: {mp4_path}...")
    cmd_mp4 = [
        "ffmpeg", "-y",
        "-framerate", "3",
        "-i", f"{frames_dir}/frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=1200:-2",
        mp4_path
    ]
    subprocess.run(cmd_mp4, check=True)

    gif_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    mp4_size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
    print(f"[SUCCESS] GIF created: {gif_path} ({gif_size_mb:.2f} MB)")
    print(f"[SUCCESS] MP4 created: {mp4_path} ({mp4_size_mb:.2f} MB)")

    # Cleanup frames
    shutil.rmtree(frames_dir)
    print("[INFO] Intermediate frames cleaned up.")

if __name__ == "__main__":
    main()
