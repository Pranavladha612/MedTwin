import os
import subprocess
import sys
import time

# Ensure emoji/unicode prints don't crash on Windows' default cp1252 console,
# both for this process and for the main.py / streamlit subprocesses below.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def launch_project():
    print("🚀 MEDTWIN: Starting Unified Pipeline...")

    # 1. Run the AI Generator
    print("\n[1/2] Executing AI Generation & Clinical Grounding...")
    subprocess.run([sys.executable, "main.py"])

    # 2. Small delay to ensure files are written
    time.sleep(2)

    # 3. Launch the Dashboard
    print("\n[2/2] Launching Clinical Dashboard...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])
    except KeyboardInterrupt:
        print("\n🛑 MedTwin System Shutting Down.")


if __name__ == "__main__":
    launch_project()
