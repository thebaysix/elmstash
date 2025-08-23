"""
Demo Launcher - Easy way to start the Elmstash demo UI.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit demo UI."""
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🔍 Starting Elmstash Demo UI...")
    print("=" * 50)
    print("This demo showcases the clean separation architecture:")
    print("🟢 Observer: Objective measurements")
    print("🟠 Evaluator: Subjective assessments") 
    print("🔴 Integration: Combined insights")
    print("=" * 50)
    print()
    print("The demo will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the demo.")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "demo_ui.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user.")
    except Exception as e:
        print(f"❌ Error starting demo: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check that all dependencies are installed: pip install -r requirements_demo.txt")
        print("3. Ensure you're in the project root directory")

if __name__ == "__main__":
    main()