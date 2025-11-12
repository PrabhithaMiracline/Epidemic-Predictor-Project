import subprocess
import threading
import time
import os
from pyngrok import ngrok

# ----------------------------
# PATH CONFIGURATION
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLASK_APP = os.path.join(SCRIPT_DIR, "location_app.py")
STREAMLIT_APP = os.path.join(SCRIPT_DIR, "streamlit_app.py")
SMS_ALERTS = os.path.join(SCRIPT_DIR, "sms_alerts.py")

# ----------------------------
# CREATE NGROK TUNNELS FIRST
# ----------------------------
print("🌐 Creating public tunnels via ngrok...")
try:
    ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN_HERE")  # required only once
except Exception:
    pass  # ignore if already set

# Create tunnels BEFORE launching apps
public_url = ngrok.connect(8501)  # Streamlit
gps_url = ngrok.connect(5000)     # Flask

print(f"✅ Public Streamlit URL: {public_url}")
print(f"✅ Public GPS URL: {gps_url}")
print("-----------------------------------------------------")

# ----------------------------
# FUNCTION TO RUN FLASK APP
# ----------------------------
def run_flask():
    print("🚀 Starting Flask location app...")
    subprocess.Popen(["python", FLASK_APP])
    time.sleep(3)
    print("✅ Flask server running locally on http://localhost:5000")

# ----------------------------
# FUNCTION TO RUN STREAMLIT APP
# ----------------------------
def run_streamlit():
    print("🚀 Launching Streamlit dashboard...")
    subprocess.run(["streamlit", "run", STREAMLIT_APP, "--server.port", "8501"])

# ----------------------------
# FUNCTION TO TRIGGER SMS ALERT
# ----------------------------
def trigger_sms_alerts():
    print("📤 Running SMS alerts...")
    os.system(f"python {SMS_ALERTS}")
    print("✅ SMS alert process completed.")

# ----------------------------
# MAIN ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    # Start Flask in background
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Wait for Flask to initialize
    time.sleep(4)

    # Run Streamlit app
    run_streamlit()
