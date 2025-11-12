import pandas as pd
from twilio.rest import Client
import requests
from geopy.distance import geodesic
import os
import sys

# Fix encoding on Windows terminals
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve correct path for user_location.txt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USER_LOC_FILE = os.path.join(SCRIPT_DIR, "..", "user_location.txt")
from geopy.distance import geodesic

# --- Twilio credentials (replace with your actual ones) ---
ACCOUNT_SID = "AC343e2d0971ea2dd239ed7a039c59b876"
AUTH_TOKEN = "267bd77bfb666c25106ac00380e43735"
TWILIO_PHONE = "+16182668569"   # your Twilio sender number

# --- Recipient lists ---
PUBLIC_NUMBERS = ["+918939875071"]     # public awareness numbers

# --- Diseases to check ---
diseases = ["dengue", "malaria", "cholera"]

# --- Auto-detect user location ---
def get_user_location():
    # First, try to read from user_location.txt (GPS-based)
    try:
        with open("user_location.txt", "r") as f:
            lat, lon = map(float, f.read().strip().split(','))
            print(f"📍 Using GPS location from file: {lat}, {lon}")
            return lat, lon
    except (FileNotFoundError, ValueError):
        print("⚠️ No GPS location file found. Falling back to IP geolocation.")

    # Fallback to IP-based geolocation
    try:
        response = requests.get('http://ip-api.com/json/')
        data = response.json()
        if data['status'] == 'success':
            return data['lat'], data['lon']
        else:
            raise Exception("Geolocation failed")
    except Exception as e:
        print(f"⚠️ Geolocation failed: {e}. Using default location.")
        return 20.0, 79.0  # Default lat/lon near Gadchiroli

user_lat, user_lon = get_user_location()

# --- Find nearest district ---
try:
    districts_df = pd.read_csv("epidemic-predictor-project/out_multi/dengue/resource_allocation_report.csv")[['district', 'latitude', 'longitude']].drop_duplicates()
    districts_df['distance'] = districts_df.apply(
        lambda row: geodesic((user_lat, user_lon), (row['latitude'], row['longitude'])).km, axis=1
    )
    nearest_district = districts_df.loc[districts_df['distance'].idxmin(), 'district']
    print(f"📍 Nearest district based on location: {nearest_district}")
except FileNotFoundError:
    nearest_district = "Gadchiroli"  # Fallback
    print(f"⚠️ No district data found. Using default district: {nearest_district}")

# --- Initialize Twilio client ---
client = Client(ACCOUNT_SID, AUTH_TOKEN)
any_alerts = False

for disease in diseases:
    hospital_info = None
    send_alert = False

    # First, check for outbreak in predictions
    try:
        predictions_file = f"epidemic-predictor-project/out_multi/{disease}/all_predictions_full.csv"
        predictions_df = pd.read_csv(predictions_file)

        # Debug: show available columns
        print(f"📄 Columns in {disease} file:", list(predictions_df.columns))

        # Determine outbreak condition dynamically
        if 'outbreak_flag' in predictions_df.columns:
            outbreak_rows = predictions_df[
                (predictions_df['district'] == nearest_district) &
                (predictions_df['outbreak_flag'] == True)
            ]
        elif 'risk_level' in predictions_df.columns:
            outbreak_rows = predictions_df[
                (predictions_df['district'] == nearest_district) &
                (predictions_df['risk_level'].str.lower().isin(['high', 'severe', 'critical']))
            ]
        elif 'predicted_cases' in predictions_df.columns:
            outbreak_rows = predictions_df[
                (predictions_df['district'] == nearest_district) &
                (predictions_df['predicted_cases'] > 10)  # threshold
            ]
        else:
            print(f"⚠️ No recognizable outbreak indicator in {disease} file.")
            outbreak_rows = pd.DataFrame()

        if not outbreak_rows.empty:
            send_alert = True
            # Now, try to load hospital info
            try:
                hospitals_file = f"epidemic-predictor-project/out_multi/{disease}/hotspots_with_nearest_hospitals.csv"
                hospitals_df = pd.read_csv(hospitals_file)
                nearest_hospital = hospitals_df[hospitals_df['district'] == nearest_district]
                if not nearest_hospital.empty:
                    hosp_row = nearest_hospital.iloc[0]
                    hospital_info = f"Nearest hospital: {hosp_row['hospital_name']} at {hosp_row.get('address', 'N/A')}, Distance: {hosp_row['distance_km']} km."
                else:
                    hospital_info = "Potential outbreak detected. Please visit your nearest healthcare facility."
            except FileNotFoundError:
                hospital_info = "Potential outbreak detected. Please visit your nearest healthcare facility."
        else:
            print(f"✅ No outbreak predicted for {disease} in {nearest_district}.")
    except FileNotFoundError:
        print(f"⚠️ No predictions file found for {disease}. Skipping.")

    if send_alert:
        any_alerts = True
        public_msg = (
            f"⚠️ Health Advisory: Check for {disease} symptoms in {nearest_district}.\n"
            # f"Stay safe. Use preventive measures and visit your nearest hospital if symptoms appear.\n"
            f"{hospital_info}"
        )

        for num in PUBLIC_NUMBERS:
            # Dry-run: Comment out actual SMS sending for testing
            try:
                message = client.messages.create(
                    body=public_msg,
                    from_=TWILIO_PHONE,
                    to=num
                )
                print(f"✅ Public advisory sent for {disease} - {nearest_district} - SID: {message.sid}")
            except Exception as e:
                print(f"⚠️ Failed to send SMS for {disease} - {nearest_district}: {e}")
            print(f"📤 [DRY-RUN] Would send to {num}: {public_msg}")

            # Send WhatsApp message
            # try:
            #     whatsapp_message = client.messages.create(
            #         from_='whatsapp:+14155238886',  # Twilio Sandbox number
            #         body=f'⚠️ Health Alert: {disease.capitalize()} risk detected in {nearest_district}.',
            #         to=f'whatsapp:+918939875071'
            #     )
            #     print(f"✅ WhatsApp message sent for {disease} - {nearest_district} - SID: {whatsapp_message.sid}")
            # except Exception as e:
            #     print(f"⚠️ Failed to send WhatsApp message for {disease} - {nearest_district}: {e}")
    else:
        print(f"✅ No alert needed for {disease} in {nearest_district}.")

if not any_alerts:
    print(f"\n✅ No nearest hospital data found for {nearest_district} in any disease.")
