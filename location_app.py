from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# HTML page asks for location via browser GPS
HTML_PAGE = """
<!DOCTYPE html>
<html>
  <body style="font-family:sans-serif; text-align:center; margin-top:50px;">
    <h3>Fetching your live location…</h3>
    <script>
      navigator.geolocation.getCurrentPosition(
        pos => {
          fetch("/location", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              lat: pos.coords.latitude,
              lon: pos.coords.longitude
            })
          }).then(() => { document.body.innerHTML = "<h3>✅ Location received. You can close this tab.</h3>"; });
        },
        err => { document.body.innerHTML = "<h3>❌ Failed to get location.</h3>"; }
      );
    </script>
  </body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/location', methods=['POST'])
def receive_location():
    data = request.json
    print(f"📍 Live GPS Location: {data['lat']}, {data['lon']}")
    with open("user_location.txt", "w") as f:
        f.write(f"{data['lat']},{data['lon']}")
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(port=5000)
