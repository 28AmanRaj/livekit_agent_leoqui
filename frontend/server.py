import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
from dotenv import load_dotenv
from livekit import api

# Load env variables from root directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, "..", ".env")
load_dotenv(dotenv_path)

class LiveKitHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/token':
            # Generate a token for the user "student" to join "tutor-room"
            token = api.AccessToken(
                os.getenv("LIVEKIT_API_KEY"),
                os.getenv("LIVEKIT_API_SECRET")
            )
            token.with_identity("student")
            token.with_name("Student")
            import uuid
            room_name = f"tutor-room-{uuid.uuid4().hex[:8]}"
            token.with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
            ))
            
            jwt_token = token.to_jwt()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "token": jwt_token,
                "url": os.getenv("LIVEKIT_URL")
            }
            self.wfile.write(json.dumps(response).encode())
            return
            
        # Default behavior serves files
        return super().do_GET()

if __name__ == '__main__':
    port = 8000
    # Change directory to the frontend folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Starting server at http://localhost:{port}")
    print("Serving frontend for LiveKit Agent Testing...")
    server = HTTPServer(('localhost', port), LiveKitHandler)
    server.serve_forever()
