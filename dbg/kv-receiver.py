#!/usr/bin/env python3
"""Simple KV cache receiver: saves POSTed data to /tmp/kv.bin"""

from http.server import HTTPServer, BaseHTTPRequestHandler

class KVHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/kv':
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(content_length)

        with open('/tmp/kv.bin', 'wb') as f:
            f.write(data)

        print(f"Received {len(data)} bytes, saved to /tmp/kv.bin")

        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress default logging

if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', 11655), KVHandler)
    print("KV receiver listening on 127.0.0.1:11655")
    server.serve_forever()
