"""
llama.cpp 流式代理 — 强制关闭 stream，兼容 Cherry Studio

用法: python3 llama-proxy.py
Cherry Studio 里填: http://<ip>:8119/v1
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
import logging

LLAMA_URL = "http://127.0.0.1:8118/v1"
PROXY_PORT = 8119

class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        req_data = json.loads(body)

        # 强制关闭 stream
        if req_data.get("stream", False):
            req_data["stream"] = False
            body = json.dumps(req_data).encode()
            logging.info(f"强制关闭 stream: {req_data.get('model', 'unknown')}")

        # 转发到 llama-server
        target = f"{LLAMA_URL}{self.path}"
        req = Request(target, data=body, headers={
            "Content-Type": "application/json",
        })
        resp = urlopen(req)
        resp_data = resp.read()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_data)))
        self.end_headers()
        self.wfile.write(resp_data)

    def do_GET(self):
        target = f"{LLAMA_URL}{self.path}"
        resp = urlopen(target)
        self.send_response(200)
        self.send_header("Content-Type", resp.headers.get("Content-Type", "text/plain"))
        self.end_headers()
        self.wfile.write(resp.read())

    def log_message(self, fmt, *args):
        logging.info(f"{self.client_address[0]} {fmt % args}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    server = HTTPServer(("0.0.0.0", PROXY_PORT), ProxyHandler)
    logging.info(f"代理启动: 0.0.0.0:{PROXY_PORT} -> llama-server:8118")
    server.serve_forever()
