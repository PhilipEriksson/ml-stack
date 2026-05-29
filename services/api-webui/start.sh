#!/usr/bin/env bash
set -e

# Start FastAPI proxy in background on port 8000
echo "Starting FastAPI proxy on port 8000..."
cd /opt/api && nohup uvicorn app:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &

# IPv6 -> IPv4 proxy so localhost (::1) works
nohup python3 -c "
import socket, http.client, threading

def handle(conn, data):
    parts = data.split(b'\r\n\r\n', 1)
    headers = parts[0].decode()
    body = parts[1] if len(parts) > 1 else b''
    first_line = headers.split('\n')[0]
    method, path, _ = first_line.split(' ', 2)
    s = http.client.HTTPConnection('127.0.0.1', 8000)
    s.request(method, path, body, {h.split(': ', 1)[0]: h.split(': ', 1)[1] for h in headers.split('\n')[1:] if h})
    r = s.getresponse()
    conn.sendall(f'HTTP/1.1 {r.status} {r.reason}\r\n'.encode())
    for k, v in r.getheaders():
        conn.sendall(f'{k}: {v}\r\n'.encode())
    conn.sendall(b'\r\n')
    conn.sendall(r.read())
    conn.close()
    s.close()

srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
srv.bind(('::1', 8000))
srv.listen(5)
while True:
    c, a = srv.accept()
    d = c.recv(65536)
    threading.Thread(target=handle, args=(c, d), daemon=True).start()
" > /tmp/api6.log 2>&1 &

# Set WebUI to point at our proxy
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-http://127.0.0.1:8000/v1}"

# Start Open WebUI - try the image's start script, fallback to uvicorn directly
echo "Starting Open WebUI..."
export PORT=8080
if [ -f /app/backend/start.sh ]; then
    exec /app/backend/start.sh
elif [ -f /app/backend/open_webui/main.py ]; then
    cd /app/backend && exec uvicorn open_webui.main:app --host 0.0.0.0 --port 8080
else
    echo "ERROR: Cannot find Open WebUI entry point"
    exit 1
fi
