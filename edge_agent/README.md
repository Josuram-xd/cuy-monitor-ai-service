# edge_agent — the farm laptop

Reads the **Samsung A12** camera and microphone (app **IP Webcam**) over the farm Wi-Fi and sends them to the AI
service: 1–2 JPEG frames per second to `POST /ai/frames` and ~1 s WAV clips to `POST /ai/audio`, with the
`X-API-Key` header. It never stores frames or audio on disk.

```
A12 (IP Webcam) ──LAN──► laptop: python -m edge_agent ──HTTPS──► https://cuymonitor.duckdns.org/ai/...
   /shot.jpg  /audio.wav
```

How it behaves when something fails:

| Failure | What it does |
|---|---|
| The phone does not answer | Waits 1 s, 2 s, 4 s … up to 30 s and tries again |
| No internet, or the service answers 5xx / 429 | Drops that frame or clip and retries with the same backoff; it never piles up old frames |
| The upload is slower than the microphone | Keeps at most 5 clips; the oldest one is dropped |
| 401 / 400 (wrong key or cage code) | Logs the error and tries again once a minute, until someone fixes `.env` |

## 1. Prepare the phone

1. Install **IP Webcam** (Pavel Khlebovich) on the A12 and give it camera and microphone permissions.
2. In *Video preferences*: resolution 640×480 (more is not needed and costs upload), quality 70.
3. In *Audio mode*: enabled. In *Power management*: keep the screen off but the app running.
4. Tap **Start server** and note the address on the screen, e.g. `http://192.168.1.50:8080`.
5. Give the phone a **fixed IP** in the router (DHCP reservation), so the address never changes.
6. From the laptop, open `http://<phone>:8080/shot.jpg` in a browser: you must see the cage.

## 2. Configure

```bash
cp edge_agent/config.example.env edge_agent/.env   # Windows: copy edge_agent\config.example.env edge_agent\.env
```

Fill in `A12_URL`, `AI_URL`, `API_KEY` (the same as `infra/.env` of the backend) and `CAGE_ID` (the code shown
in the dashboard, "Mis jaulas"). `.env` is never committed.

## 3. Install as a service

The agent needs Python 3.14 and two packages (`httpx`, `pydantic-settings`), not the whole AI service.

### Windows (NSSM)

```powershell
cd C:\cuy-monitor-ai-service
py -3.14 -m venv .venv-edge
.venv-edge\Scripts\pip install -r edge_agent\requirements.txt
.venv-edge\Scripts\python -m edge_agent          # try it: Ctrl+C after seeing "Edge agent for cage-1"

# NSSM (https://nssm.cc) runs it as a Windows service that starts with the laptop and restarts on failure
nssm install CuyEdgeAgent C:\cuy-monitor-ai-service\.venv-edge\Scripts\python.exe "-m edge_agent"
nssm set CuyEdgeAgent AppDirectory C:\cuy-monitor-ai-service
nssm set CuyEdgeAgent AppStdout C:\cuy-monitor-ai-service\edge-agent.log
nssm set CuyEdgeAgent AppStderr C:\cuy-monitor-ai-service\edge-agent.log
nssm set CuyEdgeAgent AppRotateFiles 1
nssm set CuyEdgeAgent AppRotateBytes 5000000
nssm start CuyEdgeAgent
```

Without NSSM: *Task Scheduler → Create task*, trigger "At startup", action
`C:\cuy-monitor-ai-service\.venv-edge\Scripts\python.exe` with arguments `-m edge_agent` and *Start in*
`C:\cuy-monitor-ai-service`; in *Settings* tick "If the task fails, restart every 1 minute".

Keep the laptop awake: *Settings → System → Power* → screen off when plugged in "Never" is not needed, but
**Sleep: Never** is. Also turn off "Allow the computer to turn off this device to save power" for the Wi-Fi
adapter in Device Manager.

### Linux (systemd)

```bash
sudo useradd --system --home /opt/cuy-monitor-ai-service cuymonitor
sudo git clone https://github.com/Josuram-xd/cuy-monitor-ai-service.git /opt/cuy-monitor-ai-service
cd /opt/cuy-monitor-ai-service
sudo python3.14 -m venv .venv-edge && sudo .venv-edge/bin/pip install -r edge_agent/requirements.txt
sudo cp edge_agent/config.example.env edge_agent/.env && sudo nano edge_agent/.env
sudo chown -R cuymonitor: /opt/cuy-monitor-ai-service && sudo chmod 600 edge_agent/.env
sudo cp edge_agent/cuy-edge-agent.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now cuy-edge-agent
journalctl -u cuy-edge-agent -f      # the log
# never suspend: the camera stops being read
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

### Linux with Docker (optional)

```bash
docker build -f edge_agent/Dockerfile -t cuy-edge-agent .
docker run -d --restart unless-stopped --env-file edge_agent/.env --name cuy-edge-agent cuy-edge-agent
```

## 4. Check it works

- The log says `Edge agent for cage-1: 1.0 fps ...` and no errors.
- In the EC2: `docker compose logs -f ai-service` shows the frames and clips arriving.
- `https://cuymonitor.duckdns.org/ai/health` answers, and the cage in the dashboard changes within a minute.
