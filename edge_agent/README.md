# edge_agent — the farm laptop

Reads the **Samsung A12** camera and microphone (app **IP Webcam**) over the farm Wi-Fi, or a **TP-Link Tapo C200**
by RTSP (see section 1b), and sends them to the AI service: 1–2 JPEG frames per second to `POST /ai/v1/frames` and ~1 s WAV clips to `POST /ai/v1/audio`, with the
`X-API-Key` header. It never stores frames or audio on disk.

```
A12 (IP Webcam) ──LAN──► laptop: python -m edge_agent ──HTTPS──► https://cuymonitor.duckdns.org/ai/...
   /shot.jpg  /audio.wav
```

How it behaves when something fails:

| Failure | What it does |
|---|---|
| The phone does not answer | Waits 1 s, 2 s, 4 s … up to 30 s and tries again |
| The Tapo stream stops or stalls | Restarts ffmpeg with the same backoff |
| No internet, or the service answers 5xx / 429 | Drops that frame or clip and retries with the same backoff; it never piles up old frames |
| The upload is slower than the microphone | Keeps at most 5 clips; the oldest one is dropped |
| 401 / 400 (wrong key or cage code) | Logs the error and tries again once a minute, until someone fixes `.env` |

La cámara se elige desde la app (página de la jaula → Cámara): el agente la lee del backend cada
15 s, cambia de cámara sin reiniciarse y reporta si la ve bien. `A12_URL` y `CAMERA_RTSP_URL` solo son
respaldo para cuando la app no tiene cámara. La sección 1b es la referencia técnica del RTSP.

## 1. Prepare the phone

1. Install **IP Webcam** (Pavel Khlebovich) on the A12 and give it camera and microphone permissions.
2. In *Video preferences*: resolution 640×480 (more is not needed and costs upload), quality 70.
3. In *Audio mode*: enabled. In *Power management*: keep the screen off but the app running.
4. Tap **Start server** and note the address on the screen, e.g. `http://192.168.1.50:8080`.
5. Give the phone a **fixed IP** in the router (DHCP reservation), so the address never changes.
6. From the laptop, open `http://<phone>:8080/shot.jpg` in a browser: you must see the cage.

## 1b. Alternative: TP-Link Tapo C200 (RTSP)

The Tapo C200 has no HTTP snapshot, only an RTSP stream, so the agent reads it with **ffmpeg**. Use it instead of
the phone: set `CAMERA_RTSP_URL` and leave `A12_URL` empty.

1. In the **Tapo app**, open the camera → *Device settings* → *Advanced settings* → *Camera account* (the menu names change with the app version) and create a
   user and password for the camera. This is **not** the TP-Link cloud login: the RTSP stream only accepts this
   camera account.
2. Put the camera and the laptop on the **same LAN** (the same router). The C200 only connects to **2.4 GHz** Wi-Fi.
3. Give the camera a **fixed IP** in the router (DHCP reservation) and note it, e.g. `192.168.1.60`.
4. Build the URL: `rtsp://USER:PASSWORD@192.168.1.60:554/stream2`. If the password has symbols like `@ : / # %`,
   write them percent-encoded (`@` becomes `%40`).
   - `stream1` is the high quality stream (1080p); `stream2` the low one (360p). Use **`stream2`**: the frames are
     smaller to upload and the agent only needs 1–2 per second.
5. Install **ffmpeg** on the laptop and check it with `ffmpeg -version` in a new terminal:
   - Windows: `winget install Gyan.FFmpeg`
   - Linux: `sudo apt install ffmpeg`
   - The Docker image does not include ffmpeg: on Docker, add `RUN apt-get install -y ffmpeg` to `edge_agent/Dockerfile`.
     Without ffmpeg the agent keeps running and the app shows that the laptop needs it.
6. The camera audio comes from the same stream. If the camera has no audio enabled, set `AUDIO_ENABLED=false`,
   otherwise the agent keeps retrying the audio and logs a warning every few seconds.

The log shows the URL with the password hidden (`rtsp://USER:***@192.168.1.60:554/stream2`). The password stays only
in `edge_agent/.env`.

## 2. Configure

```bash
cp edge_agent/config.example.env edge_agent/.env   # Windows: copy edge_agent\config.example.env edge_agent\.env
```

Fill in `AI_URL`, `API_KEY` (the same as `infra/.env` of the backend) and `CAGE_ID` (the code shown
in the dashboard, "Mis jaulas"). `.env` is never committed. `A12_URL` or `CAMERA_RTSP_URL` are optional fallbacks:
the camera itself is set up in the app.

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
