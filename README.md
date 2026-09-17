# Dog Bark Detector (RTSP → YAMNet → MQTT)

Lightweight, self-hosted dog bark detector for a camera's audio stream.
No cloud, no browser — runs headless in Docker.

## How it works

1. `ffmpeg` pulls just the audio track from your camera's RTSP stream and
   converts it to raw 16kHz mono PCM.
2. The script reads it in 1-second chunks and runs each chunk through
   **YAMNet** (Google's audio event classifier, same model the
   dogbarkingdetector.com site uses in-browser) via `tflite-runtime` —
   a few MB, no GPU needed, fine on a Pi or a small VM.
3. When one of the dog-vocalization classes (Bark, Yip, Howl, Bow-wow,
   Growling, Whimper) crosses your confidence threshold, it publishes an
   MQTT message — pick it up in Home Assistant for logging, notifications,
   or automations.

The model and class-map CSV are downloaded automatically on first run and
cached in the `models/` volume, so restarts don't re-download them.

## Setup

1. Edit `docker-compose.yml`:
   - `RTSP_URL` — your camera's RTSP audio+video URL (audio track needs to
     exist; check your camera supports audio over RTSP — many cheap
     cameras disable it by default in their settings).
   - `MQTT_HOST` / `MQTT_USER` / `MQTT_PASS` — point at your existing
     Home Assistant / Mosquitto broker.
   - `THRESHOLD` — sensitivity, 0.0-1.0. 0.3 is the same default as the
     web tool; raise it if you get false positives, lower it if barks
     are missed.
   - `SOUND_KEYWORDS` — comma-separated substrings matched (case-insensitive)
     against YAMNet's ~520 class names. Default covers dog vocalizations
     (`bark,yip,howl,bow-wow,growling,whimper`); change or extend it to
     watch for other sounds too, e.g. `bark,whimper,smoke detector,glass`.

2. Build and run:
   ```bash
   docker compose up -d --build
   ```

3. Watch logs:
   ```bash
   docker compose logs -f
   ```

4. In Home Assistant, add an MQTT sensor/trigger listening on
   `dogbark/detected` (topic name configurable) to log events or fire
   notifications.

## Notes

- If your camera doesn't expose an audio track over RTSP, you'll need a
  separate USB/network mic near the dog instead — same script works, just
  swap the ffmpeg input for the mic device.
- CPU usage is minimal since inference only runs once per second on a
  1-second window, not continuously.
- `tflite-runtime` wheels are architecture-specific; the slim Python
  image + pip install should cover x86_64 and most ARM64 (Pi 4/5). Also
  note some prebuilt TFLite/LiteRT-family wheels assume AVX/AVX2 CPU
  instructions — very old x86_64 hardware (e.g. pre-2013 AMD/Intel low-power
  chips like the Turion II Neo in the HP MicroServer N54L) may lack this and
  crash immediately with exit code 132 (SIGILL). `tflite-runtime` is the
  safer choice for such hardware.
