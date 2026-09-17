#!/usr/bin/env python3
"""
Lightweight dog bark detector for an RTSP camera audio stream.

Pipeline:
  RTSP stream --ffmpeg--> raw 16kHz mono PCM --> YAMNet (tflite) --> MQTT event

Env vars (see docker-compose.yml):
  RTSP_URL        rtsp://user:pass@camera-ip:554/stream1
  MQTT_HOST       e.g. homeassistant.local
  MQTT_PORT       default 1883
  MQTT_TOPIC      default "dogbark/detected"
  MQTT_USER / MQTT_PASS   optional
  THRESHOLD       confidence 0.0-1.0, default 0.3 (matches the site's default)
  COOLDOWN_SEC    minimum seconds between repeated log/publish events, default 5
"""

import csv
import io
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np
import paho.mqtt.client as mqtt
import ai_edge_litert.interpreter as tflite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bark_detector")
if hasattr(time, "tzset"):
    time.tzset()  # picks up the TZ env var (e.g. Europe/Amsterdam) immediately

MODEL_URL = "https://huggingface.co/thelou1s/yamnet/resolve/main/lite-model_yamnet_classification_tflite_1.tflite"
CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

# Sounds we care about (case-insensitive substring match against YAMNet's
# display names) -- mirrors the classes the browser tool highlights.
# Overridable via the SOUND_KEYWORDS env var (comma-separated), e.g.
# "bark,yip,howl,bow-wow,growling,whimper,cat,meow"
DEFAULT_SOUND_KEYWORDS = "bark,yip,howl,bow-wow,growling,whimper"

SAMPLE_RATE = 16000
CHUNK_SECONDS = 1
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

MODEL_PATH = "/app/models/yamnet.tflite"
CLASS_MAP_PATH = "/app/models/yamnet_class_map.csv"


def ensure_model_files():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        log.info("Downloading YAMNet tflite model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not os.path.exists(CLASS_MAP_PATH):
        log.info("Downloading YAMNet class map...")
        urllib.request.urlretrieve(CLASS_MAP_URL, CLASS_MAP_PATH)


def load_class_names():
    names = []
    with open(CLASS_MAP_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["display_name"])
    return names


def dog_sound_indices(class_names, keywords):
    return {
        i
        for i, name in enumerate(class_names)
        if any(k in name.lower() for k in keywords)
    }


def start_ffmpeg(rtsp_url):
    """Spawn ffmpeg pulling audio only from the RTSP stream as raw s16le PCM."""
    cmd = [
        "ffmpeg",
        "-loglevel", "panic",
        "-nostats",
        "-fflags", "+igndts",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vn",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-f", "s16le",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=CHUNK_SAMPLES * 2)


def read_chunk(proc):
    raw = proc.stdout.read(CHUNK_SAMPLES * 2)  # 2 bytes per int16 sample
    if len(raw) < CHUNK_SAMPLES * 2:
        return None
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def main():
    rtsp_url = os.environ["RTSP_URL"]
    mqtt_host = os.environ.get("MQTT_HOST", "localhost")
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    mqtt_topic = os.environ.get("MQTT_TOPIC", "dogbark/detected")
    mqtt_user = os.environ.get("MQTT_USER")
    mqtt_pass = os.environ.get("MQTT_PASS")
    threshold = float(os.environ.get("THRESHOLD", "0.3"))
    cooldown = float(os.environ.get("COOLDOWN_SEC", "5"))
    keywords_raw = os.environ.get("SOUND_KEYWORDS", DEFAULT_SOUND_KEYWORDS)
    keywords = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]

    ensure_model_files()
    class_names = load_class_names()
    target_idx = dog_sound_indices(class_names, keywords)
    log.info(f"Sound keywords: {keywords}")
    log.info(f"Watching for classes: {[class_names[i] for i in target_idx]}")
    if not target_idx:
        log.error("SOUND_KEYWORDS matched no YAMNet classes - check spelling/env var")
        sys.exit(1)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    client = mqtt.Client()
    if mqtt_user:
        client.username_pw_set(mqtt_user, mqtt_pass)
    client.connect(mqtt_host, mqtt_port, 60)
    client.loop_start()

    last_event = 0.0

    while True:
        proc = start_ffmpeg(rtsp_url)
        log.info("ffmpeg started, listening...")
        try:
            while True:
                audio = read_chunk(proc)
                if audio is None:
                    log.warning("Stream ended/broken, reconnecting...")
                    break

                interpreter.resize_tensor_input(input_details[0]["index"], [len(audio)])
                interpreter.allocate_tensors()
                interpreter.set_tensor(input_details[0]["index"], audio)
                interpreter.invoke()
                scores = interpreter.get_tensor(output_details[0]["index"])
                mean_scores = scores.mean(axis=0)

                best_idx = max(target_idx, key=lambda i: mean_scores[i])
                confidence = float(mean_scores[best_idx])

                if confidence >= threshold:
                    now = time.time()
                    if now - last_event >= cooldown:
                        last_event = now
                        label = class_names[best_idx]
                        payload = json.dumps({
                            "sound": label,
                            "confidence": round(confidence, 3),
                            "ts": int(now),
                        })
                        log.info(f"DETECTED: {label} ({confidence:.2f})")
                        client.publish(mqtt_topic, payload)
        finally:
            proc.kill()
            time.sleep(2)  # brief pause before reconnecting


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
