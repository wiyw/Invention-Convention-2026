# =============================================================================
#  OmniScrub — main.py
#  Arduino App Lab Python script for Arduino Uno Q
#
#  Bricks (add all four in App Lab UI):
#    - video_object_detection   AIBrick camera + object detection
#    - web_ui_html              Dashboard + WebSocket on :7000
#    - dbstorage_sqlstore       SQLite — sessions, map_nodes, detections_log
#    - dbstorage_tsstore        InfluxDB — sensor readings over time
#
#  Place omniscrub.html in the assets/ folder of your App Lab project.
# =============================================================================

import json
import subprocess
import math
import time
import datetime

from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.video_objectdetection import VideoObjectDetection
from arduino.app_bricks.dbstorage_sqlstore import SQLStore
from arduino.app_bricks.dbstorage_tsstore import TimeSeriesStore

# --------------- Bricks ---------------
ui  = WebUI()
vod = VideoObjectDetection(confidence=0.55, debounce_sec=0.15)
sql = SQLStore("omniscrub.db")
ts  = TimeSeriesStore()

# --------------- Shared state ---------------
state = {
    "mode":          "idle",
    "speed":         50,
    "mop":           False,
    "vacuum":        False,
    "scoop":         "closed",
    "sensors":       {"f": 0, "l": 0, "r": 0},
    "detections":    [],
    "map_nodes":     [],
    "heading":       0.0,
    "pos":           [0.5, 0.5],
    "session_start": None,
    "session_id":    None,
    "area_cleaned":  0.0,
    "obstacles_hit": 0,
}

MAP_W, MAP_H = 40, 30
grid = [[0] * MAP_W for _ in range(MAP_H)]

# =============================================================================
# DATABASE SETUP
# =============================================================================

def init_database():
    sql.create_table("sessions", {
        "id":           "INTEGER PRIMARY KEY AUTOINCREMENT",
        "started_at":   "TEXT",
        "ended_at":     "TEXT",
        "duration_sec": "INTEGER",
        "area_m2":      "REAL",
        "obstacles":    "INTEGER",
    })
    sql.create_table("map_nodes", {
        "id":           "INTEGER PRIMARY KEY AUTOINCREMENT",
        "session_id":   "INTEGER",
        "x":            "REAL",
        "y":            "REAL",
        "heading":      "REAL",
        "f_cm":         "INTEGER",
        "l_cm":         "INTEGER",
        "r_cm":         "INTEGER",
        "objects_json": "TEXT",
        "ts":           "TEXT",
    })
    sql.create_table("detections_log", {
        "id":           "INTEGER PRIMARY KEY AUTOINCREMENT",
        "session_id":   "INTEGER",
        "ts":           "TEXT",
        "label":        "TEXT",
        "confidence":   "REAL",
        "bbox_json":    "TEXT",
    })
    print("[DB] Tables ready")

# =============================================================================
# BRIDGE CALLBACKS  (MCU -> Python)
# =============================================================================

def on_sensor_data(raw: str):
    """Called by the MCU sketch every ~200ms with fresh ultrasonic readings."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        state["sensors"] = data

        _update_map_from_sensors(data)

        # Push live data to dashboard
        ui.send_message("sensor_update", {
            "f": data.get("f", 0),
            "l": data.get("l", 0),
            "r": data.get("r", 0),
        })

        # Log to time-series DB
        ts_ms = int(datetime.datetime.now().timestamp() * 1000)
        ts.write_sample("front_cm", float(data.get("f", 0)), ts_ms)
        ts.write_sample("left_cm",  float(data.get("l", 0)), ts_ms)
        ts.write_sample("right_cm", float(data.get("r", 0)), ts_ms)

    except Exception as e:
        print(f"[sensor_data] {e}")

def on_servo_ack(msg: str):
    ui.send_message("servo_ack", {"msg": msg})

Bridge.provide("sensor_data", on_sensor_data)
Bridge.provide("servo_ack",   on_servo_ack)

# =============================================================================
# OBJECT DETECTION  (AIBrick -> Python)
# =============================================================================

def on_detection(detections: dict):
    """detections: dict keyed by label -> {"confidence": float, "bbox": [...]}"""
    state["detections"] = detections
    objects_this_frame = []

    for label, info in detections.items():
        obj = {
            "label":      label,
            "confidence": round(info.get("confidence", 0), 3),
            "bbox":       info.get("bbox", []),
        }
        objects_this_frame.append(obj)

        try:
            sql.store("detections_log", {
                "session_id": state["session_id"],
                "ts":         datetime.datetime.now().isoformat(),
                "label":      label,
                "confidence": obj["confidence"],
                "bbox_json":  json.dumps(obj["bbox"]),
            })
        except Exception as e:
            print(f"[detection log] {e}")

    if state["mode"] == "mapping" and objects_this_frame:
        _tag_map_node(objects_this_frame)

    ui.send_message("detections", {"detections": objects_this_frame})

vod.on_detect_all(on_detection)

# =============================================================================
# MAPPING
# =============================================================================

def _update_map_from_sensors(data):
    f  = data.get("f", 999)
    l  = data.get("l", 999)
    r  = data.get("r", 999)
    cx = int(state["pos"][0] * (MAP_W - 1))
    cy = int(state["pos"][1] * (MAP_H - 1))
    grid[cy][cx] = 1

    def _mark(dist_cm, angle_offset):
        angle_rad = math.radians(state["heading"] + angle_offset)
        cells = int(min(dist_cm, 200) / 10)
        for i in range(1, cells):
            gx = cx + int(math.cos(angle_rad) * i)
            gy = cy - int(math.sin(angle_rad) * i)
            if 0 <= gx < MAP_W and 0 <= gy < MAP_H and grid[gy][gx] == 0:
                grid[gy][gx] = 1
        wx = cx + int(math.cos(angle_rad) * cells)
        wy = cy - int(math.sin(angle_rad) * cells)
        if 0 <= wx < MAP_W and 0 <= wy < MAP_H and dist_cm < 150:
            grid[wy][wx] = 2

    _mark(f, 0); _mark(l, 90); _mark(r, -90)

    flat = [cell for row in grid for cell in row]
    ui.send_message("map_update", {
        "w": MAP_W, "h": MAP_H, "cells": flat,
        "pos": state["pos"], "heading": state["heading"],
    })

def _tag_map_node(objects):
    px, py = state["pos"]
    node = {
        "x":       round(px, 3),
        "y":       round(py, 3),
        "heading": round(state["heading"], 1),
        "f":       state["sensors"].get("f", 0),
        "l":       state["sensors"].get("l", 0),
        "r":       state["sensors"].get("r", 0),
        "objects": objects,
        "ts":      datetime.datetime.now().isoformat(),
    }
    state["map_nodes"].append(node)
    ui.send_message("map_node", node)

    try:
        sql.store("map_nodes", {
            "session_id":   state["session_id"],
            "x":            node["x"],
            "y":            node["y"],
            "heading":      node["heading"],
            "f_cm":         node["f"],
            "l_cm":         node["l"],
            "r_cm":         node["r"],
            "objects_json": json.dumps(objects),
            "ts":           node["ts"],
        })
    except Exception as e:
        print(f"[map_node save] {e}")

# =============================================================================
# COMMANDS  (Dashboard -> Python -> MCU)
# =============================================================================

def handle_cmd(sid, data):
    try:
        d = json.loads(data) if isinstance(data, str) else data
    except Exception:
        return

    action = d.get("type", d.get("action", ""))

    if action == "drive":
        direction = d.get("value", d.get("dir", "stop"))
        Bridge.call("cmd_drive", direction)
        step = 0.02
        ang  = math.radians(state["heading"])
        moves = {
            "forward":  ( math.cos(ang) * step, -math.sin(ang) * step),
            "backward": (-math.cos(ang) * step,  math.sin(ang) * step),
            "left":     (0.0, 0.0),
            "right":    (0.0, 0.0),
            "stop":     (0.0, 0.0),
        }
        dx, dy = moves.get(direction, (0.0, 0.0))
        state["pos"][0] = max(0.01, min(0.99, state["pos"][0] + dx))
        state["pos"][1] = max(0.01, min(0.99, state["pos"][1] + dy))
        if direction == "left":  state["heading"] = (state["heading"] + 15) % 360
        if direction == "right": state["heading"] = (state["heading"] - 15) % 360
        ui.send_message("pos_update", {
            "x": state["pos"][0] * 100,
            "y": state["pos"][1] * 100,
            "heading": state["heading"],
        })

    elif action == "speed":
        state["speed"] = int(d.get("value", 50))
        Bridge.call("cmd_speed", str(state["speed"]))

    elif action == "mop":
        state["mop"] = (d.get("value") == "on")
        Bridge.call("cmd_mop", "on" if state["mop"] else "off")

    elif action == "vacuum":
        state["vacuum"] = (d.get("value") == "on")
        Bridge.call("cmd_vacuum", "on" if state["vacuum"] else "off")

    elif action == "scoop":
        state["scoop"] = d.get("value", "close")
        Bridge.call("cmd_scoop", state["scoop"])

    elif action == "start_map":
        _start_session()
        state["mode"] = "mapping"
        state["map_nodes"] = []
        for row in grid:
            for i in range(len(row)): row[i] = 0
        ui.send_message("mode_change", {"mode": "mapping"})
        Bridge.call("cmd_drive", "forward")

    elif action == "stop_map":
        state["mode"] = "idle"
        Bridge.call("cmd_drive", "stop")
        flat = [cell for row in grid for cell in row]
        ui.send_message("map_update", {
            "nodes": state["map_nodes"],
            "w": MAP_W, "h": MAP_H, "cells": flat,
        })
        ui.send_message("mode_change", {"mode": "idle"})
        _end_session()

    elif action == "start_clean":
        _start_session()
        state["mode"] = "cleaning"
        state["area_cleaned"]  = 0.0
        state["obstacles_hit"] = 0
        Bridge.call("cmd_drive",  "forward")
        Bridge.call("cmd_vacuum", "on")
        Bridge.call("cmd_mop",    "on")
        ui.send_message("mode_change", {"mode": "cleaning"})

    elif action == "stop_clean":
        dur = int(time.time() - (state["session_start"] or time.time()))
        state["mode"] = "idle"
        Bridge.call("cmd_drive",  "stop")
        Bridge.call("cmd_vacuum", "off")
        Bridge.call("cmd_mop",    "off")
        ui.send_message("session_end", {
            "duration_sec": dur,
            "area_m2":      round(state["area_cleaned"], 2),
            "obstacles":    state["obstacles_hit"],
        })
        ui.send_message("mode_change", {"mode": "idle"})
        _end_session()

    elif action == "get_state":
        history = _load_session_history()
        sensor_history = _load_sensor_history(minutes=30)
        ui.send_message("full_state", {
            "mode":            state["mode"],
            "speed":           state["speed"],
            "mop":             state["mop"],
            "vacuum":          state["vacuum"],
            "scoop":           state["scoop"],
            "sensors":         state["sensors"],
            "pos":             state["pos"],
            "heading":         state["heading"],
            "session_history": history,
            "sensor_history":  sensor_history,
        })

ui.on_message("cmd", handle_cmd)

# =============================================================================
# SESSION HELPERS
# =============================================================================

def _start_session():
    state["session_start"] = time.time()
    state["area_cleaned"]  = 0.0
    state["obstacles_hit"] = 0
    now = datetime.datetime.now().isoformat()
    try:
        sql.store("sessions", {
            "started_at":   now,
            "duration_sec": 0,
            "area_m2":      0.0,
            "obstacles":    0,
        })
        rows = sql.read("sessions", order_by="id DESC", limit=1)
        state["session_id"] = rows[0]["id"] if rows else None
        print(f"[DB] Session started id={state['session_id']}")
    except Exception as e:
        print(f"[session start] {e}")

def _end_session():
    if not state["session_id"]:
        return
    dur = int(time.time() - (state["session_start"] or time.time()))
    now = datetime.datetime.now().isoformat()
    try:
        sql.store("sessions", {
            "id":           state["session_id"],
            "ended_at":     now,
            "duration_sec": dur,
            "area_m2":      round(state["area_cleaned"], 2),
            "obstacles":    state["obstacles_hit"],
        })
        print(f"[DB] Session {state['session_id']} closed — {dur}s")
    except Exception as e:
        print(f"[session end] {e}")
    state["session_id"] = None

def _load_session_history():
    try:
        rows = sql.read("sessions", order_by="id DESC", limit=50)
        return rows if rows else []
    except Exception as e:
        print(f"[load history] {e}")
        return []

def _load_sensor_history(minutes=30):
    try:
        start = f"-{minutes}m"
        front = ts.read_samples(measure="front_cm", start_from=start, aggr_window="10s", aggr_func="mean", limit=200)
        left  = ts.read_samples(measure="left_cm",  start_from=start, aggr_window="10s", aggr_func="mean", limit=200)
        right = ts.read_samples(measure="right_cm", start_from=start, aggr_window="10s", aggr_func="mean", limit=200)
        return {
            "front": [{"ts": s[1], "value": s[2]} for s in (front or [])],
            "left":  [{"ts": s[1], "value": s[2]} for s in (left  or [])],
            "right": [{"ts": s[1], "value": s[2]} for s in (right or [])],
        }
    except Exception as e:
        print(f"[load sensor history] {e}")
        return {}

# =============================================================================
# MAIN LOOP  (obstacle avoidance + area tracking, runs every ~300ms)
# =============================================================================

def main_loop():
    if state["mode"] != "cleaning":
        return

    f = state["sensors"].get("f", 999)
    l = state["sensors"].get("l", 999)
    r = state["sensors"].get("r", 999)

    if f < 25:
        state["obstacles_hit"] += 1
        Bridge.call("cmd_drive", "stop")
        time.sleep(0.2)
        if r >= l:
            Bridge.call("cmd_drive", "right")
            state["heading"] = (state["heading"] - 90) % 360
        else:
            Bridge.call("cmd_drive", "left")
            state["heading"] = (state["heading"] + 90) % 360
        time.sleep(0.5)
        Bridge.call("cmd_drive", "forward")
    else:
        state["area_cleaned"] = round(state["area_cleaned"] + 0.003, 3)
        ui.send_message("area_update", {"area_m2": state["area_cleaned"]})

    ang  = math.radians(state["heading"])
    step = 0.005
    state["pos"][0] = max(0.01, min(0.99, state["pos"][0] + math.cos(ang) * step))
    state["pos"][1] = max(0.01, min(0.99, state["pos"][1] - math.sin(ang) * step))
    ui.send_message("pos_update", {
        "x":       state["pos"][0] * 100,
        "y":       state["pos"][1] * 100,
        "heading": state["heading"],
    })

    time.sleep(0.3)

# =============================================================================
# RUN
# =============================================================================

init_database()

# Expose AIBrick camera stream (internal :4912) publicly on :4913
try:
    subprocess.Popen(
        ['socat', 'TCP-LISTEN:4913,fork,reuseaddr', 'TCP:127.0.0.1:4912'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print('[socat] AIBrick stream forwarded :4912 -> :4913')
except Exception as e:
    print(f'[socat] Could not start: {e}')

print("[OmniScrub] Starting — http://<UNO-Q-IP>:7000")
App.run(user_loop=main_loop)