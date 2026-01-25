import json
import os
import sys
import argparse

# Helper clamps
def clamp(v, a, b):
    return max(a, min(b, v))

# Deterministic mapping logic mirroring prompt.txt
def choose_detection(detections):
    priority = ['person', 'bicycle', 'car', 'truck']
    # try to find highest-priority detection
    for p in priority:
        for d in detections:
            if d.get('label') == p:
                return d
    # fallback to highest confidence
    best = None
    best_conf = -1.0
    for d in detections:
        c = d.get('confidence') or 0.0
        if c > best_conf:
            best = d
            best_conf = c
    return best

def compute_cmd_and_summary(det, width, height, ultrasonics=None):
    # defaults
    if det is None:
        # nothing detected -> stop
        return ('CMD L0 R0 T200', {"action":"stop","left":0,"right":0,"duration_ms":200,"reason":"no_detection"})

    bbox_norm = det.get('bbox_norm', {})
    cx = float(bbox_norm.get('cx', 0.5))
    w = float(bbox_norm.get('w', 0.0))
    conf = float(det.get('confidence', 0.8) or 0.8)
    label = det.get('label', 'object')

    # ultrasonic safety overrides
    if ultrasonics:
        l45 = ultrasonics.get('left45')
        r45 = ultrasonics.get('right45')
        ccen = ultrasonics.get('center')
        try:
            if ccen is not None and float(ccen) < 20.0:
                cmd = 'CMD L0 R0 T200'
                summary = {"action":"stop","left":0,"right":0,"duration_ms":200,"reason":"center obstacle"}
                return cmd, summary
            if l45 is not None and r45 is not None:
                if float(l45) < 20.0 and float(r45) < 20.0:
                    cmd = 'CMD L0 R0 T200'
                    summary = {"action":"stop","left":0,"right":0,"duration_ms":200,"reason":"both side obstacles"}
                    return cmd, summary
                if float(l45) < 20.0 and float(r45) >= 20.0:
                    # prefer turn right
                    action = 'turn_right'
                    duration = 300
                    base_speed = 120
                    left = clamp(base_speed + 40, 0, 255)
                    right = clamp(base_speed - 40, 0, 255)
                    cmd = f'CMD L{left} R{right} T{duration}'
                    summary = {"action":action,"left":left,"right":right,"duration_ms":duration,"reason":"left45 obstacle"}
                    return cmd, summary
                if float(r45) < 20.0 and float(l45) >= 20.0:
                    action = 'turn_left'
                    duration = 300
                    base_speed = 120
                    left = clamp(base_speed - 40, 0, 255)
                    right = clamp(base_speed + 40, 0, 255)
                    cmd = f'CMD L{left} R{right} T{duration}'
                    summary = {"action":action,"left":left,"right":right,"duration_ms":duration,"reason":"right45 obstacle"}
                    return cmd, summary
        except Exception:
            pass

    # proximity handling
    if w >= 0.5:
        cmd = 'CMD L0 R0 T200'
        summary = {"action":"stop","left":0,"right":0,"duration_ms":200,"reason":"very close"}
        return cmd, summary
    if w >= 0.35:
        # close -> short stop-and-turn strategy (choose stop)
        cmd = 'CMD L0 R0 T200'
        summary = {"action":"stop","left":0,"right":0,"duration_ms":200,"reason":"close object w>=0.35"}
        return cmd, summary

    # steering and base speed
    base_speed = int(round(160 * conf)) if conf is not None else 140
    base_speed = clamp(base_speed, 50, 220)
    steering_offset = int((0.5 - cx) * 2 * 80)
    left = clamp(base_speed - steering_offset, 0, 255)
    right = clamp(base_speed + steering_offset, 0, 255)

    # decide action type
    if 0.45 <= cx <= 0.55:
        action = 'forward'
        duration = 500
    elif cx < 0.45:
        action = 'turn_left'
        duration = 300
    else:
        action = 'turn_right'
        duration = 300

    cmd = f'CMD L{left} R{right} T{duration}'
    reason = f"{label} {'centered' if action=='forward' else ('left' if action=='turn_left' else 'right')}, cx={cx:.2f}, conf={conf:.2f}"
    summary = {"action":action,"left":left,"right":right,"duration_ms":duration,"reason":reason}
    return cmd, summary


def main():
    parser = argparse.ArgumentParser(description='Decide next robot action from result.json')
    parser.add_argument('--result', '-r', default='result.json', help='path to detection result JSON')
    parser.add_argument('--ultrasonic', '-u', help='optional JSON file with ultrasonics or inline JSON string')
    parser.add_argument('--write-prompt', '-w', action='store_true', help='write filled prompt file for qwen (prompt_filled.txt)')
    args = parser.parse_args()

    if not os.path.exists(args.result):
        print(f'Error: result file not found: {args.result}', file=sys.stderr)
        sys.exit(2)

    with open(args.result, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detections = data.get('detections', [])
    width = data.get('width')
    height = data.get('height')

    ultrasonics = None
    if args.ultrasonic:
        try:
            if os.path.exists(args.ultrasonic):
                with open(args.ultrasonic, 'r', encoding='utf-8') as uf:
                    ultrasonics = json.load(uf)
            else:
                ultrasonics = json.loads(args.ultrasonic)
        except Exception:
            ultrasonics = None

    chosen = choose_detection(detections)
    cmd, summary = compute_cmd_and_summary(chosen, width, height, ultrasonics=ultrasonics)

    # print two-line output
    print(cmd)
    print(json.dumps(summary, separators=(',', ':')))

    # optionally write filled prompt
    if args.write_prompt:
        prompt_path = os.path.join(os.path.dirname(args.result), 'prompt_filled.txt')
        # load base prompt
        base_prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.txt')
        try:
            with open(base_prompt_path, 'r', encoding='utf-8') as pf:
                base = pf.read()
        except Exception:
            base = ""
        filled = base.replace('<<DETECTIONS_JSON>>', json.dumps(data, indent=2))
        # if ultrasonics present, append their data to prompt
        if ultrasonics:
            filled += '\n\nUltrasonics provided to assistant:\n' + json.dumps(ultrasonics)
        with open(prompt_path, 'w', encoding='utf-8') as out:
            out.write(filled)
        print(f'Wrote filled prompt to: {prompt_path}', file=sys.stderr)

if __name__ == '__main__':
    main()
