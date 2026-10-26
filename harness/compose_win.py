"""Combine the OBS recording of a replay with the per-step panels into one video.

    compose_win.py [VIDEO] [--panels DIR] [--timeline FILE] [--out FILE] [--ffmpeg EXE]

VIDEO: the OBS recording (default: the newest .mkv/.mp4 in OBS's recording folder, the Videos
folder or the Desktop). Its start time comes from OBS's default file name ("YYYY-MM-DD
HH-MM-SS"), else the file's creation time (--start overrides both). timeline.jsonl
(run_replay_win.py) has the wall-clock time of every replay step; panel i is shown from --lead
seconds before step i's action is sent until the same point of step i+1, so the panel is up
while the game waits on that decision and while the chosen action plays out.

The result: the intro card, then the game scaled to 1280x720 at the top-left of a 1920x1080
canvas whose rest is the panel, cut --tail seconds after the game-over screen, then the outro
card. It is copied to the OneDrive folder, which uploads it.
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")


def obs_start(path):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[ _](\d{2})-(\d{2})-(\d{2})", os.path.basename(path))
    if m:
        return dt.datetime(*map(int, m.groups())).timestamp()
    return os.path.getctime(path)


def obs_dirs():
    dirs = []
    for ini in glob.glob(os.path.join(os.environ.get("APPDATA", ""), "obs-studio", "basic", "profiles", "*", "basic.ini")):
        for line in open(ini, encoding="utf-8", errors="replace"):
            m = re.match(r"\s*(?:FilePath|RecFilePath)\s*=\s*(.+)", line)
            if m:
                dirs.append(m.group(1).strip())
    return dirs + [os.path.join(HOME, "Videos"), os.path.join(HOME, "Desktop")]


def newest_video():
    vids = [p for d in obs_dirs() if os.path.isdir(d) for ext in ("mkv", "mp4", "mov", "flv")
            for p in glob.glob(os.path.join(d, f"*.{ext}"))]
    if not vids:
        raise SystemExit("no OBS recording found; pass its path")
    return max(vids, key=os.path.getmtime)


def probe(ffmpeg, path):
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace").stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    return dur, "Audio:" in out


def load_steps(path):
    """Step rows of the timeline (the last try of each step, if the replay was resumed) and the game-over time."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    last = {}
    for r in rows:
        if "i" in r:
            last[r["i"]] = r
    over = next((r["wall"] for r in rows if r.get("mark") == "game_over"), None)
    return [last[i] for i in sorted(last)], over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--panels", default=os.path.join(HERE, "panels"))
    ap.add_argument("--timeline", default=os.path.join(HERE, "timeline.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "sts2_a10_full_65_ai.mp4"))
    ap.add_argument("--ffmpeg", default=r"D:\APPS\5KPlayer\ffmpeg.exe")
    ap.add_argument("--onedrive", default=os.environ.get("OneDrive") or os.path.join(HOME, "OneDrive"))
    ap.add_argument("--start", type=float, default=None, help="recording start (unix time) if the file name has none")
    ap.add_argument("--lead", type=float, default=0.8, help="show a step's panel this long before its action is sent")
    ap.add_argument("--intro", type=float, default=6.0)
    ap.add_argument("--outro", type=float, default=8.0)
    ap.add_argument("--tail", type=float, default=10.0, help="keep this much of the recording after the game-over screen")
    args = ap.parse_args()
    video = args.video or newest_video()
    t0 = args.start if args.start is not None else obs_start(video)
    total, has_audio = probe(args.ffmpeg, video)
    steps, over = load_steps(args.timeline)
    if total is None:
        total = (over or steps[-1]["wall"]) - t0 + args.tail
    game = min(total, over - t0 + args.tail) if over else total     # the part of the recording that is used
    print(f"video {video}\n  starts {dt.datetime.fromtimestamp(t0)}, {total / 60:.1f} min (using {game / 60:.1f}); {len(steps)} steps")
    marks = [max(0.0, s["wall"] - t0 - args.lead) for s in steps]
    if steps[0]["wall"] - t0 < 0 or marks[-1] > game:
        raise SystemExit("the timeline does not fall inside the recording: check the start time (--start)")
    marks[0] = 0.0                                   # the first panel covers the run's start too
    entries = [("intro.png", args.intro)]
    for k, s in enumerate(steps):
        entries.append((f"step_{s['i']:04d}.png", (marks[k + 1] if k + 1 < len(steps) else game) - marks[k]))
    entries.append(("outro.png", args.outro))
    missing = [n for n, _ in entries if not os.path.exists(os.path.join(args.panels, n))]
    if missing:
        raise SystemExit(f"{len(missing)} panels missing in {args.panels}, e.g. {missing[0]}")
    listing = os.path.join(HERE, "panels.txt")
    with open(listing, "w", encoding="utf-8") as fh:
        for name, d in entries + [entries[-1][:1] + (None,)]:     # the concat demuxer needs the last file twice
            fh.write(f"file '{os.path.join(args.panels, name).replace(os.sep, '/')}'\n")
            if d is not None:
                fh.write(f"duration {max(d, 0.04):.3f}\n")
    length = args.intro + game + args.outro
    a = args.intro
    graph = (f"[1:v]fps=30,scale=1920:1080,format=yuv420p[bg];"
             f"[0:v]trim=0:{game:.3f},setpts=PTS-STARTPTS+{a}/TB,"
             f"scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[g];"
             f"[bg][g]overlay=0:0:eof_action=pass,format=yuv420p[v]")
    maps = ["-map", "[v]"]
    if has_audio:
        ms = int(a * 1000)
        graph += f";[0:a]atrim=0:{game:.3f},asetpts=PTS-STARTPTS,aformat=channel_layouts=stereo,adelay={ms}|{ms},apad[a]"
        maps += ["-map", "[a]", "-c:a", "aac", "-b:a", "160k"]
    # libx264: this ffmpeg (4.2) only knows NVENC's legacy presets, which current drivers reject
    cmd = [args.ffmpeg, "-hide_banner", "-y", "-i", video, "-f", "concat", "-safe", "0", "-i", listing,
           "-filter_complex", graph, *maps, "-c:v", "libx264", "-preset", "medium", "-crf", "21",
           "-t", f"{length:.2f}", "-movflags", "+faststart", args.out]
    print(" ".join(cmd), flush=True)
    if subprocess.call(cmd) != 0:
        raise SystemExit("ffmpeg failed")
    if os.path.isdir(args.onedrive):
        dst = os.path.join(args.onedrive, os.path.basename(args.out))
        shutil.copy2(args.out, dst)
        print("copied to OneDrive:", dst)
    else:
        print("no OneDrive folder at", args.onedrive)
    print("done:", args.out)


if __name__ == "__main__":
    main()
