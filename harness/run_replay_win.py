"""One-click real-game replay of a recorded engine run on a Windows PC (for an OBS recording).

    run_replay_win.bat            (double-click; it runs this with the project's Python)

1. installs the STS2MCP mod into the game's mods folder (any older copy is backed up);
2. sets up the non-Steam modded profile the replay needs (all epochs, Ironclad A10) under
   %APPDATA%/SlayTheSpire2/default/1/modded, backing up anything already there;
3. starts the game with --force-steam=off (no Steam, no cloud sync), walks the menus to
   Ironclad's character select;
4. waits for Enter (start the OBS recording first), then replays the run step by step
   (replay_real.py), logging each step's wall-clock time and frame to timeline.jsonl;
5. plays the ending (the final rewards, the Architect) to the game-over screen;
6. waits for Enter (stop the OBS recording first), then builds the video with the model's
   panels (compose_win.py) and copies it to OneDrive.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GAME = r"E:\SteamLibrary\steamapps\common\Slay the Spire 2"
EXE = os.path.join(GAME, "SlayTheSpire2.exe")
PROFILE = os.path.join(os.environ["APPDATA"], "SlayTheSpire2", "default", "1")
API = "http://localhost:15526/api/v1/singleplayer"
SEED = "full_65"
STAMP = time.strftime("%Y%m%d_%H%M%S")


def say(msg):
    print(msg, flush=True)


def backup(path):
    if os.path.exists(path):
        dst = f"{path}.bak_{STAMP}"
        shutil.copy2(path, dst)
        say(f"  backed up {path} -> {dst}")


def install_mod():
    mods = os.path.join(GAME, "mods")
    os.makedirs(mods, exist_ok=True)
    for f in ("STS2_MCP.dll", "STS2_MCP.json"):
        backup(os.path.join(mods, f))
        shutil.copy2(os.path.join(HERE, "mods", f), os.path.join(mods, f))
    say("mod installed")


def setup_profile():
    saves = os.path.join(PROFILE, "modded", "profile1", "saves")
    os.makedirs(saves, exist_ok=True)
    prog = os.path.join(saves, "progress.save")
    ok = False
    if os.path.exists(prog):
        try:
            d = json.load(open(prog, encoding="utf-8"))
            ok = any(c.get("id") == "CHARACTER.IRONCLAD" and c.get("max_ascension", 0) >= 10 for c in d.get("character_stats", []))
        except Exception:
            ok = False
    if not ok:
        for f in ("progress.save", "prefs.save"):
            backup(os.path.join(saves, f))
            shutil.copy2(os.path.join(HERE, "profile", f), os.path.join(saves, f))
    for f in ("settings.save", "settings.save.backup"):
        p = os.path.join(PROFILE, f)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            if (d.get("mod_settings") or {}).get("mods_enabled") is not True:
                backup(p)
                d["mod_settings"] = {"mods_enabled": True}
                json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
    say("profile ready (Ironclad A10, all unlocks, mods enabled)")


def get():
    with urllib.request.urlopen(API + "?format=json", timeout=10) as r:
        return json.load(r)


def post(action, **kw):
    req = urllib.request.Request(API, data=json.dumps({"action": action, **kw}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)
    say(f"  {action} {kw} -> {res.get('status')} {res.get('message', '')[:60]}")
    time.sleep(2.5)
    return res


def wait_api(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return get()
        except Exception:
            time.sleep(2)
    sys.exit("the mod API did not come up: is the game running with mods enabled?")


def to_character_select():
    for _ in range(30):
        st = get()
        t, screen = st.get("state_type"), st.get("menu_screen")
        opts = [o if isinstance(o, str) else o.get("name") for o in st.get("options") or []]
        if t == "menu" and screen == "character_select":
            post("menu_select", option="IRONCLAD")
            return
        if t == "menu" and "abandon_run" in opts:
            post("menu_select", option="abandon_run")
        elif t == "menu" and screen == "popup" and opts:
            post("menu_select", option=opts[0])          # e.g. confirm the abandon
        elif t == "menu" and screen == "tutorial":
            post("menu_select", option="advance")
        elif t == "menu" and "singleplayer" in opts:
            post("menu_select", option="singleplayer")
        elif t == "menu" and "standard" in opts:
            post("menu_select", option="standard")
        else:
            time.sleep(3)
    sys.exit(f"could not reach character select (state {get().get('state_type')})")


def mark(timeline, what):
    st = get()
    with open(timeline, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"mark": what, "frame": st.get("frame"), "wall": round(time.time(), 2), "state_type": st.get("state_type")}) + "\n")


def play_ending(timeline):
    for _ in range(20):
        st = get()
        t = st.get("state_type")
        if t == "game_over":
            mark(timeline, "game_over")
            return
        if t == "rewards":
            post("proceed")
        elif t == "event":
            ev = st.get("event") or {}
            if ev.get("in_dialogue"):
                post("advance_dialogue")
            else:
                post("choose_event_option", index=(ev.get("options") or [{}])[0].get("index", 0))
        else:
            time.sleep(3)


def main():
    resume = "--resume" in sys.argv
    timeline = os.path.join(HERE, "timeline.jsonl")
    if not resume:
        install_mod()
        setup_profile()
        if os.path.exists(timeline):
            os.rename(timeline, f"{timeline}.{STAMP}")
        say("starting the game (no Steam)...")
        subprocess.Popen([EXE, "--force-steam=off"], cwd=GAME)
        wait_api()
        time.sleep(5)
        to_character_select()
        say("")
        say("==> Start the OBS recording now, then press Enter to begin the replay.")
        input()
        mark(timeline, "start")
        argv = ["--from", "0", "--seed", SEED]
    else:
        argv = []   # continue from the checkpoint replay_real.py keeps
    rc = subprocess.call([sys.executable, os.path.join(HERE, "replay_real.py"), os.path.join(HERE, "states_fix6.jsonl"),
                          "--timeline", timeline, "--log", os.path.join(HERE, "replay.log")] + argv)
    if rc != 0:
        say(f"the replay stopped (exit {rc}); see replay.log. Fix, then run with --resume.")
        return
    play_ending(timeline)
    say("")
    say("==> Done: the run is won. Stop the OBS recording, then press Enter to build the video.")
    say(f"    timeline: {timeline}")
    input()
    time.sleep(3)          # let OBS finish writing the file
    subprocess.call([sys.executable, os.path.join(HERE, "compose_win.py"), "--timeline", timeline])


if __name__ == "__main__":
    main()
