"""The tournament director. Keeps the trainers alive (restart + resume after a crash), pins each to its CPU share,
runs a ranking round every 30 minutes, applies the catch-up rule (pause a leader > PAUSE_GAP ahead, resume below
RESUME_GAP), knocks out the weakest at the scheduled times, and runs the final.
State: results/tournament.json. Events: runs/tournament/events.jsonl (shown on the site).
    setsid nohup .venv/bin/python -m tournament.supervisor > logs/supervisor.log 2>&1 &"""
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback

from tron import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TDIR = os.path.join(ROOT, "runs", "tournament")
STATE_P = os.path.join(RES, "tournament.json")
PY = os.path.join(ROOT, ".venv", "bin", "python")
IDS = [m["id"] for m in config.CONTESTANTS]
LBL = {m["id"]: m["label"] for m in config.CONTESTANTS + config.BOTS}
FIRST_ROUND_AFTER_MIN = 20


def now():
    return dt.datetime.now().astimezone()


def stamp():
    return time.strftime("%Y-%m-%d %H:%M")


def event(kind, title, text="", **kw):
    os.makedirs(TDIR, exist_ok=True)
    with open(os.path.join(TDIR, "events.jsonl"), "a") as f:
        f.write(json.dumps({"t": stamp(), "kind": kind, "title": title, "text": text, **kw}) + "\n")
    print(stamp(), kind, title, flush=True)


def load_state():
    if os.path.exists(STATE_P):
        s = json.load(open(STATE_P))
        if "started" in s:
            return s
    return {"started": now().isoformat(), "round": 0, "next_round_at": time.time() + FIRST_ROUND_AFTER_MIN * 60,
            "models": {m: {"status": "training"} for m in IDS}, "pauses": [], "knockouts_done": 0,
            "final_done": False, "leader": None, "restarts": {}, "note": ""}


def save_state(s):
    tmp = STATE_P + ".tmp"
    json.dump(s, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE_P)


def active(s):
    return [m for m in IDS if s["models"][m]["status"] in ("training", "paused")]


def cpu_plan(n):
    """cores per model and the cpu set for each, plus cpus for ranking rounds."""
    if n >= 5:
        per, rank = 1, "5-7"
    elif n == 4:
        per, rank = 2, "0-7"
    elif n == 3:
        per, rank = 2, "6-7"
    else:                      # the duel: PPO and Mini-AlphaZero, 4 cores each; rounds run at low priority on all
        per, rank = 4, "0-7"
    return per, rank


def pgrep(pattern):
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]


def trainer_pattern(m):
    # anchored on the full interpreter path: a pattern starting with "-m" is read by pgrep as an option
    return f"^{PY} -m models\\.{m} --run {m}( |$)"


def trainer_pids(m):
    """Main trainer processes only: worker processes forked by a trainer (same command line) are left out."""
    pids = pgrep(trainer_pattern(m))
    out = []
    for p in pids:
        try:
            ppid = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid not in pids:
            out.append(p)
    return out


def trainer_alive(m):
    return bool(trainer_pids(m))


def start_trainer(s, m, cpus, cores):
    resume = os.path.exists(os.path.join(ROOT, "runs", m, "latest.json"))
    cmd = ["taskset", "-c", cpus, PY, "-m", f"models.{m}", "--run", m, "--cores", str(cores)] + (["--resume"] if resume else [])
    log = open(os.path.join(ROOT, "logs", f"{m}.log"), "a")
    env = dict(os.environ, PYTHONPATH=ROOT, NUMBA_NUM_THREADS=str(cores), OMP_NUM_THREADS=str(cores))
    subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    s["models"][m].update({"cpus": cpus, "cores": cores, "started_at": time.time()})


def stop_trainer(m, wait=120):
    for p in pgrep(trainer_pattern(m)):
        try:
            os.kill(p, 15)
        except ProcessLookupError:
            pass
    t = time.time()
    while trainer_alive(m) and time.time() - t < wait:
        time.sleep(2)
    for p in pgrep(trainer_pattern(m)):
        os.kill(p, 9)


def assign(s):
    """CPU sets for the active models (deterministic: in contestant order)."""
    act = active(s)
    per, _ = cpu_plan(len(act))
    out = {}
    for i, m in enumerate(act):
        lo = i * per
        out[m] = f"{lo}" if per == 1 else f"{lo}-{lo + per - 1}"
    return out, per


def train_text(m):
    p = os.path.join(ROOT, "runs", m, "stats.csv")
    if not os.path.exists(p):
        return "starting"
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "starting"
    r = rows[-1]

    def k(v):
        v = float(v)
        return f"{v / 1e6:.1f} M" if v >= 1e6 else f"{v / 1e3:.0f} k" if v >= 1e3 else f"{v:.0f}"
    if "steps" in r and r.get("steps"):
        return f"{k(r['steps'])} steps · {k(r['games'])} games"
    if "gen" in r:
        return f"generation {r['gen']} · {k(r['games'])} games"
    return f"{k(r['games'])} self-play games"


def history():
    p = os.path.join(RES, "rating_history.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def avg_rating(m, n):
    rs = [h["ratings"][m] for h in history() if m in h["ratings"]][-n:]
    return sum(rs) / len(rs) if rs else None


def ensure_trainers(s):
    plan, per = assign(s)
    for m in active(s):
        st = s["models"][m]
        pids = trainer_pids(m)
        if len(pids) > 1:                      # never two trainers writing the same run
            for p in sorted(pids)[1:]:
                os.kill(p, 9)
            event("error", f"Two {LBL[m]} trainers were running; the newer one was stopped", model=m)
        if pids:
            continue
        if st.get("started_at") and time.time() - st["started_at"] < 90:
            continue                           # just started: give it time to show up
        marker = os.path.join(ROOT, "runs", m, "RESTART")
        if os.path.exists(marker):              # a deliberate restart (new code or settings): not a crash
            note = open(marker).read().strip()
            os.remove(marker)
            event("restart", f"{LBL[m]} restarted on purpose", note, model=m)
        elif st.get("started_at") and time.time() - st["started_at"] > 30:
            tail = ""
            try:
                tail = "".join(open(os.path.join(ROOT, "logs", f"{m}.log")).readlines()[-6:])[-800:]
            except OSError:
                pass
            n = s["restarts"].get(m, 0) + 1
            s["restarts"][m] = n
            event("crash", f"{LBL[m]} stopped unexpectedly and was restarted from its last checkpoint (restart {n})", tail, model=m)
        start_trainer(s, m, plan[m], per)


def rebalance(s, why):
    """Restart every active trainer on its new CPU share (they resume from their checkpoints)."""
    plan, per = assign(s)
    for m in active(s):
        stop_trainer(m)
        start_trainer(s, m, plan[m], per)
    event("cores", f"CPU shares changed: {per} core(s) per model", why)


def run_round(s, final=False):
    s["round"] += 0 if final else 1
    act = active(s) if not final else [m for m in IDS if s["models"][m]["status"] == "finalist"]
    per, rank_cpus = cpu_plan(len(active(s)))
    procs = 8 if final else (3 if len(act) >= 5 else 2)
    cmd = ["nice", "-n", "10", "taskset", "-c", "0-7" if final else rank_cpus, PY, "-m", "tournament.ranking",
           "--models", ",".join(act), "--procs", str(procs)]
    cmd += (["--final", "--games", "200", "--ffa", "50" if len(act) >= 3 else "0", "--no-bots"] if final
            else ["--round", str(s["round"])])
    if not final and len(act) <= 2:
        # the duel: the random and avoid-walls bots lose every game, so they only cost search time
        cmd += ["--bots", "territory,space-filler"]
    log = open(os.path.join(ROOT, "logs", "ranking.log"), "a")
    env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)


def tournament_stats(rnd):
    p = os.path.join(TDIR, "stats.csv")
    ids = IDS + [b["id"] for b in config.BOTS]
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["round", "time"] + ids)
        w.writerow([rnd["round"], rnd["t"]] + [round(rnd["ratings"].get(i), 1) if i in rnd["ratings"] else "" for i in ids])


def after_round(s):
    p = os.path.join(RES, "rounds", f"round_{s['round']:04d}.json")
    if not os.path.exists(p):
        event("error", f"Ranking round {s['round']} produced no result", "See logs/ranking.log.")
        return
    rnd = json.load(open(p))
    tournament_stats(rnd)
    if s["round"] % 4 == 0 and not pgrep("^" + PY + " -m tournament.arena"):
        # the Arena page's game library, with the newest PPO and Mini-AlphaZero (low priority, in the background)
        env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        subprocess.Popen(["nice", "-n", "10", PY, "-m", "tournament.arena", "--procs", "2"], cwd=ROOT, env=env,
                         stdout=open(os.path.join(ROOT, "logs", "arena.log"), "a"), stderr=subprocess.STDOUT, start_new_session=True)
    if s["round"] % 8 == 1 or not os.path.exists(os.path.join(RES, "play_models.json")):
        try:
            from tournament.play import update_play_files
            lead_now = max((m for m in active(s) if m in rnd["ratings"]), key=lambda m: rnd["ratings"][m], default=None)
            update_play_files(s["round"], {k: v for k, v in rnd["specs"].items() if k in IDS}, lead_now)
        except Exception:
            event("error", "Could not update the Play tab", traceback.format_exc()[-800:])
    act = active(s)
    rated = sorted([m for m in act if m in rnd["ratings"]], key=lambda m: -rnd["ratings"][m])
    txt = ", ".join(f"{LBL[m]} {round(rnd['ratings'][m])}" for m in rated)
    hl = rnd.get("highlights", [])
    event("round_best", f"Round {s['round']}: {txt}",
          f"{rnd['games']} games in {rnd['seconds']:.0f} s. Bots: " +
          ", ".join(f"{LBL[b['id']]} {round(rnd['ratings'][b['id']])}" for b in config.BOTS if b['id'] in rnd['ratings']) +
          (f"\nReplay: {hl[0]['caption']}" if hl else ""), champion=hl[0]["path"] if hl else None)
    # leader by the average of the last two rounds (one round is noisy)
    av = {m: avg_rating(m, 2) for m in rated}
    av = {m: v for m, v in av.items() if v is not None}
    if not av:
        return
    order = sorted(av, key=lambda m: -av[m])
    lead = order[0]
    if s.get("leader") != lead:
        prev = s.get("leader")
        event("lead_change", f"{LBL[lead]} takes the lead" + (f" from {LBL[prev]}" if prev else ""),
              f"Average rating over the last two rounds: " + ", ".join(f"{LBL[m]} {round(av[m])}" for m in order),
              champion=hl[0]["path"] if hl else None)
        s["leader"] = lead
    # catch-up rule
    for m in act:
        if s["models"][m]["status"] != "paused":
            continue
        others = [av[o] for o in av if o != m]
        gap = av.get(m, 0) - max(others) if others else 0
        if gap < config.RESUME_GAP:
            os.remove(os.path.join(ROOT, "runs", m, "PAUSE")) if os.path.exists(os.path.join(ROOT, "runs", m, "PAUSE")) else None
            s["models"][m]["status"] = "training"
            for pz in s["pauses"]:
                if pz["id"] == m and not pz.get("to"):
                    pz["to"] = now().isoformat()
            event("resume_training", f"{LBL[m]} starts training again", f"Its lead is down to {gap:.0f} points (< {config.RESUME_GAP:.0f}).", model=m)
    if len(order) >= 2 and s["models"][lead]["status"] == "training":
        gap = av[order[0]] - av[order[1]]
        if gap > config.PAUSE_GAP:
            open(os.path.join(ROOT, "runs", lead, "PAUSE"), "w").write(stamp())
            s["models"][lead]["status"] = "paused"
            s["pauses"].append({"id": lead, "from": now().isoformat(), "to": None})
            event("pause", f"{LBL[lead]} is {gap:.0f} points ahead: its training is paused",
                  f"Catch-up rule: it keeps playing every ranking round with its current checkpoint, and starts training again "
                  f"when the gap is under {config.RESUME_GAP:.0f}.", model=lead)


def knockout(s, k):
    act = active(s)
    av = {m: avg_rating(m, 3) for m in act}
    av = {m: (v if v is not None else -1e9) for m, v in av.items()}
    out = min(av, key=lambda m: av[m])
    stop_trainer(out)
    pz = os.path.join(ROOT, "runs", out, "PAUSE")
    if os.path.exists(pz):
        os.remove(pz)
    s["models"][out]["status"] = "out"
    s["models"][out]["out_at"] = now().isoformat()
    for p in s["pauses"]:
        if p["id"] == out and not p.get("to"):
            p["to"] = now().isoformat()
    order = sorted(av, key=lambda m: -av[m])
    event("knockout", f"Knockout {k}: {LBL[out]} is out",
          "Average rating over the last three rounds: " + ", ".join(f"{LBL[m]} {round(av[m])}" for m in order) +
          f". {LBL[out]}'s last checkpoint stays on the site; its CPU share goes to the others.", model=out)
    rebalance(s, f"After knockout {k}, {len(active(s))} models share the training cores.")


def check_retire(s):
    """results/RETIRE.json {"ids": [...], "why": "..."}: stop training these models for good (their checkpoints stay)."""
    p = os.path.join(RES, "RETIRE.json")
    if not os.path.exists(p):
        return
    req = json.load(open(p))
    os.remove(p)
    done = []
    for m in req["ids"]:
        if m in active(s):
            stop_trainer(m)
            pz = os.path.join(ROOT, "runs", m, "PAUSE")
            if os.path.exists(pz):
                os.remove(pz)
            s["models"][m]["status"] = "retired"
            s["models"][m]["out_at"] = now().isoformat()
            done.append(m)
    if done:
        event("retire", ", ".join(LBL[m] for m in done) + " stop training: the tournament becomes a duel",
              req.get("why", ""))
        rebalance(s, f"{len(active(s))} models share the training cores.")


def final(s):
    act = active(s)
    for m in act:
        stop_trainer(m)
        pz = os.path.join(ROOT, "runs", m, "PAUSE")
        if os.path.exists(pz):
            os.remove(pz)
        s["models"][m]["status"] = "finalist"
    for p in s["pauses"]:
        if not p.get("to"):
            p["to"] = now().isoformat()
    event("final", "The final starts: training has stopped",
          "Finalists: " + ", ".join(LBL[m] for m in act) + ". 200 games per pair (100 start positions, both sides)" +
          (" and 50 three-bike free-for-all games" if len(act) >= 3 else "") + ", using each model's last checkpoint.")
    save_state(s)
    p = run_round(s, final=True)
    p.wait()
    fp = os.path.join(RES, "rounds", "final.json")
    if not os.path.exists(fp):
        event("error", "The final produced no result", "See logs/ranking.log.")
        return
    f = json.load(open(fp))
    order = f["ids"]
    win = order[0]
    s["models"][win]["status"] = "winner"
    ffa = f.get("ffa", {})
    event("final", f"🏆 {LBL[win]} wins the Tron Tournament",
          "Final ratings (1v1): " + ", ".join(f"{LBL[m]} {round(f['ratings'][m])}" for m in order) +
          (". Free-for-all wins: " + ", ".join(f"{LBL.get(k, k)} {v}" for k, v in sorted(ffa['wins'].items(), key=lambda x: -x[1])) if ffa else ""),
          champion=f["highlights"][0]["path"] if f.get("highlights") else None)
    s["final_done"] = True
    s["note"] = f"Final played. Winner: {LBL[win]}."


def main():
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    s = load_state()
    if not os.path.exists(os.path.join(TDIR, "events.jsonl")):
        event("start", "The tournament starts",
              "Five models start training from scratch: " + ", ".join(LBL[m] for m in IDS) +
              f". First ranking round in {FIRST_ROUND_AFTER_MIN} minutes, then every {config.ROUND_EVERY_MIN} minutes.")
    s["note"] = ""
    ranking = None
    sched = [(dt.datetime.fromisoformat(x["t"]), x["kind"]) for x in config.SCHEDULE]
    kos = [t for t, k in sched if k == "knockout"]
    final_t = [t for t, k in sched if k == "final"][0]
    while True:
        try:
            if s["final_done"]:
                save_state(s)
                time.sleep(60)
                continue
            if now() >= final_t and ranking is None:
                final(s)
                save_state(s)
                continue
            if ranking is not None and ranking.poll() is not None:
                after_round(s)
                s["processed_round"] = s["round"]
                ranking = None
            if ranking is None and s.get("processed_round", 0) < s["round"] and not pgrep("^" + PY + " -m tournament.ranking"):
                after_round(s)                    # a round that finished while the supervisor was restarting
                s["processed_round"] = s["round"]
            if ranking is None:
                check_retire(s)
            if ranking is None and s["knockouts_done"] < len(kos) and now() >= kos[s["knockouts_done"]] and len(active(s)) > 2:
                s["knockouts_done"] += 1
                knockout(s, s["knockouts_done"])
            ensure_trainers(s)
            if ranking is None and time.time() >= s["next_round_at"] and now() < final_t:
                ranking = run_round(s)
                s["next_round_at"] = time.time() + config.ROUND_EVERY_MIN * 60
            for m in IDS:
                s["models"][m]["train"] = train_text(m)
            save_state(s)
        except Exception:
            event("error", "Supervisor error (it keeps running)", traceback.format_exc()[-1500:])
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main())
