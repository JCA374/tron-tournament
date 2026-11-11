"""The Play tab: each model's checkpoint from a recent ranking round plays a fixed set of 10 games.
update_play_files() is called by the supervisor every few rounds; publish.py renders each (model, game) clip once."""
import json
import os

from models.common import ROOT, load_agent
from tournament.match import play_game
from tron import config

RES = os.path.join(ROOT, "results")

GAMES = [
    {"id": "sf1", "title": "vs space-filler · A", "desc": "1v1, starting near the left wall, facing up", "opp": "space-filler", "seed": 11},
    {"id": "sf2", "title": "vs space-filler · B", "desc": "1v1, starting near the centre", "opp": "space-filler", "seed": 23},
    {"id": "sf3", "title": "vs space-filler · C", "desc": "1v1, starting near a corner", "opp": "space-filler", "seed": 37},
    {"id": "sf4", "title": "vs space-filler · D", "desc": "1v1, starting facing the wall", "opp": "space-filler", "seed": 41},
    {"id": "aw1", "title": "vs avoid-walls", "desc": "1v1 against the random-but-careful bot", "opp": "avoid-walls", "seed": 53},
    {"id": "aw2", "title": "vs avoid-walls · B", "desc": "1v1, a different start", "opp": "avoid-walls", "seed": 67},
    {"id": "ld1", "title": "vs the leader", "desc": "1v1 against the current leader (the leader plays its own mirror)", "opp": "leader", "seed": 71},
    {"id": "ld2", "title": "vs the leader · B", "desc": "1v1 against the current leader, another start", "opp": "leader", "seed": 89},
    {"id": "ffa1", "title": "Free-for-all", "desc": "every model on this list in one arena", "opp": "ffa", "seed": 97},
    {"id": "ffa2", "title": "Free-for-all · B", "desc": "every model on this list, other starts", "opp": "ffa", "seed": 131},
]


def update_play_files(round_no, specs, leader):
    """specs: {model id: snapshot spec} from the round; leader: model id."""
    d = os.path.join(RES, "play")
    os.makedirs(d, exist_ok=True)
    models = []
    for m in config.CONTESTANTS:
        if m["id"] not in specs:
            continue
        p = os.path.join(d, f"{m['id']}.json")
        json.dump(dict(specs[m["id"]], id=m["id"]), open(p, "w"))
        models.append({"id": f"{m['id']}_r{round_no}", "label": m["label"], "path": os.path.relpath(p, ROOT),
                       "note": f"{m['label']}: checkpoint from ranking round {round_no}. {m['method']}."})
    json.dump(models, open(os.path.join(RES, "play_models.json"), "w"), indent=1)
    json.dump(GAMES, open(os.path.join(RES, "play_games.json"), "w"), indent=1)
    json.dump({"leader": leader}, open(os.path.join(d, "leader.json"), "w"))


def play_tab_game(spec_path, game):
    spec_path = os.path.join(ROOT, spec_path)
    spec = json.load(open(spec_path))
    me = spec["id"]
    lab = config.BY_ID[me]["label"]
    d = os.path.dirname(spec_path)
    if game["opp"] == "ffa":
        ids = [json.load(open(os.path.join(d, f"{m['id']}.json")))["id"] for m in config.CONTESTANTS
               if os.path.exists(os.path.join(d, f"{m['id']}.json"))]
        ids = [me] + [i for i in ids if i != me]
        specs = [json.load(open(os.path.join(d, f"{i}.json"))) for i in ids]
        agents = [load_agent(s, seed=game["seed"] + k) for k, s in enumerate(specs)]
        names = [config.BY_ID[i]["label"] for i in ids]
        colors = [config.BY_ID[i]["color"] for i in ids]
    else:
        if game["opp"] == "leader":
            lead = json.load(open(os.path.join(d, "leader.json")))["leader"]
            ospec = json.load(open(os.path.join(d, f"{lead}.json")))
            oname = config.BY_ID[lead]["label"] + (" (mirror)" if lead == me else "")
            ocol = config.BY_ID[lead]["color"] if lead != me else "#adb5bd"
        else:
            ospec, oname, ocol = {"kind": game["opp"]}, config.BY_ID[game["opp"]]["label"], config.BY_ID[game["opp"]]["color"]
        agents = [load_agent(spec, seed=game["seed"]), load_agent(ospec, seed=game["seed"] + 1)]
        names, colors = [lab, oname], [config.BY_ID[me]["color"], ocol]
    rep = play_game(agents, names, colors, game["seed"])
    rep["me"] = lab
    return rep
