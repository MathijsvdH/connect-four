# Streamlit web app: play Connect Four against the DQN agent from connect_four_deep_rl.ipynb.
# Every game is saved to data/games.jsonl. With LEARN on, the agent also fine-tunes on all saved games after each game.
# Run:  pip install -r requirements.txt   then   streamlit run app.py
import os, threading
from datetime import datetime
import streamlit as st
from connect4 import DEVICE, Board, AI, save_game, load_games, finetune, warm_up

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODELS = [os.path.join(HERE, "models", f) for f in ("c4_strong.pt", "c4_baseline.pt")]   # exported by the notebook (section 10)
LIVE_MODEL = os.path.join(HERE, "models", "c4_live.pt")      # base model + what it learned in the app; deleting it = reset
GAMES_LOG = os.path.join(HERE, "data", "games.jsonl")
LEARN_STEPS = 100                                             # gradient steps after every finished game
# Off for the hosted app: the cloud disk resets on every restart, and games against casual human players are
# too easy to teach the agent anything useful. Set to True to try the learning loop locally.
LEARN = False

def ui_file(name):
    with open(os.path.join(HERE, "ui", name), encoding="utf-8") as f: return f.read()

st.set_page_config(page_title="Connect Four", page_icon=":material/radio_button_checked:", layout="centered",
                   initial_sidebar_state="collapsed")
st.html(f"<style>{ui_file('page.css')}</style>")
ss = st.session_state
for k, v in dict(starter="Alternate", depth=2, learn=True, show_eval=False, game_id=0, note="", results=[]).items(): ss.setdefault(k, v)


# ---- The board: a custom component (ui/board.js + ui/board.css) that animates the game and sends clicks back
@st.cache_resource
def board_component():
    return st.components.v2.component("c4_board", css=ui_file("board.css"), js=ui_file("board.js"))


# ---- Model loading
# The live model is used once it exists. The cache is keyed on the file's modification time,
# so new weights saved after a game are picked up automatically on the next rerun.
def model_path():
    return next((p for p in ((LIVE_MODEL,) if LEARN else ()) + tuple(BASE_MODELS) if os.path.exists(p)), None)

@st.cache_resource(max_entries=2)
def load_ai(path, mtime):
    return AI(path)

def get_ai():
    p = model_path()
    return load_ai(p, os.path.getmtime(p))

@st.cache_resource
def warm_up_gpu():                                            # once per server process, in the background
    threading.Thread(target=warm_up, daemon=True).start()


# ---- Game flow (callbacks: they run before the page is redrawn)
def ai_move():
    a, scores = get_ai().choose(ss.board, ss.depth)
    ss.evals = [None if s == float("-inf") else round(float(s), 3) for s in scores]   # the AI's view of every column
    ss.board.apply_action(a)

def human_move(c):
    b = ss.board
    if b.is_terminal() or b.current_player() != ss.human_player or c not in b.legal_actions(): return
    b.apply_action(c)
    if not b.is_terminal(): ai_move()
    if b.is_terminal():
        record_game("draw" if b.winner is None else ("human" if b.winner == ss.human_player else "ai"))
        ss.pending_learn = LEARN and ss.learn

def on_move():
    m = ss.c4.get("move")
    if m and m["n"] == len(ss.board.moves): human_move(int(m["col"]))   # ignore stale or double clicks

def record_game(result):
    save_game(GAMES_LOG, dict(time=datetime.now().isoformat(timespec="seconds"), human_player=ss.human_player,
                              moves=ss.board.moves, result=result, depth=ss.depth, model=os.path.basename(model_path())))
    ss.saved = True
    ss.results = ss.results + [result]                       # this session's score only: a refresh resets it

def new_game():
    if "board" in ss and ss.board.moves and not ss.saved: record_game("unfinished")   # an abandoned game is still data
    starter = ss.starter or "Alternate"
    if starter == "Alternate": ss.human_player = 1 - ss.get("human_player", 1)
    else:                      ss.human_player = 0 if starter == "You" else 1
    ss.board, ss.saved, ss.evals, ss.pending_learn = Board(), False, None, False
    ss.game_id += 1
    if ss.human_player == 1: ai_move()

def reset_model():
    if os.path.exists(LIVE_MODEL): os.remove(LIVE_MODEL)
    load_ai.clear(); ss.note = "Model reset to the notebook export"


# ---- Learning after a finished game
# A fresh copy of the current model is fine-tuned on all saved games (newest game weighted most, see
# connect4.finetune) and saved as the live model; the cached player is never modified in place.
def learn_from_games():
    games = [g["moves"] for g in load_games(GAMES_LOG)]
    ai = AI(model_path())
    loss = finetune(ai, games, steps=LEARN_STEPS)
    updates = ai.ck.get("updates", 0) + 1
    ai.save(LIVE_MODEL, updates=updates, games_seen=len(games))
    ss.note = f"Learned from {len(games)} games (update {updates}, loss {loss:.3f})"


if model_path() is None:
    st.error("No trained model found. Run sections 9–10 of the notebook and put `c4_strong.pt` in the `models/` folder next to `app.py`.")
    st.stop()
warm_up_gpu()
if "board" not in ss: new_game()


# ---- Page: board component, then a collapsed settings panel
b = ss.board
res = ss.results
learning = bool(ss.get("pending_learn"))
board_component()(
    key="c4", height="content",
    data=dict(game=ss.game_id, moves=b.moves, human=ss.human_player, over=b.is_terminal(), winner=b.winner,
              win=[[int(r), int(c)] for r, c in b.win_cells], evals=ss.evals, show_eval=ss.show_eval,
              record=[res.count("human"), res.count("draw"), res.count("ai")], learning=learning, note=ss.note),
    on_move_change=on_move, on_new_change=new_game)

if learning:                                                  # the board is already on screen while this runs
    learn_from_games(); ss.pending_learn = False
    st.rerun()

with st.expander("Settings"):
    st.segmented_control("First move (next game)", ["You", "AI", "Alternate"], key="starter")
    st.slider("AI look-ahead (moves)", 0, 4, key="depth",
              help="0 = the network's greedy move. Higher = negamax search with the network judging the positions at the horizon: stronger, a bit slower.")
    if LEARN: st.toggle("Learn after every game", key="learn")
    st.toggle("Show the AI's evaluation", key="show_eval")
    ck = get_ai().ck
    # model = (f"c4_live.pt, fine-tuned {ck.get('updates', 0)} times in this app" if model_path() == LIVE_MODEL
             #else f"{os.path.basename(model_path())}, as exported by the notebook")
    # st.caption(f"Model: {model} · running on {DEVICE.type.upper()} · {len(res)} games saved in data/games.jsonl")
    if LEARN:
        sure = st.checkbox("Forget what the AI learned here (saved games are kept)")
        st.button("Reset model", on_click=reset_model, disabled=not sure or not os.path.exists(LIVE_MODEL))
