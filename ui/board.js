// Connect Four board component for app.py (Streamlit custom component v2).
// Python sends the game state as `data`; this script draws it, animates discs that are new on screen and
// sends clicks back as trigger values: "move" = {col, n} (n = move number, to ignore stale clicks), "new".
const ROWS = 6, COLS = 7, PAD = 0.2, GAP = 0.12;      // board padding and preview-row gap, in cells
const MAX_CELL = 68;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const wait = (ms) => new Promise((res) => setTimeout(res, reduced ? 0 : ms));
const views = new WeakMap();                           // parentElement -> {ui, st}
const memo = new Map();                                // component key -> {game, count}: discs already shown before a re-mount

// ---- one-time DOM construction
function build(root) {
  const wrap = document.createElement("div");          // root is a ShadowRoot: no insertAdjacentHTML there
  root.appendChild(wrap);
  wrap.insertAdjacentHTML("beforeend", `
    <div class="c4">
      <div class="head">
        <div><div class="title">Connect Four</div><div class="sub">against a Double DQN that learns from every game</div></div>
        <div class="status"><span class="dot"></span><span class="txt"></span></div>
      </div>
      <div class="stage locked">
        <div class="back"></div>
        <div class="discs"><div class="colhi"></div><div class="ghost"></div></div>
        <div class="front"></div>
        <div class="hits"></div>
      </div>
      <div class="evals hidden"></div>
      <div class="foot"><div class="meta"><div class="rec"></div><div class="note"></div></div><button class="new">New game</button></div>
    </div>`);
  const $ = (s) => root.querySelector(s);
  const ui = { c4: $(".c4"), stage: $(".stage"), discs: $(".discs"), ghost: $(".ghost"), colhi: $(".colhi"), hits: $(".hits"),
               evals: $(".evals"), status: $(".status"), dot: $(".status .dot"), txt: $(".status .txt"),
               rec: $(".rec"), note: $(".note"), newBtn: $(".new"), cell: MAX_CELL };

  // front plate: rounded rectangle with 42 round holes (even-odd fill), drawn in cell units so it scales crisply
  const W = COLS + 2 * PAD, H = ROWS + 2 * PAD, k = 0.4, q = 0.3;
  let d = `M${q} 0H${W - q}Q${W} 0 ${W} ${q}V${H - q}Q${W} ${H} ${W - q} ${H}H${q}Q0 ${H} 0 ${H - q}V${q}Q0 0 ${q} 0Z`;
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
    const x = PAD + c + 0.5, y = PAD + r + 0.5;
    d += `M${x - k} ${y}a${k} ${k} 0 1 0 ${2 * k} 0a${k} ${k} 0 1 0 ${-2 * k} 0Z`;
  }
  $(".front").innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><path fill-rule="evenodd" d="${d}"/></svg>`;
  for (let c = 0; c < COLS; c++) {
    ui.hits.insertAdjacentHTML("beforeend", `<div data-c="${c}"></div>`);
    ui.evals.insertAdjacentHTML("beforeend", `<div class="ev"><span>&nbsp;</span><div class="bar"><i></i></div></div>`);
  }

  // every size is a multiple of --cell, recomputed when the page width changes
  const fit = () => {
    ui.cell = Math.max(28, Math.min(MAX_CELL, ui.c4.clientWidth / (COLS + 2 * PAD)));
    const s = ui.c4.style;
    s.setProperty("--cell", `${ui.cell}px`); s.setProperty("--pad", `${PAD * ui.cell}px`); s.setProperty("--gap", `${GAP * ui.cell}px`);
    s.setProperty("--stage-w", `${(COLS + 2 * PAD) * ui.cell}px`);
  };
  new ResizeObserver(fit).observe(ui.c4); fit();
  return ui;
}

// ---- animations
function placeDisc(ui, r, c, p) {
  const el = document.createElement("div");
  el.className = `disc p${p}`; el.dataset.rc = `${r},${c}`;
  el.style.setProperty("--r", r); el.style.setProperty("--c", c);
  ui.discs.insertBefore(el, ui.colhi);
  return el;
}

function drop(ui, r, c, p) {
  // gravity: accelerate down from the preview row, land, one small bounce
  const el = placeDisc(ui, r, c, p);
  if (reduced) return Promise.resolve();
  const dist = el.offsetTop, bounce = Math.min(dist * 0.06, ui.cell * 0.12);
  return el.animate([
    { transform: `translateY(${-dist}px)`, easing: "cubic-bezier(.55, 0, 1, .45)" },
    { transform: "translateY(0)", offset: 0.78, easing: "cubic-bezier(0, 0, .3, 1)" },
    { transform: `translateY(${-bounce}px)`, offset: 0.89, easing: "cubic-bezier(.7, 0, 1, 1)" },
    { transform: "translateY(0)" },
  ], { duration: 200 + 120 * Math.sqrt(ROWS - r), fill: "backwards" }).finished.catch(() => {});
}

function clearDiscs(ui, discs) {
  // new game: the discs fall out through the bottom of the board, top rows last
  const h = ui.stage.clientHeight;
  return Promise.all(discs.map((el) => el.animate(
    [{ transform: "translateY(0)", opacity: 1 }, { transform: `translateY(${h}px)`, opacity: 0.6 }],
    { duration: reduced ? 1 : 480, delay: reduced ? 0 : (+el.style.getPropertyValue("--r")) * 35 + Math.random() * 50,
      easing: "cubic-bezier(.55, 0, .85, .35)", fill: "forwards" }).finished.catch(() => {})))
    .then(() => discs.forEach((el) => el.remove()));
}

function setStatus(ui, text, dotClass) {
  if (ui.txt.textContent === text && ui.dot.className === `dot ${dotClass}`) return;
  ui.status.classList.add("swap");
  setTimeout(() => { ui.txt.textContent = text; ui.dot.className = `dot ${dotClass}`; ui.status.classList.remove("swap"); }, reduced ? 0 : 160);
}

function showGhost(ui, st, c) {
  const on = c !== null && st.canPlay;
  const gx = `${(c ?? 0) * ui.cell}px`;
  ui.ghost.className = `ghost p${st.data.human}${on ? " on" : ""}`;
  ui.colhi.classList.toggle("on", on);
  if (on) { ui.ghost.style.setProperty("--gx", gx); ui.colhi.style.setProperty("--gx", gx); }
}

// ---- after the animations: status, input lock, winning line, evaluations
function settle(ui, st) {
  const d = st.data;
  const synced = st.moves.length === d.moves.length;
  st.canPlay = synced && !d.over && st.moves.length % 2 === d.human;
  ui.stage.classList.toggle("locked", !st.canPlay);
  if (d.over) {
    for (const [r, c] of d.win) ui.discs.querySelector(`[data-rc="${r},${c}"]`)?.classList.add("win");
    ui.stage.classList.add("over");
    setStatus(ui, d.winner === null ? "Draw" : d.winner === d.human ? "You win" : "AI wins", d.winner === null ? "" : `p${d.winner}`);
  } else if (st.canPlay) setStatus(ui, "Your move", `p${d.human}`);
  else setStatus(ui, "Thinking", `p${1 - d.human} busy`);
  showGhost(ui, st, st.hover);

  const show = d.show_eval && Array.isArray(d.evals);
  ui.evals.classList.toggle("hidden", !show);
  if (show) {
    const best = d.evals.reduce((b, v, i) => (v !== null && (b < 0 || v > d.evals[b]) ? i : b), -1);
    [...ui.evals.children].forEach((ev, i) => {
      const v = d.evals[i];
      ev.classList.toggle("best", i === best);
      ev.querySelector("span").textContent = v === null ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
      ev.querySelector("i").style.setProperty("--w", v === null ? "0%" : `${((v + 1) / 2) * 100}%`);
    });
  }
}

// ---- entry point: called by Streamlit on every script run
export default function (component) {
  const { data, parentElement, setTriggerValue, key } = component;
  let view = views.get(parentElement);
  if (!view) {
    const ui = build(parentElement);
    const st = { game: null, moves: [], heights: Array(COLS).fill(0), chain: Promise.resolve(), canPlay: false, hover: null, data };
    view = { ui, st };
    views.set(parentElement, view);

    ui.hits.addEventListener("click", (e) => {
      const c = Number(e.target.dataset.c);
      if (Number.isNaN(c) || !st.canPlay || st.heights[c] >= ROWS) return;
      // optimistic: the human's disc falls immediately, Python answers with the AI's move
      const i = st.moves.length, r = st.heights[c]++;
      st.moves.push(c); st.canPlay = false; ui.stage.classList.add("locked"); showGhost(ui, st, null);
      setStatus(ui, "Thinking", `p${1 - st.data.human} busy`);
      st.chain = st.chain.then(() => drop(ui, r, c, i % 2));
      memo.set(key, { game: st.game, count: st.moves.length });
      st.send("move", { col: c, n: i });
    });
    ui.hits.addEventListener("pointermove", (e) => {
      const c = Number(e.target.dataset.c);
      if (!Number.isNaN(c) && c !== st.hover) { st.hover = c; showGhost(ui, st, c); }
    });
    ui.hits.addEventListener("pointerleave", () => { st.hover = null; showGhost(ui, st, null); });
    ui.newBtn.addEventListener("click", () => st.send("new", Date.now()));
  }
  const { ui, st } = view;
  st.send = setTriggerValue; st.data = data;

  // a new game: clear the board (instantly on a fresh mount)
  if (data.game !== st.game) {
    const old = [...ui.discs.querySelectorAll(".disc")];
    const known = memo.get(key);
    st.instant = known && known.game === data.game ? known.count : 0;   // re-mounted mid-game: don't replay old drops
    st.game = data.game; st.moves = []; st.heights = Array(COLS).fill(0);
    st.chain = st.chain.then(() => clearDiscs(ui, old)).then(() => ui.stage.classList.remove("over"));
  }
  // out of sync (should not happen): rebuild this game's discs without animation
  if (!data.moves.slice(0, st.moves.length).every((m, i) => m === st.moves[i])) {
    ui.discs.querySelectorAll(".disc").forEach((el) => el.remove());
    st.moves = []; st.heights = Array(COLS).fill(0); st.instant = data.moves.length;
  }
  // drop the moves that are not on screen yet, one after the other
  for (const m of data.moves.slice(st.moves.length)) {
    const i = st.moves.length, r = st.heights[m]++, animate = i >= (st.instant || 0);
    st.moves.push(m);
    st.chain = st.chain.then(() => (animate ? wait(i === 0 ? 120 : 260).then(() => drop(ui, r, m, i % 2)) : placeDisc(ui, r, m, i % 2)));
  }
  memo.set(key, { game: st.game, count: st.moves.length });
  st.chain = st.chain.then(() => settle(ui, st));

  // footer updates right away
  const [w, dr, l] = data.record;
  ui.rec.innerHTML = `You <b>${w}</b> &middot; Draws <b>${dr}</b> &middot; AI <b>${l}</b>`;
  ui.note.textContent = data.learning ? "Learning from this game..." : data.note;
  ui.note.classList.toggle("learning", !!data.learning);
}
