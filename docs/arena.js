// Arena: replays games from data/arena.json (start positions + moves) with the same rules as the Python simulator.
(async function () {
  const A = await get('arena.json');
  const FIN = await get('final.json');            // exists once the final has been played
  const cv = $('board'), ctx = cv.getContext('2d');
  if (!A) { $('meta').textContent = 'The arena library is being built. Check back in a few minutes.'; return; }
  const N = A.board, S = N + 2;
  const DY = [-1, 0, 1, 0], DX = [0, 1, 0, -1];
  const E = Object.fromEntries(A.entrants.map((e) => [e.id, e]));
  let pick = ['ppo', 'az'], mode = 'pair', list = [], gi = 0, game = null, speed = 1, playing = true;
  const SPEEDS = [[0.5, '½×'], [1, '1×'], [2, '2×'], [4, '4×'], [8, '8×']];

  // ---------------------------------------------------------------- picking
  function chips(el, slot) {
    el.innerHTML = A.entrants.map((e) => `<button class="chip ${pick[slot] === e.id ? 'on' : ''}" data-s="${slot}" data-id="${e.id}"
      style="--c:${e.color}"><span class="dot" style="background:${e.color}"></span>${esc(e.label)}<span class="st">${e.status}</span></button>`).join('');
  }
  function drawPick() { chips($('pickA'), 0); chips($('pickB'), 1); }
  document.addEventListener('click', (ev) => {
    const b = ev.target.closest('button.chip'); if (!b) return;
    const s = +b.dataset.s, id = b.dataset.id;
    if (pick[1 - s] === id) pick[1 - s] = pick[s];      // picking the other side's fighter swaps them
    pick[s] = id; drawPick();
  });
  $('fight').onclick = () => { mode = 'pair'; load(); };
  $('ffa').onclick = () => { mode = 'ffa'; load(); };
  if (FIN && FIN.games.length) {
    const fb = document.createElement('button'); fb.className = 'fight'; fb.id = 'finalbtn';
    fb.textContent = `🏆 Watch the final (${FIN.games.length} games)`;
    fb.onclick = () => { mode = 'final'; load(); };
    $('ffa').after(fb);
  }
  $('next').onclick = () => { if (list.length) { gi = (gi + 1) % list.length; start(); } };
  $('replay').onclick = () => start();
  $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
  $('speeds').innerHTML = SPEEDS.map(([v, t]) => `<button data-sp="${v}" class="${v === speed ? 'on' : ''}">${t}</button>`).join('');
  $('speeds').onclick = (ev) => { const b = ev.target.closest('button'); if (!b) return; speed = +b.dataset.sp;
    [...$('speeds').children].forEach((x) => x.classList.toggle('on', x === b)); };

  function load() {
    if (mode === 'final') { list = FIN.games.slice(); const ids = list[0].ids; pick = [ids[0], ids[1]].sort(); drawPick(); }
    else if (mode === 'ffa') list = A.ffa.slice();
    else list = (A.pairs[[...pick].sort().join('|')] || []).slice();
    gi = 0;
    series();
    start();
  }
  function series() {
    if (mode === 'ffa') {
      const w = {}; let d = 0;
      list.forEach((g) => { if (g.result < 0) d++; else w[g.ids[g.result]] = (w[g.ids[g.result]] || 0) + 1; });
      $('series').innerHTML = `<div class="k">Free-for-all · ${list.length} games</div>` +
        Object.entries(w).sort((a, b) => b[1] - a[1]).map(([id, n]) => `<div class="row"><span class="dot" style="background:${E[id].color}"></span>${esc(E[id].label)}<b>${n}</b></div>`).join('') +
        (d ? `<div class="row muted">draws<b>${d}</b></div>` : '');
      return;
    }
    const [a, b] = pick, wa = list.filter((g) => g.result >= 0 && g.ids[g.result] === a).length,
      wb = list.filter((g) => g.result >= 0 && g.ids[g.result] === b).length, d = list.length - wa - wb;
    $('series').innerHTML = `<div class="k">${mode === 'final' ? '🏆 The final' : 'Series'} · ${list.length} games</div>
      <div class="score"><span style="color:${E[a].color}">${esc(E[a].label)}</span> <b>${wa}</b> – <b>${wb}</b> <span style="color:${E[b].color}">${esc(E[b].label)}</span></div>
      ${d ? `<div class="muted small">${d} draw${d > 1 ? 's' : ''}</div>` : ''}`;
  }

  // ---------------------------------------------------------------- simulation (same rules as tron/sim.py)
  function newState(g) {
    const grid = new Int8Array(S * S);
    for (let i = 0; i < S; i++) { grid[i] = grid[(S - 1) * S + i] = grid[i * S] = grid[i * S + S - 1] = 1; }
    const n = g.ids.length, pos = g.pos.map((p) => p.slice()), dirs = g.dirs.slice(), alive = Array(n).fill(true);
    pos.forEach((p, b) => { grid[p[0] * S + p[1]] = 2 + b; });
    return { grid, pos, dirs, alive, t: 0, n, prev: pos.map((p) => p.slice()), sep: null };
  }
  function step(st, acts) {
    const n = st.n, ny = [], nx = [], dead = Array(n).fill(false);
    st.prev = st.pos.map((p) => p.slice());
    for (let b = 0; b < n; b++) if (st.alive[b]) {
      const d = (st.dirs[b] + acts[b] - 1 + 4) % 4; st.dirs[b] = d;
      ny[b] = st.pos[b][0] + DY[d]; nx[b] = st.pos[b][1] + DX[d];
      if (st.grid[ny[b] * S + nx[b]] !== 0) dead[b] = true;
    }
    for (let b = 0; b < n; b++) if (st.alive[b]) for (let c = b + 1; c < n; c++)
      if (st.alive[c] && ny[b] === ny[c] && nx[b] === nx[c]) { dead[b] = dead[c] = true; }
    const died = [];
    for (let b = 0; b < n; b++) if (st.alive[b]) {
      st.pos[b] = [ny[b], nx[b]];
      if (dead[b]) { st.alive[b] = false; died.push(b); } else st.grid[ny[b] * S + nx[b]] = 2 + b;
    }
    st.t++;
    return died;
  }
  // multi-source BFS from every live head: owner of each empty cell (-1 = nobody / tie)
  function territory(st) {
    const dist = new Int16Array(S * S).fill(-1), own = new Int8Array(S * S).fill(-1), q = new Int32Array(S * S);
    let h = 0, t = 0;
    st.pos.forEach((p, b) => { if (st.alive[b]) { const c = p[0] * S + p[1]; dist[c] = 0; own[c] = b; q[t++] = c; } });
    while (h < t) {
      const c = q[h++];
      for (let d = 0; d < 4; d++) {
        const cc = c + DY[d] * S + DX[d];
        if (st.grid[cc] !== 0) continue;
        if (dist[cc] < 0) { dist[cc] = dist[c] + 1; own[cc] = own[c]; q[t++] = cc; }
        else if (dist[cc] === dist[c] + 1 && own[cc] !== own[c]) own[cc] = -2;     // reached at the same time: contested
      }
    }
    const share = Array(st.n).fill(0);
    for (let i = 0; i < S * S; i++) if (own[i] >= 0 && st.grid[i] === 0) share[own[i]]++;
    return { own, share };
  }
  function region(st, b) {        // empty cells reachable by bike b
    const seen = new Uint8Array(S * S), q = new Int32Array(S * S); let h = 0, t = 0, n = 0;
    const p = st.pos[b]; q[t++] = p[0] * S + p[1]; seen[q[0]] = 1;
    while (h < t) { const c = q[h++]; for (let d = 0; d < 4; d++) { const cc = c + DY[d] * S + DX[d];
      if (!seen[cc] && st.grid[cc] === 0) { seen[cc] = 1; q[t++] = cc; n++; } } }
    return { seen, n };
  }
  function checkSeparation(st) {
    if (st.n !== 2 || st.sep || !st.alive[0] || !st.alive[1]) return;
    const r0 = region(st, 0), [y, x] = st.pos[1];
    const touch = [[-1, 0], [1, 0], [0, -1], [0, 1]].some(([dy, dx]) => r0.seen[(y + dy) * S + x + dx]);
    if (!touch) {
      const r1 = region(st, 1);
      st.sep = { t: st.t, a: r0.n, b: r1.n };
      const big = r0.n >= r1.n ? 0 : 1;
      toast(`🧱 Walled off at tick ${st.t}! ${E[game.ids[0]].label} ${r0.n} cells · ${E[game.ids[1]].label} ${r1.n} cells`, E[game.ids[big]].color);
    }
  }

  // ---------------------------------------------------------------- drawing
  let st, tick = 0, acc = 0, sparks = [], terr = null, ended = false;
  const cell = () => cv.width / S;
  function hexA(h, a) { const n = parseInt(h.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }
  function start() {
    game = list[gi];
    $('banner').className = 'banner'; $('toast').className = 'toast';
    if (!game) { $('meta').textContent = 'No games for this pair yet.'; return; }
    st = newState(game); tick = 0; acc = 0; sparks = []; ended = false; terr = territory(st);
    playing = true; $('play').textContent = '⏸ Pause';
    const jump = +(new URLSearchParams(location.search).get('tick') || 0);   // ?tick=150 jumps into the game
    while (st.t < jump && !ended) advance();
    $('meta').textContent = `Game ${gi + 1} of ${list.length} · start position ${game.seed % 1000} · library from ${A.updated}`;
  }
  function advance() {
    const n = game.ids.length;
    if (tick * n >= game.acts.length) return false;
    const acts = [...game.acts.slice(tick * n, tick * n + n)].map(Number);
    const died = step(st, acts); tick++;
    died.forEach((b) => burst(st.pos[b], E[game.ids[b]].color));
    terr = territory(st);
    checkSeparation(st);
    if (tick * n >= game.acts.length) finish();
    return true;
  }
  function finish() {
    ended = true;
    const r = game.result, txt = r < 0 ? 'Draw!' : `${E[game.ids[r]].label} wins!`;
    const b = $('banner'); b.textContent = txt; b.style.setProperty('--c', r < 0 ? '#fff' : E[game.ids[r]].color); b.className = 'banner show';
  }
  function burst(p, color) {
    const c = cell();
    for (let i = 0; i < 40; i++) {
      const a = Math.random() * Math.PI * 2, v = (0.5 + Math.random() * 2.5) * c / 6;
      sparks.push({ x: (p[1] + .5) * c, y: (p[0] + .5) * c, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 1, color });
    }
    sparks.push({ ring: true, x: (p[1] + .5) * c, y: (p[0] + .5) * c, r: c, life: 1, color });
  }
  let toastT = 0;
  function toast(msg, color) { const t = $('toast'); t.textContent = msg; t.style.setProperty('--c', color); t.className = 'toast show'; toastT = performance.now(); }
  function draw(frac) {
    const c = cell(), W = cv.width;
    ctx.fillStyle = '#07080d'; ctx.fillRect(0, 0, W, W);
    ctx.strokeStyle = 'rgba(120,140,255,0.06)'; ctx.lineWidth = 1;
    for (let i = 1; i < S; i++) { ctx.beginPath(); ctx.moveTo(i * c, c); ctx.lineTo(i * c, W - c); ctx.stroke(); ctx.beginPath(); ctx.moveTo(c, i * c); ctx.lineTo(W - c, i * c); ctx.stroke(); }
    if (!st) return;
    // territory tint
    if ($('terr').checked && terr) {
      for (let i = 0; i < S * S; i++) {
        const o = terr.own[i]; if (o < 0 || st.grid[i] !== 0) continue;
        ctx.fillStyle = hexA(E[game.ids[o]].color, 0.13);
        ctx.fillRect((i % S) * c, Math.floor(i / S) * c, c, c);
      }
    }
    // walls
    ctx.fillStyle = '#2b3150';
    for (let i = 0; i < S; i++) { ctx.fillRect(i * c, 0, c, c); ctx.fillRect(i * c, W - c, c, c); ctx.fillRect(0, i * c, c, c); ctx.fillRect(W - c, i * c, c, c); }
    // trails with glow
    for (let b = 0; b < st.n; b++) {
      const col = E[game.ids[b]].color;
      ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = st.alive[b] ? c * 0.9 : 0;
      ctx.fillStyle = st.alive[b] ? col : hexA(col, 0.35);
      for (let i = 0; i < S * S; i++) if (st.grid[i] === 2 + b) ctx.fillRect((i % S) * c + c * .12, Math.floor(i / S) * c + c * .12, c * .76, c * .76);
      ctx.restore();
    }
    // heads, sliding smoothly toward the next cell
    for (let b = 0; b < st.n; b++) {
      if (!st.alive[b]) continue;
      const col = E[game.ids[b]].color, p = st.pos[b];
      ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = c * 1.6; ctx.fillStyle = '#fff';
      ctx.fillRect(p[1] * c + c * .02, p[0] * c + c * .02, c * .96, c * .96); ctx.restore();
    }
    // sparks
    sparks = sparks.filter((s) => s.life > 0);
    for (const s of sparks) {
      if (s.ring) { ctx.strokeStyle = hexA(s.color, s.life); ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, 7); ctx.stroke(); s.r += c * .5; s.life -= .03; }
      else { ctx.fillStyle = hexA(s.color, s.life); ctx.fillRect(s.x - 2, s.y - 2, 4, 4); s.x += s.vx; s.y += s.vy; s.vx *= .96; s.vy *= .96; s.life -= .018; }
    }
    // live panel
    const tot = terr ? terr.share.reduce((a, b) => a + b, 0) || 1 : 1;
    $('live').innerHTML = `<div class="k">Tick ${st.t}</div>` + game.ids.map((id, b) => `<div class="lbar ${st.alive[b] ? '' : 'dead'}">
      <span class="dot" style="background:${E[id].color}"></span><span class="nm">${esc(E[id].label)}${st.alive[b] ? '' : ' 💥'}</span>
      <span class="tr"><span style="width:${100 * (terr ? terr.share[b] : 0) / tot}%;background:${E[id].color}"></span></span>
      <span class="pc">${terr && st.alive[b] ? Math.round(100 * terr.share[b] / tot) + '%' : ''}</span></div>`).join('') +
      (st.sep ? `<div class="muted small">walled off at tick ${st.sep.t}: ${st.sep.a} vs ${st.sep.b} cells</div>` : '');
    if (performance.now() - toastT > 3500) $('toast').className = 'toast';
  }
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(100, now - last); last = now;
    if (playing && st && !ended) {
      acc += dt * speed * 14 / 1000;               // 14 ticks per second at 1x
      while (acc >= 1 && !ended) { acc -= 1; advance(); }
    }
    draw(acc);
    requestAnimationFrame(frame);
  }
  function fit() {
    const w = Math.min(640, cv.parentElement.getBoundingClientRect().width || 640);
    const px = Math.max(S * 6, Math.floor(w * (window.devicePixelRatio || 1) / S) * S);     // sharp on high-DPI screens
    cv.width = cv.height = px;
  }
  window.addEventListener('resize', fit);
  if (FIN && new URLSearchParams(location.search).get('final')) mode = 'final';
  fit(); drawPick(); load(); requestAnimationFrame(frame);
})();
