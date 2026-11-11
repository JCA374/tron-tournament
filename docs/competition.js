// Competition page: data/competition.json is written by site_hooks.extra() from results/.
const STATUS_TXT = { training: ['training', 'ok'], paused: ['paused: leading', 'warn'], out: ['knocked out', 'bad'],
  finalist: ['finalist', 'ok'], winner: ['🏆 winner', 'ok'], waiting: ['not started', ''], bot: ['fixed bot', ''] };
function nextMilestone(sched) {
  const now = new Date();
  return (sched || []).find((s) => new Date(s.t) > now);
}
function tick(sched) {
  const m = nextMilestone(sched);
  if (!m) { $('countdown').textContent = '🏁 The tournament is over.'; return; }
  const s = (new Date(m.t) - new Date()) / 1000;
  const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), mi = Math.floor(s % 3600 / 60);
  $('countdown').textContent = `⏱ ${d ? d + ' d ' : ''}${h} h ${mi} min until ${m.label}`;
}
const fmtN = (v) => v == null ? '–' : v >= 1e6 ? (v / 1e6).toFixed(1) + ' M' : v >= 1e3 ? (v / 1e3).toFixed(1) + ' k' : String(v);
const sign = (v) => v == null ? '' : (v > 0 ? '+' : '') + Math.round(v);
(async function () {
  const c = await get('competition.json');
  if (!c) { $('updated').textContent = 'No data yet: the tournament is being built.'; return; }
  tick(c.schedule); setInterval(() => tick(c.schedule), 30000);
  $('updated').textContent = `Updated ${c.updated}. ${c.rounds} ranking round(s) played. ${c.note || ''}`;
  const models = c.models || [];
  const leader = models.find((m) => !m.bot && m.rating != null);
  $('cards').innerHTML = [
    ['Leader', leader ? leader.label : '–', leader ? `rating ${Math.round(leader.rating)}` : 'no round yet'],
    ['Still in', String(models.filter((m) => !m.bot && m.status !== 'out').length), 'of 5 models'],
    ['Rounds', String(c.rounds), 'every 30 minutes'],
    ['Games played', fmtN(c.games), 'in ranking rounds'],
  ].map(([k, v, s]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`).join('');
  $('board').innerHTML = '<tr><th>#</th><th>Model</th><th>Rating</th><th>Δ</th><th>Avg 3</th><th>Score vs field</th><th>Status</th><th>Training</th></tr>' +
    models.map((m, i) => {
      const [st, cls] = STATUS_TXT[m.status] || [m.status, ''];
      return `<tr class="${m.bot ? 'bot' : ''}"><td>${i + 1}</td><td><span class="dot" style="background:${m.color}"></span><b>${esc(m.label)}</b>
        <div class="muted small">${esc(m.method || '')}</div></td>
        <td class="num"><b>${m.rating == null ? '–' : Math.round(m.rating)}</b></td><td class="num">${sign(m.delta)}</td>
        <td class="num">${m.avg3 == null ? '–' : Math.round(m.avg3)}</td><td class="num">${pct(m.score)}</td>
        <td><span class="badge ${cls}">${esc(st)}</span></td>
        <td class="small">${m.bot ? '' : esc(m.train || '')}</td></tr>`;
    }).join('');
  // rating chart
  if (window.Chart && (c.history || []).length) {
    const ds = models.map((m) => ({
      label: m.label, borderColor: m.color, backgroundColor: m.color, borderDash: m.bot ? [5, 4] : [],
      borderWidth: m.bot ? 1 : 2, pointRadius: m.bot ? 0 : 2, tension: .2, spanGaps: false,
      data: c.history.map((h) => ({ x: new Date(h.t).getTime(), y: h.ratings[m.id] ?? null })),
    }));
    const pauses = c.pauses || [];
    const shade = { id: 'shade', beforeDraw(ch) {
      const { ctx, chartArea: a, scales: { x } } = ch; ctx.save();
      for (const p of pauses) { const x0 = x.getPixelForValue(new Date(p.from).getTime()), x1 = x.getPixelForValue(p.to ? new Date(p.to).getTime() : Date.now());
        ctx.fillStyle = (models.find((m) => m.id === p.id) || {}).color + '22'; ctx.fillRect(Math.max(a.left, x0), a.top, Math.min(a.right, x1) - Math.max(a.left, x0), a.bottom - a.top); }
      ctx.restore(); } };
    new Chart($('chart'), { type: 'line', data: { datasets: ds }, plugins: [shade], options: { animation: false, maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      scales: { x: { type: 'linear', ticks: { color: cssv('--muted'), callback: (v) => new Date(v).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' }) }, grid: { color: cssv('--line') } },
        y: { title: { display: true, text: 'rating', color: cssv('--muted') }, ticks: { color: cssv('--muted') }, grid: { color: cssv('--line') } } },
      plugins: { legend: { labels: { color: cssv('--ink'), boxWidth: 12 } },
        tooltip: { callbacks: { title: (it) => new Date(it[0].parsed.x).toLocaleString() } } } } });
  }
  // head to head
  const H = c.h2h;
  if (H && H.ids.length) {
    const lab = (id) => (models.find((m) => m.id === id) || { label: id }).label;
    $('h2hnote').textContent = `round ${H.round}`;
    $('h2h').innerHTML = '<tr><th></th>' + H.ids.map((id) => `<th>${esc(lab(id))}</th>`).join('') + '</tr>' +
      H.ids.map((r, i) => `<tr><td><b>${esc(lab(r))}</b></td>` + H.ids.map((col, j) => {
        const v = H.score[i][j];
        if (i === j || v == null) return '<td class="num muted">·</td>';
        const hue = v >= .5 ? 'var(--good)' : 'var(--bad)';
        return `<td class="num" style="background:color-mix(in srgb, ${hue} ${Math.round(Math.abs(v - .5) * 120)}%, transparent)">${Math.round(100 * v)}%</td>`;
      }).join('') + '</tr>').join('');
  }
  $('gifs').innerHTML = (c.replays || []).map((r) => `<figure><img loading="lazy" src="${r.src}" title="click to replay" alt="replay">
    <figcaption>${esc(r.caption)}</figcaption></figure>`).join('') || '<p class="muted">No games yet.</p>';
  document.addEventListener('click', (e) => { if (e.target.tagName === 'IMG' && e.target.closest('.gifs')) { const s = e.target.src.split('#')[0]; e.target.src = s + '#' + Date.now(); } });
  $('timeline').innerHTML = (c.events || []).map((e) => `<div class="ev"><div class="body"><div class="t">${esc(e.t)}</div>
    <h4><span class="tag ${e.kind}">${esc(e.kind.replace(/_/g, ' '))}</span>${esc(e.title)}</h4><p>${esc(e.text || '')}</p></div>
    ${e.replay ? `<img loading="lazy" src="${e.replay}" title="click to replay" alt="replay">` : ''}</div>`).join('') || '<p class="muted">Nothing yet.</p>';
})();
