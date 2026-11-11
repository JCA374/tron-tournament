// Log page. Everything project-specific comes from site.json (status.config): goal, chart series, cards, names.
function countdown(deadline, text) {
  if (!deadline) return;
  const tick = () => {
    const s = Math.max(0, (new Date(deadline) - new Date()) / 1000);
    const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
    $('countdown').textContent = s ? `⏱ ${d} d ${h} h ${m} min left${text ? ' until ' + text : ''}` : 'Deadline reached.';
  };
  tick(); setInterval(tick, 30000);
}
async function main() {
  const [status, stats, log] = await Promise.all([get('status.json'), get('stats.json'), get('log.json')]);
  if (!status) { $('updated').textContent = 'No data yet.'; return; }
  const C = status.config || {};
  countdown(C.deadline, C.deadline_text);
  $('updated').textContent = `Updated ${status.updated}.`;
  const metric = (C.goal || {}).metric || 'landed', target = (C.goal || {}).target;
  const rows = [];
  for (const [k, r] of Object.entries(status.goals || {})) rows.push([r.label || k, r, r.note || 'goal check']);
  for (const [k, r] of Object.entries(status.baselines || {})) rows.push([(C.names || {})[k] || k, r, 'baseline']);
  for (const [k, r] of Object.entries(status.bakeoff || {})) rows.push([(C.names || {})[k] || k, r, 'bake-off']);
  rows.sort((a, b) => (b[1][metric] ?? 0) - (a[1][metric] ?? 0));
  const extra = (C.goal || {}).columns || [];
  $('goaltable').innerHTML = `<tr><th>Agent</th><th>${esc((C.goal || {}).label || metric)}</th><th>95% interval</th>${extra.map((c) => `<th>${esc(c[1])}</th>`).join('')}<th>n</th></tr>` +
    rows.map(([n, r, note]) => `<tr><td>${esc(n)}<div class="muted small">${esc(note)}</div></td><td class="num"><b>${pct(r[metric])}</b></td>
      <td class="num">${r.ci ? pct(r.ci[0]) + ' – ' + pct(r.ci[1]) : '–'}</td>${extra.map((c) => `<td class="num">${pct(r[c[0]])}</td>`).join('')}<td class="num">${r.n ?? ''}</td></tr>`).join('') +
    (target ? `<tr><td><b>Goal</b></td><td class="num"><b>${pct(target)}</b></td><td colspan="${2 + extra.length}" class="muted">${esc(C.goal_text || '')}</td></tr>` : '');
  const runs = status.runs || [];
  const main = runs.find((r) => r.alive) || runs[runs.length - 1];
  if (main) {
    const l = main.last || {};
    const card = (c) => { let v = l[c.key]; if (c.fmt === 'pct') v = pct(+v); if (c.fmt === 'M') v = v ? (v / 1e6).toFixed(1) + ' M' : '–'; return [c.label, v ?? '–', c.sub || '']; };
    $('cards').innerHTML = [['Run', main.run, main.alive ? 'training now' : 'stopped'], ['Stage', l.stage || '–', `${main.passed.length} stage(s) passed`],
      ...(C.cards || []).map(card), ['Best check', main.best_goal ? pct(main.best_goal[metric]) : '–', 'best of many: flatters']]
      .map(([k, v, s]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`).join('');
    chart((stats || {})[main.run] || [], main.run, C.chart || {}, target);
  }
  $('log').innerHTML = (log || []).map(eventHtml).join('') || '<p class="muted">Nothing yet.</p>';
}
function chart(rows, run, cfg, target) {
  if (!window.Chart || !rows.length || !cfg.series) return;
  const X = (r) => +r[cfg.x || 'it'] * (cfg.x_scale || 1);
  const ds = cfg.series.map((s) => {
    const pts = rows.filter((r) => r[s.key] !== '' && r[s.key] != null).map((r) => ({ x: X(r), y: (s.scale ?? 100) * r[s.key] }));
    const col = cssv(s.color || '--accent');
    return { label: s.label, data: pts, borderColor: col, backgroundColor: col, showLine: s.style !== 'points', pointRadius: s.style === 'line' ? 0 : 2.5, borderWidth: 1.5, tension: .2 };
  });
  const ys = ds.flatMap((d) => d.data.map((p) => p.y));
  const yMin = ys.length ? Math.max(0, Math.floor((Math.min(...ys) - 2) / 5) * 5) : 0;     // autoscale
  const yMax = cfg.y_max ?? Math.ceil((Math.max(...ys, target ? 100 * target : 0) + 1) / 2) * 2;
  if (target) ds.push({ label: `goal ${pct(target, 0)}`, data: [{ x: X(rows[0]), y: 100 * target }, { x: X(rows[rows.length - 1]), y: 100 * target }], borderColor: cssv('--bad'), borderDash: [5, 4], borderWidth: 1, pointRadius: 0 });
  $('chartnote').textContent = `Run ${run}. x axis: ${cfg.x_label || cfg.x || 'iteration'}.`;
  new Chart($('chart'), { type: 'line', data: { datasets: ds }, options: { animation: false, maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    scales: { x: { type: 'linear', ticks: { color: cssv('--muted') }, grid: { color: cssv('--line') } },
      y: { min: yMin, max: yMax, title: { display: true, text: cfg.y_label || '%', color: cssv('--muted') }, ticks: { color: cssv('--muted') }, grid: { color: cssv('--line') } } },
    plugins: { legend: { labels: { color: cssv('--ink'), boxWidth: 12 } } } } });
}
function chk(c) { return c ? Object.entries(c).filter(([k, v]) => typeof v === 'number' && k !== 'n').map(([k, v]) => `${k} ${v <= 1 ? pct(v) : v.toFixed(1)}`).join(', ') + (c.n ? ` (n=${c.n})` : '') : ''; }
function eventHtml(e) {
  let [title, body] = (window.eventText && window.eventText(e)) || [null, null];
  if (title == null) {
    switch (e.kind) {
      case 'journal': title = e.title; body = e.text; break;
      case 'run_start': title = `Run ${e.run} started`; body = e.note; break;
      case 'stage_start': title = `Stage started: ${e.stage}`; body = e.expect ? 'Expected: ' + e.expect : ''; break;
      case 'goal_best': title = 'Best check so far'; body = chk(e.check) + "\nBest of many checks: flattered by luck (winner's curse)."; break;
      case 'check_pass': title = `Check passed in ${e.stage}, confirming on a fresh block`; body = `First: ${chk(e.check)}\nConfirm: ${chk(e.confirm)}`; break;
      case 'stage_pass': title = `✅ Stage passed: ${e.stage}`; body = `Check: ${chk(e.check)}\nConfirm: ${chk(e.confirm)}`; break;
      case 'stage_cap': title = `Stage ${e.stage} reached its cap; moving on`; break;
      case 'crash': title = '⚠️ Training crashed and was restarted from its last checkpoint'; body = e.note; break;
      case 'override': title = `Manual change: ${e.what}`; body = e.reason; break;
      case 'goal_check': title = `🎯 Goal check: ${pct(e.landed ?? e.value)} on ${e.n} unseen episodes`; body = `${e.label || e.policy || ''}. 95% interval ${e.ci ? pct(e.ci[0]) + ' – ' + pct(e.ci[1]) : '–'}. ${e.note || ''}`; break;
      default: title = e.kind.replace(/_/g, ' '); body = e.note || '';
    }
  }
  const img = e.replay ? `<img loading="lazy" src="${e.replay}" title="click to replay" alt="replay of the agent at this moment">` : '';
  return `<div class="ev"><div class="body"><div class="t">${esc(e.t)}${e.run ? ' · run ' + esc(e.run) : ''}</div>
    <h4><span class="tag ${e.kind}">${e.kind.replace(/_/g, ' ')}</span>${esc(title)}</h4><p>${esc(body)}</p></div>${img}</div>`;
}
main();
