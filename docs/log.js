// Log page: data/log.json (journal + runs/*/events.jsonl, written by publish.py).
(async function () {
  const [log, status] = await Promise.all([get('log.json'), get('status.json')]);
  if (status) $('updated').textContent = `Updated ${status.updated}.`;
  const items = log || [];
  const runs = ['all', ...new Set(items.map((e) => e.run || (e.kind === 'journal' ? 'journal' : '')).filter(Boolean))];
  let pick = 'all';
  function draw() {
    $('filter').innerHTML = runs.map((r) => `<button class="${r === pick ? 'on' : ''}" data-r="${r}">${esc(runLabel(r))}</button>`).join('');
    const shown = items.filter((e) => pick === 'all' || (e.run || (e.kind === 'journal' ? 'journal' : '')) === pick);
    $('log').innerHTML = shown.slice(0, 400).map(eventHtml).join('') || '<p class="muted">Nothing yet.</p>';
  }
  document.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-r]'); if (b) { pick = b.dataset.r; draw(); }
    if (e.target.tagName === 'IMG' && e.target.closest('.ev')) { const s = e.target.src.split('#')[0]; e.target.src = s + '#' + Date.now(); }
  });
  draw();
})();
function eventHtml(e) {
  let [title, body] = (window.eventText && window.eventText(e)) || [null, null];
  if (title == null) {
    if (e.kind === 'journal') { title = e.title; body = e.text; }
    else { title = e.title || e.kind.replace(/_/g, ' '); body = e.text || e.note || ''; }
  }
  const img = e.replay ? `<img loading="lazy" src="${e.replay}" title="click to replay" alt="replay">` : '';
  return `<div class="ev"><div class="body"><div class="t">${esc(e.t)}${e.run ? ' · ' + esc(runLabel(e.run)) : ''}</div>
    <h4><span class="tag ${e.kind}">${e.kind.replace(/_/g, ' ')}</span>${esc(title)}</h4><p>${esc(body)}</p></div>${img}</div>`;
}
