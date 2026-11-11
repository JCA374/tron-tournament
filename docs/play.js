// Play tab: pick a model and a game; the clip plays once (never loops). data/play.json is written by publish.py.
(async function () {
  const d = await get('play.json');
  if (!d || !d.models.length) { $('player').innerHTML = '<p class="muted" style="padding:16px">No clips yet.</p>'; return; }
  let mi = 0, gi = 0;
  const vid = $('video');
  vid.loop = false;                                    // play once: a clip ends on its outcome
  function draw() {
    const m = d.models[mi], g = d.games[gi], c = m.clips[g.id];
    $('models').innerHTML = d.models.map((x, i) => `<button class="${i === mi ? 'on' : ''}" data-m="${i}">${esc(x.label)}</button>`).join('');
    $('mnote').textContent = m.note || '';
    $('games').innerHTML = d.games.map((x, i) => {
      const cc = m.clips[x.id];
      const b = cc ? `<span class="badge ${cc.ok ? 'ok' : 'bad'}">${esc(cc.result)}</span>` : '';
      return `<button class="game ${i === gi ? 'on' : ''}" data-g="${i}"><div class="ttl">${esc(x.title)}</div><div class="d">${esc(x.desc || '')}</div>${b}</button>`;
    }).join('');
    $('caption').innerHTML = c ? `<b>${esc(m.label)}</b> on <b>${esc(g.title)}</b>: <span class="badge ${c.ok ? 'ok' : 'bad'}">${esc(c.result)}</span> ${esc(c.detail || '')}` : 'not rendered yet';
    if (c && !vid.src.endsWith(c.src)) { vid.src = c.src; vid.play().catch(() => {}); }
    $('cmp').innerHTML = d.models.map((x) => { const cc = x.clips[g.id]; return cc ? `<figure><video src="${cc.src}" controls preload="metadata" muted playsinline></video>
      <figcaption>${esc(x.label)} · <span class="badge ${cc.ok ? 'ok' : 'bad'}">${esc(cc.result)}</span></figcaption></figure>` : ''; }).join('');
  }
  document.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.m) { mi = +b.dataset.m; draw(); }
    if (b.dataset.g) { gi = +b.dataset.g; draw(); }
  });
  $('next').onclick = () => { gi = (gi + 1) % d.games.length; draw(); };
  vid.addEventListener('ended', () => { $('next').classList.add('on'); setTimeout(() => $('next').classList.remove('on'), 1500); });
  draw();
})();
