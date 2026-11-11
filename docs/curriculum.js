(async function () {
  const [cur, status] = await Promise.all([get('curricula.json'), get('status.json')]);
  const runs = (status && status.runs) || [];
  $('runs').innerHTML = runs.slice().reverse().filter((r) => cur && cur[r.run]).map((r) => {
    const c = cur[r.run];
    const stIdx = c.stages.findIndex((s) => s.name === (r.last || {}).stage);
    return `<h2>Run ${esc(r.run)} <span class="muted small">${r.alive ? 'training now' : 'stopped'}</span></h2>
      <p>${esc(c.note || '')}</p>` + c.stages.map((s, i) => {
      const passed = r.passed.includes(s.name);
      const cls = passed ? 'passed' : (i === stIdx && r.alive) ? 'current' : '';
      const sup = s.support ? `<dt>Training wheels</dt><dd>${esc(kv({ support: s.support }))}, then off; checks run without help</dd>` : '';
      const hard = s.hard_p ? `<dt>Hard cases</dt><dd>${pct(s.hard_p, 0)} of episodes repeat situations the agent recently failed</dd>` : '';
      return `<div class="stage ${cls}"><div class="n">${i + 1}</div><div>
        <h3>${esc(s.name)} <span class="muted small">· ${passed ? 'passed' : i === stIdx ? 'running' : i < stIdx ? 'done' : 'waiting'} · cap ${s.cap}</span></h3>
        <dl><dt>World</dt><dd>${esc(worldText(s.world, s.ramp))}</dd>${sup}${hard}
        <dt>Reward</dt><dd>${esc(rewardText(s.reward))}</dd><dt>Pass</dt><dd>${esc(passText(s.pass))}</dd>
        <dt>Expected</dt><dd>${esc(s.expect || '')}</dd></dl></div></div>`;
    }).join('');
  }).join('');
})();
