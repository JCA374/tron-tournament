// Project-specific names and event texts.
window.MODELS = { ppo: 'PPO', dqn: 'DQN', neat: 'NEAT', cmaes: 'CMA-ES', az: 'Mini-AlphaZero', tournament: 'Tournament', journal: 'Journal', all: 'All' };
window.runLabel = (r) => window.MODELS[r] || r;
window.eventText = (e) => {
  switch (e.kind) {
    case 'run_start': return [`${runLabel(e.run)} started training`, e.note || ''];
    case 'resume': return [`${runLabel(e.run)} resumed from its checkpoint`, e.note || ''];
    case 'crash': return [`⚠️ ${runLabel(e.run)} crashed and was restarted from its last checkpoint`, e.note || ''];
    case 'change': return [`Change to ${runLabel(e.run)}: ${e.what}`, (e.reason ? 'Why: ' + e.reason : '') + (e.before ? '\nBefore: ' + e.before : '')];
    default: return [e.title || null, e.text || e.note || ''];
  }
};
