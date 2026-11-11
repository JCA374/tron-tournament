// Shared by every page: navigation, helpers, formatters. Site settings come from site.js (written by publish.py from
// site.json). A project can override worldText / rewardText / passText / eventText in project.js (loaded after this).
const SITE = window.SITE || { title: 'Project', repo: '', tabs: [] };
const TABS = SITE.tabs && SITE.tabs.length ? SITE.tabs : [['index.html', 'Log'], ['curriculum.html', 'Curriculum'],
  ['senses.html', 'How it sees'], ['model.html', 'The model'], ['play.html', 'Play'], ['speed.html', 'Speed'],
  ['lessons.html', 'Lessons'], ['podcast/index.html', 'Podcast']];
(function () {
  const here = location.pathname.split('/').pop() || 'index.html';
  const nav = document.createElement('nav'); nav.className = 'top';
  nav.innerHTML = `<a class="brand" href="index.html">${SITE.icon || '🧪'} ${SITE.title}</a>` + TABS.map(([h, t]) =>
    `<a href="${h}" class="${h === here ? 'on' : ''}">${t}</a>`).join('') + (SITE.repo ? `<a href="https://github.com/${SITE.repo}">Code</a>` : '');
  document.body.prepend(nav);
  // replay GIFs play once; click to watch again
  document.addEventListener('click', (e) => {
    if (e.target.tagName === 'IMG' && e.target.closest('.ev')) { const s = e.target.src.split('#')[0]; e.target.src = s + '#' + Date.now(); }
  });
})();
const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x === '' || x == null || isNaN(x)) ? '–' : (100 * x).toFixed(d) + '%';
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const get = (f) => fetch('data/' + f + '?t=' + Date.now()).then((r) => r.ok ? r.json() : null).catch(() => null);
const cssv = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const kv = (o = {}) => Object.entries(o).map(([k, v]) => `${k.replace(/_/g, ' ')} ${Array.isArray(v) ? v.join('→') : v}`).join(', ');
window.worldText = window.worldText || ((w, ramp) => kv(w) + (ramp && Object.keys(ramp).length ? '; ramps: ' + kv(ramp) : ''));
window.rewardText = window.rewardText || ((r) => kv(r) || 'default');
window.passText = window.passText || ((p) => kv(p));
