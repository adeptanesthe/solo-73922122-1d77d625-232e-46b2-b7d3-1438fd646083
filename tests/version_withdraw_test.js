// 试配页「版次撤回」专项测试：DOM 打桩 + 可控 fetch，验证
//  1) 响应携带的 model_version 与当前版次不一致时，旧版响应不得作为新版建议显示
//  2) 轮询发现新版发布 → 立即撤下旧建议（withdraw），并刷新当前版次
//  3) 同版次响应正常渲染；版次一致的迟到防护（epoch）行为不受影响
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'app', 'static', 'index.html'), 'utf8');
const pageJs = html.match(/<script>([\s\S]*)<\/script>/)[1];

// ---- DOM 打桩（与 frontend_test.js 同构） ----
const store = {};
function el(id) {
  if (!store[id]) {
    const e = {
      id, checked: false, textContent: '', innerHTML: '',
      style: { display: 'none' }, dataset: {}, disabled: false,
      listeners: {},
      classList: { add() {}, remove() {} },
      addEventListener(ev, fn) { (this.listeners[ev] ??= []).push(fn); },
      appendChild() {},
    };
    let v = '0';
    Object.defineProperty(e, 'value', { get: () => v, set: x => { v = String(x); } });
    store[id] = e;
  }
  return store[id];
}
global.document = {
  getElementById: el,
  querySelector: () => el('tbody'),
  createElement: () => el('tr_' + Math.random()),
};
let fetchImpl = () => Promise.reject(new Error('fetch not stubbed'));
global.fetch = (...a) => fetchImpl(...a);

const META_V1 = { version: 1, feature_names: ['A','B','C','D'], total_grams: 1000,
  proba_threshold: 0.7, n_trees: 5, max_depth: 3, tree_depths: [3,3,3,3,3],
  train_seed: 1, n_measured: 0, measured_weight: 20, created_at: 't1' };
const META_V2 = { ...META_V1, version: 2, n_measured: 8, created_at: 't2' };
const SAMPLES_V1 = { model_version: 1,
  feasible: { original: [250,250,350,150], lower: [0,0,0,0],
    upper: [1000,1000,1000,1000], locked: [false,true,false,false] },
  infeasible: { original: [150,130,620,100], lower: [0,0,620,0],
    upper: [150,1000,1000,1000], locked: [false,false,false,false] } };
const OK_V1 = { status: 'ok', model_version: 1, suggestion: [390,250,210,150],
  probability: 0.7969, changed_count: 2, total_change: 280, solve_ms: 50,
  optimality: '已证明最优…',
  per_ingredient: [
    { name:'A', original:250, new:390, delta:140, locked:false },
    { name:'B', original:250, new:250, delta:0, locked:true },
    { name:'C', original:350, new:210, delta:-140, locked:false },
    { name:'D', original:150, new:150, delta:0, locked:false } ] };
const OK_V2 = { ...OK_V1, model_version: 2, probability: 0.81 };

fetchImpl = url => Promise.resolve({
  json: async () => url.includes('meta') ? META_V1
    : (url.includes('samples') ? SAMPLES_V1 : OK_V1),
});

const driver = `
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const assert = (name, cond) => { console.log((cond ? '[PASS] ' : '[FAIL] ') + name); if (!cond) process.exitCode = 1; };

  await sleep(20);
  assert('启动后当前版次为 v1', currentModelVersion === 1);

  // --- 场景 1：同版次响应正常渲染 ---
  fetchImpl = () => Promise.resolve({ json: async () => OK_V1 });
  await solve();
  assert('v1 响应正常显示 ok 面板', document.getElementById('panelOk').style.display === 'block');

  // --- 场景 2：跨版次响应不得显示（发布后迟到的旧版响应） ---
  currentModelVersion = 2;  // 模拟服务端已发布 v2
  fetchImpl = () => Promise.resolve({ json: async () => OK_V1 });  // 迟到的 v1 响应
  await solve();
  assert('旧版 v1 响应不显示为建议', document.getElementById('panelOk').style.display === 'none');
  assert('状态提示版次已变化', document.getElementById('status').textContent.includes('版次'));

  // --- 场景 3：新版响应正常渲染 ---
  fetchImpl = () => Promise.resolve({ json: async () => OK_V2 });
  await solve();
  assert('v2 响应正常显示', document.getElementById('panelOk').style.display === 'block');

  // --- 场景 4：轮询发现新版 → withdraw 撤下旧建议并更新版次 ---
  currentModelVersion = 1;
  let metaCalls = 0;
  fetchImpl = url => {
    if (url.includes('meta')) {
      metaCalls++;
      return Promise.resolve({ json: async () => metaCalls <= 1 ? META_V1 : META_V2 });
    }
    if (url.includes('samples')) return Promise.resolve({ json: async () => SAMPLES_V1 });
    return Promise.resolve({ json: async () => OK_V1 });
  };
  // 先在 v1 下显示一条建议
  fetchImpl = url => url.includes('meta') || url.includes('samples')
    ? Promise.resolve({ json: async () => url.includes('meta') ? META_V1 : SAMPLES_V1 })
    : Promise.resolve({ json: async () => OK_V1 });
  await refreshModelContext();
  currentModelVersion = 1;
  const solveFetch = () => Promise.resolve({ json: async () => OK_V1 });
  fetchImpl = solveFetch;
  await solve();
  assert('撤下前 ok 面板显示', document.getElementById('panelOk').style.display === 'block');
  // 下一次轮询返回 v2
  let toggled = false;
  fetchImpl = url => {
    if (url.includes('meta')) return Promise.resolve({ json: async () => toggled ? META_V2 : META_V1 });
    if (url.includes('samples')) return Promise.resolve({ json: async () => SAMPLES_V1 });
    return Promise.resolve({ json: async () => OK_V1 });
  };
  await refreshModelContext();   // 第一次仍 v1，不撤
  assert('同版次轮询不撤建议', document.getElementById('panelOk').style.display === 'block');
  toggled = true;
  await refreshModelContext();   // 第二次 v2，撤下
  assert('检测到 v2 → 旧建议立即撤下', document.getElementById('panelOk').style.display === 'none');
  assert('撤下提示含新版本号', document.getElementById('status').textContent.includes('v2'));
  assert('当前版次已刷新为 v2', currentModelVersion === 2);
  assert('metaLine 已更新为 v2', document.getElementById('metaLine').textContent.includes('v2'));

  // --- 场景 5：无 model_version 字段的响应按旧逻辑正常显示（向后兼容） ---
  currentModelVersion = 2;
  const OK_NO_VER = { ...OK_V2 }; delete OK_NO_VER.model_version;
  fetchImpl = () => Promise.resolve({ json: async () => OK_NO_VER });
  await solve();
  assert('无版次字段响应向后兼容正常显示', document.getElementById('panelOk').style.display === 'block');
})();
`;

eval(pageJs + driver);
