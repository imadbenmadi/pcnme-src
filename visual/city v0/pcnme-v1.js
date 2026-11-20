(function () {

const COL = { bg:0x0b0e13, ground:0x0f141a, road:0x1c232c, dash:0x3a4552, dome:0x7f9bb8, anchor:0xff6fa8, serving:0x4da3ff, exec:0x3ddc84, car:0xffd23f, broker:0x9aa3ad, cloud:0x9fd4ff, task:0x4da3ff, result:0x3ddc84, ack:0xffd23f, state:0x8a94a3, attach:0xffffff, gps:0xffffff, dead:0x3a4049, wan:0x9fd4ff, hot:0xff4545, link:0x5f6b78, led:0x1c2b24, fail:0xff4545 };
const H6 = n => '#' + n.toString(16).padStart(6, '0');
const ZDEF = [
  { id:'Z1', x:-18, z:0, r:8.5, side:-1 },
  { id:'Z2', x:-5, z:0, r:9, side:1 },
  { id:'Z3', x:18, z:0, r:8, side:-1 },
  { id:'Z4', x:12, z:-16, r:9, side:1 },
  { id:'Z5', x:-17, z:-16, r:7.5, side:-1, late:true },
];
const NEIGH = { Z1:['Z2','Z5',null,null], Z2:['Z1','Z3','Z4','Z5'], Z3:['Z2','Z4',null,null], Z4:['Z3','Z2','Z5',null], Z5:['Z1','Z4','Z2',null] };
const PEERS = [['Z1','Z2'],['Z1','Z5'],['Z2','Z3'],['Z2','Z4'],['Z2','Z5'],['Z3','Z4'],['Z4','Z5']];
const RING = [[-28,0],[28,0],[28,-16],[-28,-16]], RL = 144;
const BROKER = { x:2, z:-8 }, CLOUD_Y = 27, HERO = 7, HYST = 0.3, LIGHT_X = -11.5;
const SANS = "'IBM Plex Sans', sans-serif", MONO = "'IBM Plex Mono', monospace";
const PCOL = { task:COL.task, result:COL.result, ack:COL.ack, state:COL.state, attach:COL.attach, gps:COL.gps };

const FLOWS = [
  { n:1, code:'C2·C4', title:'Normal task' },
  { n:2, code:'C1', title:'Boulder' },
  { n:3, code:'C7·C3', title:'Zone change' },
  { n:4, code:'C13', title:'Overlap' },
  { n:5, code:'C8', title:'Car stopped' },
  { n:6, code:'C12', title:'Coverage gap' },
  { n:7, code:'C14', title:'Exec failure' },
  { n:8, code:'C9', title:'Deadline' },
  { n:9, code:'—', title:'Zone dies / joins' },
  { n:10, code:'S3', title:'Cloud congestion' },
];

function ringPos(s, lane) {
  s = ((s % RL) + RL) % RL;
  for (let i = 0; i < 4; i++) {
    const a = RING[i], b = RING[(i + 1) % 4], L = Math.hypot(b[0]-a[0], b[1]-a[1]);
    if (s <= L) { const dx = (b[0]-a[0])/L, dz = (b[1]-a[1])/L; return { x:a[0]+dx*s - dz*lane, z:a[1]+dz*s + dx*lane, dx, dz }; }
    s -= L;
  }
  return { x:-28, z:0, dx:1, dz:0 };
}
const equip = z => ({ x:z.x, z:z.z + z.side*3.4 });
const kind = id => id.split(':')[0];
const isWan = (a, b) => a === 'cloud' || b === 'cloud';
function along(pts, t, d) {
  if (pts.length === 2) return [pts[0], pts[1], t];
  const ls = []; let L = 0;
  for (let i = 0; i < pts.length-1; i++) { const v = d(pts[i], pts[i+1]); ls.push(v); L += v; }
  let r = t*L;
  for (let i = 0; i < ls.length; i++) { if (r <= ls[i] || i === ls.length-1) return [pts[i], pts[i+1], ls[i] ? Math.min(1, r/ls[i]) : 0]; r -= ls[i]; }
}

async function createPCNME(root, cb = {}) {
  const call = (k, ...a) => { try { cb[k] && cb[k](...a); } catch (e) { console.error(e); } };
  root.innerHTML = '';
  const S = { insets:{ left:0, right:0, top:72, bottom:56 }, speed:1, ambient:true, radio:true, time:0, paused:false, stepMode:false, gateRes:null,
    zones:{}, cars:[], packets:[], timers:[], waiters:[], tags:[], fx:[], wanLoad:0, wanMs:50, cloudSlots:new Array(24).fill(0), cloud:{ queue:0 },
    flowActive:false, flow:null, sel:null, ambT:1, cloudT:2, gpsT:0, light:{ on:false, red:true }, sigNoise:false, heroAnchor:null, inflight:{}, hover:null };
  ZDEF.forEach(d => S.zones[d.id] = { ...d, alive:true, joined:!d.late, fade:d.late?0:1, slots:[0,0,0,0], failT:[0,0,0,0], queue:0, qx:0, held:false, anchored:0, replay:0, roles:new Set() });
  const SPEC = [[1,10,1,2.2],[2,40,-1,2.5],[3,64,1,2.0],[4,90,-1,2.3],[5,112,1,2.6],[6,130,-1,2.1],[7,3,1,2.2],[8,80,1,2.4],[9,22,-1,1.9],[10,100,-1,2.2]];
  S.cars = SPEC.map(([id,s,dir,v]) => ({ id, s, dir, v, v0:v, serv:null }));
  const CAR = {}; S.cars.forEach(c => { CAR[c.id] = c; S.inflight[c.id] = 0; });
  const hero = CAR[HERO];
  const carPos = c => ringPos(c.s, c.dir > 0 ? 0.8 : -0.8);
  const zl = () => Object.values(S.zones);
  const nb = id => NEIGH[id].map(n => n && S.zones[n].joined ? n : null);
  function sig(z, p, c) {
    let s = 1 - Math.hypot(p.x-z.x, p.z-z.z)/z.r;
    if (S.sigNoise && c === hero) s += 0.1*Math.sin(S.time*4.2 + ZDEF.findIndex(d => d.id === z.id)*2.1);
    return s;
  }
  function serving(c) {
    const p = carPos(c);
    if (p.x > 4.5 && p.x < 10.5 && Math.abs(p.z) < 2.5) return null;
    const av = zl().filter(z => z.alive && z.joined && Math.hypot(p.x-z.x, p.z-z.z) <= z.r);
    if (!av.length) return null;
    let best = av[0], bs = -9;
    for (const z of av) { const s = sig(z, p, c); if (s > bs) { bs = s; best = z; } }
    const cur = av.find(z => z.id === c.serv);
    if (!cur) return best.id;
    return (best !== cur && bs > sig(cur, p, c) + HYST) ? best.id : cur.id;
  }
  S.cars.forEach(c => c.serv = serving(c));
  const servOf = id => CAR[id].serv;
  function tExit(c) {
    if (!c.serv) return '—';
    if (c.v === 0 || S.paused) return c.v === 0 ? '∞' : '—';
    const z = S.zones[c.serv];
    for (let d = .5; d <= 40; d += .5) { const p = ringPos(c.s + c.dir*d, c.dir > 0 ? .8 : -.8); if (Math.hypot(p.x-z.x, p.z-z.z) > z.r) return (d/c.v).toFixed(1) + ' s'; }
    return '> 16 s';
  }

  // engine
  const simWait = s => new Promise(res => S.timers.push({ t:s, res }));
  const simUntil = f => new Promise(res => S.waiters.push({ f, res }));
  function segDur(a, b, type) {
    const m = (type === 'gps' || type === 'state') ? .6 : 1;
    if (isWan(a, b)) return Math.min(4.5, 1.8*(1 + S.wanLoad*.12));
    const ka = kind(a), kb = kind(b);
    if (ka === 'car' || kb === 'car') return .9*m;
    if (ka === 'ap' || kb === 'ap') return .35*m;
    return .85*m;
  }
  const fx = (id, color) => S.fx.push({ id, color, t:0 });
  function enterSeg(p) {
    const i = p.seg + 1;
    if (typeof p.ids[i] === 'function') p.ids[i] = p.ids[i]();
    if (!p.ids[i]) { fx(p.ids[p.seg], COL.hot); return false; }
    p.wan = isWan(p.ids[p.seg], p.ids[i]);
    if (p.wan) S.wanLoad++;
    p.dur = segDur(p.ids[p.seg], p.ids[i], p.type); p.t = 0;
    return true;
  }
  function send(type, ids, o = {}) {
    return new Promise(res => {
      const p = { type, ids:ids.slice(), seg:0, t:0, dur:1, done:res, size:o.size || 1, faint:!!o.faint, from:o.from };
      if (typeof p.ids[0] === 'function') p.ids[0] = p.ids[0]();
      if (!enterSeg(p)) { res(false); return; }
      S.packets.push(p);
    });
  }
  function killP(p, ok) { const i = S.packets.indexOf(p); if (i >= 0) S.packets.splice(i, 1); if (p.wan) { S.wanLoad--; p.wan = false; } p.done(ok); }
  function toCar(id) { let z; return [() => { z = servOf(id); return z ? 'fog:'+z : null; }, () => 'ap:'+z, 'car:'+id]; }

  function step(dt) {
    const sdt = S.paused ? 0 : dt*S.speed;
    S.time += sdt;
    for (const c of S.cars) { c.s += c.dir*c.v*sdt; if (sdt) c.serv = serving(c); }
    for (const t of S.timers.slice()) { t.t -= sdt; if (t.t <= 0) { S.timers.splice(S.timers.indexOf(t), 1); t.res(); } }
    if (sdt) for (const w of S.waiters.slice()) if (w.f()) { S.waiters.splice(S.waiters.indexOf(w), 1); w.res(); }
    for (const p of S.packets.slice()) {
      p.t += sdt/p.dur;
      if (p.t >= 1) {
        if (p.wan) { S.wanLoad--; p.wan = false; }
        p.seg++;
        if (p.seg >= p.ids.length-1) killP(p, true);
        else if (!enterSeg(p)) killP(p, false);
      }
    }
    for (const z of zl()) {
      z.slots = z.slots.map(v => Math.max(0, v - sdt));
      z.failT = z.failT.map(v => Math.max(0, v - sdt));
      z.fade += ((z.joined ? 1 : 0) - z.fade)*Math.min(1, dt*2.5);
    }
    S.cloudSlots = S.cloudSlots.map(v => Math.max(0, v - sdt));
    S.tags = S.tags.filter(t => (t.ttl -= sdt) > 0 || (t.el && t.el.remove(), false));
    S.fx = S.fx.filter(f => (f.t += sdt || 0) < .9);
    const target = 50 + Math.min(150, Math.max(0, S.wanLoad)*14);
    S.wanMs += (target - S.wanMs)*Math.min(1, dt*2);
    if (!sdt) return;
    if (S.ambient) {
      S.ambT -= sdt; S.cloudT -= sdt;
      if (S.ambT <= 0) {
        S.ambT = .7 + Math.random()*.7;
        const zs = zl().filter(z => z.alive && z.joined), z = zs[Math.floor(Math.random()*zs.length)];
        if (z) { if (Math.random() < .6) send('state', ['fog:'+z.id, 'broker'], { faint:true, size:.6 }); else send('state', ['broker', 'fog:'+z.id], { faint:true, size:.6, from:'cloud' }); }
      }
      if (S.cloudT <= 0) { S.cloudT = 3 + Math.random()*2; send('state', ['cloud', 'broker'], { faint:true, size:.6 }); }
    }
    if (S.flowActive && hero.serv) {
      S.gpsT -= sdt;
      if (S.gpsT <= 0) {
        S.gpsT = .8;
        const s = hero.serv, a = S.heroAnchor;
        send('gps', ['car:'+HERO, 'ap:'+s, 'fog:'+s], { size:.55 });
        if (a && a !== s && S.zones[a].alive) send('gps', ['fog:'+s, 'broker', 'fog:'+a], { size:.55 });
      }
    }
  }

  // flows
  let tok = { dead:true };
  const CANCEL = {};
  function openGate() { if (S.gateRes) { const r = S.gateRes; S.gateRes = null; r(); } call('onGate', false); }
  function reset() {
    tok.dead = true;
    openGate();
    S.packets.slice().forEach(p => killP(p, false)); S.packets = [];
    S.timers.splice(0).forEach(t => t.res());
    S.waiters.splice(0).forEach(w => w.res());
    S.tags.forEach(t => t.el && t.el.remove()); S.tags = [];
    S.wanLoad = 0; S.cloudSlots.fill(0); S.cloud.queue = 0;
    for (const z of zl()) { z.alive = true; if (z.late) z.joined = false; z.roles.clear(); z.slots = [0,0,0,0]; z.failT = [0,0,0,0]; z.queue = 0; z.qx = 0; z.held = false; z.anchored = 0; }
    for (const k in S.inflight) S.inflight[k] = 0;
    S.cars.forEach(c => c.v = c.v0);
    S.light.on = false; S.sigNoise = false; S.heroAnchor = null;
    S.flowActive = false; S.flow = null;
    call('onDecision', null); call('onFlow', null); call('onTopic', null);
  }

  async function runFlow(n) {
    reset();
    const my = { dead:false, beats:0 }; tok = my;
    S.flowActive = true; S.flow = n; call('onFlow', n); call('onClear');
    const ck = () => { if (my.dead) throw CANCEL; };
    let dec = null;
    const pushDec = () => call('onDecision', dec && { ...dec, options:dec.options && dec.options.map(o => ({ ...o })) });
    const X = {
      wait: async s => { await simWait(s); ck(); },
      until: async f => { await simUntil(f); ck(); },
      send: async (t, ids, o) => { const r = await send(t, ids, o); ck(); return r; },
      fan: async (t, lists, o) => { const r = await Promise.all(lists.map(ids => send(t, ids, o))); ck(); return r; },
      say: async (text, k = 'info', topic) => {
        if (S.stepMode && my.beats > 0) { call('onGate', true); await new Promise(r => S.gateRes = r); ck(); }
        my.beats++;
        call('onLog', { text, kind:k });
        if (topic !== undefined) call('onTopic', topic);
        await simWait(.55); ck();
      },
      topic: t => call('onTopic', t),
      exec: async (node, d) => {
        const arr = node === 'cloud' ? S.cloudSlots : S.zones[node].slots, q = node === 'cloud' ? S.cloud : S.zones[node];
        let i = arr.findIndex(v => v <= 0);
        if (i < 0) { q.queue++; await simUntil(() => (i = arr.findIndex(v => v <= 0)) >= 0 || my.dead); q.queue = Math.max(0, q.queue-1); ck(); }
        arr[i] = d; await simWait(d); ck();
      },
      fail: async (node, after) => {
        const z = S.zones[node]; let i = z.slots.findIndex(v => v <= 0); if (i < 0) i = 0;
        z.slots[i] = 99; await simWait(after); z.slots[i] = 0; z.failT[i] = 2.5; fx('fog:'+node, COL.fail); ck();
      },
      busy: (node, k, d, q) => { const z = S.zones[node]; for (let i = 0; i < k; i++) z.slots[i] = Math.max(z.slots[i], d); z.qx = q || 0; simWait(d).then(() => z.qx = 0); },
      role: (z, r) => S.zones[z].roles.add(r),
      unrole: (z, r) => S.zones[z].roles.delete(r),
      hero: (x, v) => { hero.s = x + 28; hero.v = v; hero.serv = null; hero.serv = serving(hero); },
      speed: v => hero.v = v,
      tag: (id, text, color, ttl = 3, o = {}) => { const t = { id, text, color:H6(color), ttl, fn:o.fn, dashed:o.dashed }; S.tags.push(t); return t; },
      untag: t => { t.ttl = 0; },
      anchor: z => { X.role(z, 'anchor'); S.heroAnchor = z; S.zones[z].anchored++; S.inflight[HERO]++; X.tag('fog:'+z, 'input stored · subscribed car/7/attach·gps·result·ack', COL.anchor, 3.2); },
      cleanup: z => { S.zones[z].anchored = Math.max(0, S.zones[z].anchored-1); S.inflight[HERO] = Math.max(0, S.inflight[HERO]-1); S.zones[z].replay++; S.heroAnchor = null; X.tag('fog:'+z, 'input + result deleted · unsubscribed', COL.anchor, 2.6); },
      tof: (zone, kind, text) => { dec = { zone, stage:1, tof:{ kind, text }, options:null, note:'', reward:'' }; pushDec(); X.tag('fog:'+zone, 'TOF-Broker: ' + kind, kind === 'boulder' ? COL.cloud : COL.exec, 2.4); },
      masks: (zone, over = {}) => {
        const n4 = nb(zone);
        dec.options = ['here','n1','n2','n3','n4','cloud'].map((k, i) => {
          const target = k === 'here' ? zone : k === 'cloud' ? 'cloud' : n4[i-1];
          let why = over[k] || '';
          if (!why && k[0] === 'n') { if (!target) why = 'no neighbour'; else if (!S.zones[target].alive) why = 'offline'; }
          const label = k === 'here' ? 'here · ' + zone : k === 'cloud' ? 'cloud' : 'N' + i + ' · ' + (target || '—');
          return { key:k, label, target, why, masked:!!why, q:0, chosen:false };
        });
        dec.stage = 2; pushDec();
      },
      choose: (chosen, qs, note) => {
        dec.options.forEach(o => { o.q = o.masked ? 0 : (qs[o.key] ?? .2); o.chosen = o.key === chosen; });
        dec.stage = 3; dec.note = note || ''; pushDec();
        const o = dec.options.find(o => o.chosen);
        if (o) X.tag('fog:'+dec.zone, 'D3QN → ' + (o.target === 'cloud' ? 'cloud' : o.target), COL.exec, 3);
      },
      note: t => { dec.note = t; pushDec(); },
      reward: t => { if (dec) { dec.reward = t; pushDec(); } },
      publish: async (exec, anchor) => {
        const src = exec === 'cloud' ? 'cloud' : 'fog:'+exec;
        X.tag(src, 'PUBLISH car/7/result · once', COL.result, 2.2);
        await X.send('result', [src, 'broker']);
        const s = servOf(HERO), lists = [];
        if (s) lists.push(['broker', 'fog:'+s]); else fx('broker', COL.hot);
        if (anchor && anchor !== s && anchor !== exec && S.zones[anchor].alive) lists.push(['broker', 'fog:'+anchor]);
        if (lists.length) await X.fan('result', lists, { from:src });
        if (anchor && anchor !== s && anchor !== exec) X.tag('fog:'+anchor, 'result copy kept', COL.result, 2.4);
        return s;
      },
      deliver: async () => { await X.send('result', toCar(HERO)); },
      ack: async anchor => {
        const s = servOf(HERO); if (!s) return;
        await X.send('ack', ['car:'+HERO, 'ap:'+s, 'fog:'+s]);
        X.topic('ack');
        if (s !== anchor) { X.tag('fog:'+s, 'PUBLISH car/7/ack', COL.ack, 2); await X.send('ack', ['fog:'+s, 'broker', 'fog:'+anchor]); }
        else X.tag('fog:'+s, 'car/7/ack → anchor (same zone)', COL.ack, 2);
      },
      up: async z => { await X.send('task', ['car:'+HERO, 'ap:'+z, 'fog:'+z]); },
    };
    const FL = {
      1: async () => {
        X.hero(-25, .4); X.busy('Z1', 4, 14, 2);
        await X.say('Car 7 captures a frame → YOLO task, hard deadline (example 150 ms). Uploads it by radio: HTTP POST /tasks.', 'task', null);
        await X.up('Z1');
        X.anchor('Z1');
        await X.say('Z1 is the anchor: stores the input copy and subscribes to car/7/attach, gps, result, ack.', 'state');
        await X.wait(.8);
        X.tof('Z1', 'pebble', 'exec on strongest fog class < α × deadline');
        await X.say('TOF-Broker: pebble → the agent decides.');
        X.masks('Z1', { here:'queue full' });
        await X.say('Masks: Z1’s queue is full; N2–N4 have no neighbour.');
        await X.wait(.6);
        X.choose('n1', { n1:.81, cloud:.33 }, 'Highest unmasked Q-value wins.');
        await X.say('D3QN picks N1 · Z2.', 'info');
        X.role('Z2', 'exec');
        await X.say('Task input goes to Z2 over backhaul (HTTP). One task, one node.', 'task', null);
        await X.send('task', ['fog:Z1', 'fog:Z2']);
        S.zones.Z2.qx = 1; await X.say('Z2: task waits in the priority queue, then runs in a free slot.'); S.zones.Z2.qx = 0;
        await X.exec('Z2', 2.2);
        await X.say('Z2 publishes car/7/result once → broker → serving zone Z1 (here also the anchor).', 'result', 'result');
        await X.publish('Z2', 'Z1');
        await X.say('Z1 pushes the result to car 7 by radio (WebSocket).', 'result');
        await X.deliver();
        await X.say('Car 7 confirms to its serving zone → car/7/ack to the anchor.', 'ack', 'ack');
        await X.ack('Z1');
        X.reward('on time → +1 − w·latency − energy');
        X.cleanup('Z1');
        await X.say('Anchor Z1 computes the reward, deletes input + result copy, unsubscribes.', 'done', null);
        await X.wait(1);
      },
      2: async () => {
        X.hero(-24, .4);
        await X.say('Car 7 uploads a heavy task to Z1 (HTTP POST /tasks).', 'task', null);
        await X.up('Z1');
        X.anchor('Z1');
        await X.say('Z1 is the anchor: stores input, subscribes to car/7/#.', 'state');
        X.tof('Z1', 'boulder', 'exec on strongest fog class ≥ α × deadline');
        await X.say('TOF-Broker: boulder → straight to the cloud. Masks and D3QN are skipped.', 'warn');
        X.tag('fog:Z1', 'HTTP → cloud over WAN', COL.cloud, 2.5);
        await X.say('Task input travels over the WAN (50–200 ms).', 'task');
        await X.send('task', ['fog:Z1', 'cloud']);
        S.cloud.queue = 1; await X.say('Cloud: task waits in the cloud queue, then gets a slot.'); S.cloud.queue = 0;
        await X.exec('cloud', 2.2);
        await X.say('Cloud publishes car/7/result once → broker → serving zone Z1.', 'result', 'result');
        await X.publish('cloud', 'Z1');
        await X.deliver();
        await X.say('Car 7 confirms; car/7/ack reaches anchor Z1.', 'ack', 'ack');
        await X.ack('Z1');
        X.reward('on time → +1 − w·latency − energy');
        X.cleanup('Z1');
        await X.say('Anchor Z1 computes the reward and cleans up.', 'done', null);
        await X.wait(1);
      },
      3: async () => {
        X.hero(-17.5, 1.25);
        await X.say('Car 7 heads east out of Z1 and uploads a task.', 'task', null);
        await X.up('Z1');
        X.anchor('Z1');
        await X.say('Z1 is the anchor: input stored, subscribed to car/7/attach, gps, result, ack.', 'state');
        X.tof('Z1', 'pebble', 'exec on strongest fog class < α × deadline');
        X.tag('car:'+HERO, 'mobility: T_exit ≈ 3 s → next zone Z2', COL.car, 3);
        await X.say('TOF-Broker: pebble. Mobility predictor: car leaves Z1 soon, next zone Z2.');
        X.masks('Z1');
        X.choose('here', { here:.72, n1:.61, cloud:.40 }, 'Running here is fine: the per-car result topic will find the car.');
        await X.say('D3QN picks here · Z1.');
        X.role('Z1', 'exec');
        const job = X.exec('Z1', 4.2); job.catch(() => {});
        await X.until(() => servOf(HERO) === 'Z2');
        X.speed(.6);
        X.tag('fog:Z2', 'PUBLISH car/7/attach · SUBSCRIBE car/7/result', COL.attach, 3);
        await X.say('Car 7 attaches to Z2 → Z2 publishes car/7/attach and subscribes to car/7/result.', 'state', 'attach');
        await X.send('attach', ['fog:Z2', 'broker', 'fog:Z1']);
        await X.say('Anchor Z1 sees the attach. GPS now flows via Z2 (car/7/gps).', 'state', 'gps');
        await job;
        await X.say('Executor Z1 publishes car/7/result once → broker → Z2. Z1 keeps its copy.', 'result', 'result');
        await X.publish('Z1', 'Z1');
        await X.say('Z2 delivers the result to car 7 by radio.', 'result');
        await X.deliver();
        await X.say('Ack via Z2: Z2 publishes car/7/ack → anchor Z1.', 'ack', 'ack');
        await X.ack('Z1');
        X.reward('on time → +1 − w·latency − energy');
        X.cleanup('Z1');
        await X.say('Z1 computes the reward and cleans up.', 'done', null);
        await X.wait(1);
      },
      4: async () => {
        X.hero(-17, .8); S.sigNoise = true;
        const sg = () => { const p = carPos(hero); return ['Z1','Z2'].map(id => Math.max(0, sig(S.zones[id], p, hero))); };
        X.tag('car:'+HERO, '', COL.car, 60, { fn:() => { const [a, b] = sg(); return 'signal Z1 ' + a.toFixed(2) + ' · Z2 ' + b.toFixed(2) + ' → attached ' + (hero.serv || '—'); } });
        await X.say('Car 7 drives through Z1. Signal strength shown on the car (example values).', 'info', 'attach');
        await X.until(() => carPos(hero).x > -14);
        await X.say('It enters the overlap: both Z1 and Z2 are in range. It stays on Z1, the stronger signal.');
        await X.until(() => carPos(hero).x > -11.6);
        X.speed(.3);
        await X.say('Near the middle the signals cross back and forth — but the switch needs Z2 to be clearly stronger (hysteresis margin, example 0.3).', 'warn');
        await X.wait(3.2);
        await X.say('No ping-pong: car 7 stays attached to Z1.');
        X.speed(.8);
        await X.until(() => servOf(HERO) === 'Z2');
        X.tag('fog:Z2', 'PUBLISH car/7/attach', COL.attach, 2.5);
        await X.say('Z2 is now clearly stronger → one switch. Z2 publishes car/7/attach.', 'state', 'attach');
        await X.send('attach', ['fog:Z2', 'broker']);
        await X.say('One handover through the whole overlap area.', 'done');
        await X.wait(1.5);
      },
      5: async () => {
        X.hero(-17, 1.4); S.light.on = true; S.light.red = true;
        await X.say('Car 7 approaches a red light near the edge of Z1.', 'info', null);
        await X.until(() => carPos(hero).x >= LIGHT_X - 1.4);
        X.speed(0);
        await X.say('It stops. Mobility predictor: T_exit = ∞ while the car is stopped.', 'info', 'gps');
        X.tag('car:'+HERO, 'T_exit = ∞ (stopped)', COL.car, 6);
        await X.say('Car 7 uploads a task to Z1.', 'task', null);
        await X.up('Z1');
        X.anchor('Z1');
        X.tof('Z1', 'pebble', 'exec on strongest fog class < α × deadline');
        await X.say('Z1 anchors it. TOF-Broker: pebble.');
        X.masks('Z1');
        X.choose('here', { here:.79, n1:.55, cloud:.30 }, 'T_exit = ∞ → the car will not leave Z1 before the result is ready.');
        await X.say('D3QN picks here · Z1 — running locally is safe.');
        X.role('Z1', 'exec');
        await X.exec('Z1', 2);
        await X.say('Z1 publishes car/7/result → broker → Z1 → car 7.', 'result', 'result');
        await X.publish('Z1', 'Z1');
        await X.deliver();
        await X.ack('Z1');
        X.reward('on time → +1 − w·latency − energy');
        X.cleanup('Z1');
        await X.say('Ack, reward, cleanup.', 'done', null);
        S.light.red = false;
        await X.say('Light turns green; car 7 drives on.');
        X.speed(2.2);
        await X.wait(1.5);
      },
      6: async () => {
        X.hero(-3.5, 1.5);
        await X.say('Car 7 uploads a soft-deadline task (example 400 ms) just before the tunnel.', 'task', null);
        await X.up('Z2');
        X.anchor('Z2');
        X.tof('Z2', 'pebble', 'exec on strongest fog class < α × deadline');
        await X.say('Z2 anchors it. TOF-Broker: pebble.');
        X.masks('Z2');
        X.choose('here', { here:.77, n1:.41, n2:.55, n3:.38, cloud:.29 }, 'Z2 has free slots.');
        await X.say('D3QN picks here · Z2.');
        X.role('Z2', 'exec');
        await X.exec('Z2', 1.6);
        await X.until(() => servOf(HERO) === null);
        await X.say('Car 7 is in the tunnel: no radio → no GPS, no uploads.', 'warn', null);
        await X.say('Z2 publishes car/7/result — no zone serves car 7, so nobody can deliver it.', 'warn', 'result');
        await X.publish('Z2', 'Z2');
        S.zones.Z2.held = true;
        const h = X.tag('fog:Z2', 'anchor holds result · no ack yet', COL.anchor, 60);
        const o = X.tag('broker', '? soft hold limit (open)', 0x8b97a5, 60, { dashed:true });
        await X.say('Anchor Z2 holds its copy. How long a soft task may be held is still open.');
        await X.until(() => servOf(HERO) === 'Z3');
        X.speed(1);
        X.tag('fog:Z3', 'PUBLISH car/7/attach', COL.attach, 2.5);
        await X.say('Car 7 exits into Z3 → Z3 publishes car/7/attach.', 'state', 'attach');
        await X.send('attach', ['fog:Z3', 'broker', 'fog:Z2']);
        X.untag(h); X.untag(o); S.zones.Z2.held = false;
        X.tag('fog:Z2', 'PUBLISH car/7/result again', COL.result, 2.5);
        await X.say('Anchor Z2 publishes the result again → broker → Z3.', 'result', 'result');
        await X.send('result', ['fog:Z2', 'broker', ...toCar(HERO)]);
        await X.say('Z3 delivers. Ack via Z3 → anchor Z2.', 'ack', 'ack');
        await X.ack('Z2');
        X.reward('soft, on time → +1 − w·latency − energy');
        X.cleanup('Z2');
        X.note('If a hard deadline had passed first: dropped, reward −2 − energy.');
        await X.say('Cleanup. (A hard task past its deadline would be dropped with a penalty.)', 'done', null);
        await X.wait(1);
      },
      7: async () => {
        X.hero(-7, .35);
        await X.say('Car 7 uploads a hard-deadline task to Z2.', 'task', null);
        await X.up('Z2');
        X.anchor('Z2');
        X.tof('Z2', 'pebble', 'exec on strongest fog class < α × deadline');
        await X.say('Z2 anchors it. TOF-Broker: pebble.');
        X.masks('Z2', { n2:'stale' });
        await X.say('Masks: N2 · Z3 is stale (state older than 3 × period — proposed); N4 empty.');
        X.choose('n3', { here:.58, n1:.49, n3:.74, cloud:.31 }, 'Z4 has the best Q-value.');
        await X.say('D3QN picks N3 · Z4.');
        X.role('Z4', 'exec');
        await X.send('task', ['fog:Z2', 'fog:Z4']);
        await X.say('Z4 starts running the task…', 'task');
        await X.fail('Z4', 1.6);
        X.unrole('Z4', 'exec');
        X.tag('fog:Z4', 'execution error', COL.fail, 3);
        await X.say('Z4 fails while executing.', 'warn');
        X.tag('fog:Z2', '? failure timeout (open)', 0x8b97a5, 4, { dashed:true });
        await X.say('Anchor Z2 detects the failure (timeout value is open). It still has the input.');
        X.masks('Z2', { n2:'stale' });
        X.choose('here', { here:.70, n1:.52, n3:.18, cloud:.27 }, 'Hard task: one retry of the whole task, time allows.');
        await X.say('Hard task → retry once. D3QN now picks here · Z2.', 'task');
        X.role('Z2', 'exec');
        await X.exec('Z2', 1.6);
        await X.say('Z2 publishes car/7/result → broker → Z2 (serving).', 'result', 'result');
        await X.publish('Z2', 'Z2');
        await X.deliver();
        await X.ack('Z2');
        X.reward('on time → +1 − w·latency − energy');
        X.cleanup('Z2');
        X.note('Soft tasks may retry up to twice. No time left → drop, reward −2 − energy.');
        await X.say('Delivered on the retry. (Soft: up to 2 retries; no time left → drop + penalty.)', 'done', null);
        await X.wait(1);
      },
      8: async () => {
        X.hero(-25, .3); X.busy('Z1', 4, 30, 3); X.busy('Z2', 4, 3.6, 2);
        await X.say('Car 7 uploads two tasks to Z1: A (hard) and B (soft). Z1 and Z2 are busy.', 'task', null);
        await X.fan('task', [['car:'+HERO, 'ap:Z1', 'fog:Z1'], ['car:'+HERO, 'ap:Z1', 'fog:Z1']]);
        X.anchor('Z1'); S.zones.Z1.anchored++;
        X.tof('Z1', 'pebble', 'both tasks: exec on strongest fog class < α × deadline');
        X.masks('Z1', { here:'queue full', cloud:'misses deadline' });
        await X.say('Masks: here is full; cloud cannot meet the deadline over the WAN.');
        X.choose('n1', { n1:.62 }, 'Only N1 · Z2 is left for both tasks.');
        await X.say('D3QN sends A and B to N1 · Z2.');
        X.role('Z2', 'exec');
        await X.fan('task', [['fog:Z1', 'fog:Z2'], ['fog:Z1', 'fog:Z2']]);
        S.zones.Z2.qx = 2;
        await X.say('Both wait in Z2’s queue: hard first, then earliest deadline, aging for soft.');
        await X.wait(1.2);
        S.zones.Z2.qx = 1; fx('fog:Z2', COL.hot);
        X.tag('fog:Z2', 'A skipped: would surely miss hard deadline', COL.hot, 3.5);
        await X.say('By the time A could start it would surely miss its hard deadline → skipped before starting.', 'warn');
        X.tag('fog:Z1', 'A dropped · r = −2 − energy', COL.hot, 3);
        await X.say('Anchor Z1: A dropped, reward −2 − spent energy.');
        S.zones.Z2.qx = 0;
        await X.exec('Z2', 2);
        await X.say('Soft task B keeps going, runs, and is published late.', 'result', 'result');
        await X.publish('Z2', 'Z1');
        await X.deliver();
        await X.ack('Z1');
        X.reward('B late → +1 − w·latency − energy − lateness penalty');
        X.cleanup('Z1'); S.zones.Z1.anchored = 0;
        await X.say('B delivered late → reward with a lateness penalty.', 'done', null);
        await X.wait(1);
      },
      9: async () => {
        X.hero(-6, .3);
        await X.say('Car 7 uploads a task to Z2.', 'task', null);
        await X.up('Z2');
        X.anchor('Z2');
        X.tof('Z2', 'pebble', 'exec on strongest fog class < α × deadline');
        X.masks('Z2');
        X.choose('n2', { here:.51, n1:.44, n2:.78, n3:.60, cloud:.31 }, 'Z3 is idle.');
        await X.say('Z2 anchors it; D3QN places it on N2 · Z3.');
        X.role('Z3', 'exec');
        await X.send('task', ['fog:Z2', 'fog:Z3']);
        const z3 = S.zones.Z3; z3.slots[0] = 99;
        await X.say('Z3 is running the task…', 'task');
        await X.wait(1);
        z3.alive = false; z3.slots = [0,0,0,0]; z3.roles.clear(); fx('fog:Z3', COL.hot);
        X.tag('fog:Z3', 'offline', COL.hot, 30);
        await X.say('Z3 crashes. The broker publishes its last will: zone/3/status = offline (retained).', 'warn', 'status');
        await X.fan('state', [['broker', 'fog:Z2'], ['broker', 'fog:Z4']], { from:'fog:Z3' });
        await X.say('Neighbours Z2 and Z4 mask Z3. Cars in Z3 have no overlapping zone → they fall into a gap.');
        X.masks('Z2');
        X.choose('n3', { here:.51, n1:.44, n3:.70, cloud:.31 }, 'Retry: Z3 masked (offline).');
        await X.say('Anchor Z2 retries the task it had placed on Z3 → N3 · Z4.', 'task');
        X.role('Z4', 'exec');
        await X.send('task', ['fog:Z2', 'fog:Z4']);
        await X.exec('Z4', 1.6);
        await X.publish('Z4', 'Z2');
        await X.deliver();
        await X.ack('Z2');
        X.reward('on time → +1 − w·latency − energy');
        X.cleanup('Z2');
        X.unrole('Z4', 'exec');
        await X.say('Result delivered after the retry; Z2 cleans up.', 'done', 'result');
        S.zones.Z5.joined = true;
        await X.wait(1.2);
        await X.say('Z5 joins: publishes zone/5/status = online and zone/5/state.', 'state', 'status');
        await X.send('state', ['fog:Z5', 'broker']);
        await X.fan('state', [['broker', 'fog:Z1'], ['broker', 'fog:Z2'], ['broker', 'fog:Z4']], { from:'fog:Z5' });
        X.tag('fog:Z5', 'online', COL.exec, 3);
        X.tag('fog:Z5', '? neighbour position assignment (open)', 0x8b97a5, 6, { dashed:true });
        X.tof('Z2', 'pebble', 'next task');
        X.masks('Z2');
        X.note('Z5 now fills N4 for Z2 (and N2 for Z1). Z3 stays masked until it reconnects.');
        await X.say('Z5 becomes a new unmasked option for its neighbours. (How positions are assigned is open.)', 'done', 'state');
        await X.wait(1.5);
      },
      10: async () => {
        X.hero(-23, .35);
        await X.say('Rush hour: many heavy tasks are sent to the cloud over the WAN.', 'warn', null);
        for (let i = 0; i < 26; i++) {
          const cs = S.cars.filter(c => c.serv && c.id !== HERO), c = cs[Math.floor(Math.random()*cs.length)];
          if (c) {
            const z = c.serv; S.inflight[c.id]++;
            if (i % 6 === 0) X.tag('fog:'+z, 'boulder → cloud', COL.cloud, 1.2);
            (async () => {
              if (await send('task', ['car:'+c.id, 'ap:'+z, 'fog:'+z, 'cloud'])) { await X.exec('cloud', 4.5); await send('result', ['cloud', 'broker', ...toCar(c.id)]); }
              S.inflight[c.id] = Math.max(0, S.inflight[c.id]-1);
            })().catch(() => {});
          }
          await X.wait(.22);
          if (i === 12) await X.say('The WAN link saturates (red) and cloud wait grows.', 'warn');
        }
        await X.wait(.6);
        await X.say('Car 7 uploads a normal task to Z1.', 'task');
        await X.up('Z1');
        X.anchor('Z1');
        X.tof('Z1', 'pebble', 'exec on strongest fog class < α × deadline');
        X.masks('Z1');
        X.choose('here', { here:.71, n1:.58, cloud:.09 }, 'WAN ' + Math.round(S.wanMs) + ' ms + cloud queue → cloud Q-value drops.');
        await X.say('D3QN’s cloud Q-value drops → it keeps the task local.', 'info');
        X.role('Z1', 'exec');
        await X.exec('Z1', 1.6);
        await X.publish('Z1', 'Z1');
        await X.deliver();
        await X.ack('Z1');
        X.reward('on time → +1 − w·latency − energy');
        X.cleanup('Z1');
        await X.say('Delivered from the edge while the WAN is congested.', 'done', null);
        await X.until(() => S.wanLoad <= 0);
        await X.say('Cloud queue drains; the WAN recovers.', 'done');
        await X.wait(1);
      },
    };
    try { await FL[n](); } catch (e) { if (e !== CANCEL) console.error(e); return; }
    call('onLog', { text:'Flow complete.', kind:'end' });
    await simWait(3);
    if (my.dead) return;
    for (const z of zl()) z.roles.clear();
    S.flowActive = false; S.sigNoise = false; S.light.on = false; hero.v = hero.v0; call('onFlow', null); call('onTopic', null);
  }

  // ---------- 3D ----------
  function make3D() {
    const wrap = document.createElement('div'); wrap.style.cssText = 'position:absolute;inset:0;';
    root.appendChild(wrap);
    const renderer = new THREE.WebGLRenderer({ antialias:true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1)); renderer.setClearColor(COL.bg, 1);
    renderer.domElement.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:grab;';
    wrap.appendChild(renderer.domElement);
    const layer = document.createElement('div'); layer.style.cssText = 'position:absolute;inset:0;pointer-events:none;overflow:hidden;';
    wrap.appendChild(layer);
    const scene = new THREE.Scene();
    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, .1, 600);
    const ctl = { az:Math.PI*.22, el:.62, zoom:1 }, target = new THREE.Vector3(0, 9, -7);
    function place() {
      const w = wrap.clientWidth || 1, h = wrap.clientHeight || 1, a = w/h, d = 160, I = S.insets;
      const aw = Math.max(200, w - I.left - I.right), ah = Math.max(200, h - I.top - I.bottom);
      cam.position.set(target.x + d*Math.cos(ctl.el)*Math.sin(ctl.az), target.y + d*Math.sin(ctl.el), target.z + d*Math.cos(ctl.el)*Math.cos(ctl.az));
      cam.lookAt(target);
      const hh = 27*Math.max(1, 1.6/(aw/ah))/ctl.zoom*(h/ah);
      const upp = 2*hh/h, dx = (I.left - I.right)/2*upp, dy = (I.top - I.bottom)/2*upp;
      cam.left = -hh*a - dx; cam.right = hh*a - dx; cam.top = hh + dy; cam.bottom = -hh + dy; cam.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    }
    scene.add(new THREE.HemisphereLight(0xc9d8e8, 0x0b0e13, 1.7));
    const dl = new THREE.DirectionalLight(0xffffff, 1.5); dl.position.set(25, 45, 30); scene.add(dl);
    const MS = (c, o = {}) => new THREE.MeshStandardMaterial({ color:c, roughness:.85, metalness:.05, ...o });
    const MB = (c, o = {}) => new THREE.MeshBasicMaterial({ color:c, ...o });
    const unit = new THREE.BoxGeometry(1, 1, 1);
    const box = (w, h, d, mat, x, y, z, parent = scene) => { const m = new THREE.Mesh(unit, mat); m.scale.set(w, h, d); m.position.set(x, y, z); parent.add(m); return m; };
    const g0 = new THREE.Mesh(new THREE.PlaneGeometry(110, 70), MS(COL.ground)); g0.rotation.x = -Math.PI/2; g0.position.set(0, 0, -8); scene.add(g0);
    const roadM = MS(COL.road), dashM = MB(COL.dash);
    box(60,.06,3.4,roadM,0,.03,0); box(60,.06,3.4,roadM,0,.03,-16); box(3.4,.06,16,roadM,28,.03,-8); box(3.4,.06,16,roadM,-28,.03,-8);
    for (let x = -25; x <= 25; x += 3) { box(1.3,.07,.12,dashM,x,.04,0); box(1.3,.07,.12,dashM,x,.04,-16); }
    let seed = 11; const rnd = () => (seed = (seed*16807) % 2147483647)/2147483647;
    const bM = [MS(0x18202a), MS(0x1c2530), MS(0x141b23)], eqs = zl().map(equip);
    for (let x = -40; x <= 40; x += 3.2) for (let z = -30; z <= 12; z += 3.2) {
      const onRoad = Math.abs(z) < 3.4 || Math.abs(z+16) < 3.4 || ((Math.abs(x-28) < 3.4 || Math.abs(x+28) < 3.4) && z < 2 && z > -18);
      const near = eqs.some(e => Math.hypot(e.x-x, e.z-z) < 3.6) || Math.hypot(x-BROKER.x, z-BROKER.z) < 5;
      if (onRoad || near || rnd() < .45) continue;
      const h = .4 + rnd()*rnd()*3.4, w = 1.8 + rnd()*.9;
      box(w, h, w, bM[Math.floor(rnd()*3)], x + (rnd()-.5)*.6, h/2, z + (rnd()-.5)*.6);
    }
    box(6,1.9,4.6,MS(0x29313b),7.5,.95,0); box(.25,1.5,3.6,MB(0x050709),4.45,.75,0); box(.25,1.5,3.6,MB(0x050709),10.55,.75,0);
    // traffic light
    const lg = new THREE.Group(); lg.position.set(LIGHT_X, 0, 2.4); scene.add(lg);
    box(.12, 2.6, .12, MS(0x6d7682), 0, 1.3, 0, lg); box(.5, .9, .4, MS(0x1a1f26), 0, 2.8, 0, lg);
    const lampM = MB(0xff4545); const lamp = new THREE.Mesh(new THREE.SphereGeometry(.2, 12, 10), lampM); lamp.position.set(0, 2.8, .22); lg.add(lamp);

    const labels = [];
    const mkLabel = (html, at, o = {}) => {
      const d = document.createElement('div');
      d.style.cssText = 'position:absolute;left:0;top:0;display:none;white-space:nowrap;font:500 13px ' + MONO + ';color:#c9d3de;background:rgba(11,14,19,.78);padding:3px 8px;border-radius:6px;' + (o.css || '');
      d.innerHTML = html; layer.appendChild(d);
      const L = { el:d, at, vis:o.vis, upd:o.upd, op:o.op, dx:o.dx || 0 }; labels.push(L); return L;
    };
    const involved = z => z.roles.size > 0 || hero.serv === z.id;
    const ZV = {}, pickables = [], carPick = [];
    for (const z of zl()) {
      const g = new THREE.Group(); g.position.set(z.x, 0, z.z); scene.add(g);
      const domeMat = MB(COL.dome, { transparent:true, opacity:.07, depthWrite:false, side:THREE.DoubleSide });
      const dome = new THREE.Mesh(new THREE.SphereGeometry(z.r, 48, 16, 0, Math.PI*2, 0, Math.PI/2), domeMat); dome.scale.y = .42; g.add(dome);
      const ringMat = MB(COL.dome, { transparent:true, opacity:.6, depthWrite:false });
      const ring = new THREE.Mesh(new THREE.RingGeometry(z.r-.15, z.r, 96), ringMat); ring.rotation.x = -Math.PI/2; ring.position.y = .09; g.add(ring);
      const discMat = MB(COL.dome, { transparent:true, opacity:.035, depthWrite:false });
      const disc = new THREE.Mesh(new THREE.CircleGeometry(z.r, 64), discMat); disc.rotation.x = -Math.PI/2; disc.position.y = .08; g.add(disc);
      const e = equip(z), eg = new THREE.Group(); eg.position.set(e.x, 0, e.z); scene.add(eg);
      box(3.4,.14,1.9,MS(0x1d242d),0,.07,0,eg);
      const poleM = MS(0x8592a0);
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(.1,.16,5.6,8), poleM); pole.position.set(-.9,2.8,0); eg.add(pole);
      box(1,.08,.08,poleM,-.9,5.2,0,eg); box(.8,.08,.08,poleM,-.9,4.7,0,eg);
      const headM = MB(0xdfe8f2), head = new THREE.Mesh(new THREE.SphereGeometry(.32,16,12), headM); head.position.set(-.9,5.9,0); eg.add(head);
      const cabM = MS(0x26303c), cab = box(1.3,1.8,1.0,cabM,.9,.9,0,eg);
      const capM = MB(0x8a96a4); box(1.42,.1,1.12,capM,.9,1.85,0,eg);
      const leds = [0,1,2,3].map(i => box(1.0,.17,.04,MB(COL.led),.9,.4+i*.36,.52,eg));
      const ledsX = [0,1,2,3].map(i => box(.04,.17,.8,MB(COL.led),1.57,.4+i*.36,0,eg));
      [cab, pole, head].forEach(m => { m.userData.zone = z.id; pickables.push(m); });
      ZV[z.id] = { g, eg, domeMat, ringMat, discMat, leds, ledsX, capM, headM, cabM };
      mkLabel('', () => new THREE.Vector3(e.x-.9, 7.2, e.z), { css:'background:transparent;padding:0;', upd:el => {
        const html = '<span style="font:600 16px ' + SANS + ';color:#eef3f8">' + z.id + '</span> <span style="color:#8b97a5;font-size:12px">' + (!z.joined ? 'can join' : !z.alive ? 'offline' : 'fog zone') + '</span>';
        if (el.__h !== html) { el.innerHTML = html; el.__h = html; }
        el.style.opacity = !z.joined ? .4 : !z.alive ? .55 : (S.flowActive && !involved(z) ? .4 : 1);
      } });
    }
    const bg = new THREE.Group(); bg.position.set(BROKER.x, 0, BROKER.z); scene.add(bg);
    const bcM = MS(COL.broker, { emissive:0x15191e });
    const c0 = new THREE.Mesh(new THREE.CylinderGeometry(2,2.2,.3,24), MS(0x1f262f)); c0.position.y = .15; bg.add(c0);
    for (let k = 0; k < 3; k++) { const a = k*Math.PI*2/3, c = new THREE.Mesh(new THREE.CylinderGeometry(.48,.48,6,16), bcM); c.position.set(Math.cos(a)*.8, 3.2, Math.sin(a)*.8); bg.add(c); }
    const ct = new THREE.Mesh(new THREE.CylinderGeometry(1.7,1.7,.16,24), MS(0x6d7682)); ct.position.y = 6.3; bg.add(ct);
    const bb = new THREE.Mesh(new THREE.SphereGeometry(.26,16,12), MB(0xdfe6ee)); bb.position.y = 6.8; bg.add(bb);
    mkLabel('<span style="font:600 15px ' + SANS + ';color:#eef3f8">MQTT broker</span> <span style="color:#8b97a5;font-size:12px">EMQX cluster · metro edge </span>', () => new THREE.Vector3(BROKER.x, 8.3, BROKER.z));

    function pos3(id) {
      const [k, v] = id.split(':');
      if (k === 'car') { const p = carPos(CAR[v]); return new THREE.Vector3(p.x, .8, p.z); }
      if (k === 'ap') { const e = equip(S.zones[v]); return new THREE.Vector3(e.x-.9, 5.9, e.z); }
      if (k === 'fog') { const e = equip(S.zones[v]); return new THREE.Vector3(e.x+.9, 2.0, e.z); }
      if (id === 'broker') return new THREE.Vector3(BROKER.x, 6.8, BROKER.z);
      return new THREE.Vector3(BROKER.x, CLOUD_Y-.4, BROKER.z);
    }
    function seg3(a, b) {
      const A = pos3(a), B = pos3(b), ka = kind(a), kb = kind(b);
      if (ka === 'car' || kb === 'car' || ka === 'ap' || kb === 'ap' || isWan(a, b)) return [A, B];
      const M = A.clone().add(B).multiplyScalar(.5); M.y = Math.max(A.y, B.y) + 1.5 + A.distanceTo(B)*.1;
      if (a === b) M.y += 2;
      const pts = []; for (let i = 0; i <= 16; i++) { const t = i/16, u = 1-t; pts.push(new THREE.Vector3(u*u*A.x+2*u*t*M.x+t*t*B.x, u*u*A.y+2*u*t*M.y+t*t*B.y, u*u*A.z+2*u*t*M.z+t*t*B.z)); }
      return pts;
    }
    const links = [];
    for (const z of zl()) {
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(seg3('fog:'+z.id, 'broker')), new THREE.LineBasicMaterial({ color:COL.link, transparent:true, opacity:.6 }));
      scene.add(l); links.push({ l, zs:[z.id], base:.6 });
    }
    for (const [a, b] of PEERS) {
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(seg3('fog:'+a, 'fog:'+b)), new THREE.LineDashedMaterial({ color:COL.link, dashSize:.5, gapSize:.35, transparent:true, opacity:.4 }));
      l.computeLineDistances(); scene.add(l); links.push({ l, zs:[a, b], base:.4 });
    }
    { const pts = seg3('fog:Z2', 'broker'); mkLabel('backhaul · 1–5 ms', () => pts[8].clone().add(new THREE.Vector3(0, .6, 0)), { css:'color:#8b97a5;', vis:() => !S.flowActive }); }
    // WAN: broker ↔ cloud trunk + zone ↔ cloud lines
    const wanMats = [];
    const wA = pos3('broker'), wB = pos3('cloud');
    const trunkM = MB(COL.wan); wanMats.push(trunkM);
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(.06,.06,wB.y-wA.y,8), trunkM); trunk.position.set(BROKER.x, (wA.y+wB.y)/2, BROKER.z); scene.add(trunk);
    const zWan = {};
    for (const z of zl()) {
      const m = new THREE.LineBasicMaterial({ color:COL.wan, transparent:true, opacity:.22 });
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([pos3('fog:'+z.id), wB]), m); scene.add(l); zWan[z.id] = l; wanMats.push(m);
    }
    const wgM = MB(COL.hot, { transparent:true, opacity:0, depthWrite:false });
    const wg = new THREE.Mesh(new THREE.CylinderGeometry(.4,.4,wB.y-wA.y,12,1,true), wgM); wg.position.copy(trunk.position); scene.add(wg);
    mkLabel('', () => new THREE.Vector3(BROKER.x, (wA.y+wB.y)/2 + 3, BROKER.z), { dx:14, upd:el => {
      const hot = Math.min(1, Math.max(0, S.wanLoad)/9);
      el.innerHTML = 'WAN · <b style="color:' + (hot > .5 ? '#ff6b6b' : '#9fd4ff') + '">' + Math.round(S.wanMs) + ' ms</b> <span style="color:#6f7c8b">(50–200)</span>';
    } });
    const cg = new THREE.Group(); cg.position.set(BROKER.x, CLOUD_Y, BROKER.z); scene.add(cg);
    box(15,.7,8,MS(0x18222e),0,0,0,cg);
    cg.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(15,.7,8)), new THREE.LineBasicMaterial({ color:COL.cloud, transparent:true, opacity:.6 })));
    const cloudLeds = [];
    for (let j = 0; j < 3; j++) for (let i = 0; i < 8; i++) cloudLeds.push(box(1.1,.3,1.1,MB(COL.led),-5.6+i*1.6,.5,-1.6+j*1.6,cg));
    const puffM = MB(COL.cloud, { transparent:true, opacity:.05, depthWrite:false });
    [[-5,1.5,0,4.5],[1,2.5,-1,5.5],[6,1.4,.5,4],[-1,.8,2,4.5]].forEach(([x,y,z,r]) => { const m = new THREE.Mesh(new THREE.SphereGeometry(r,24,16), puffM); m.position.set(x,y,z); cg.add(m); });
    mkLabel('', () => new THREE.Vector3(BROKER.x, CLOUD_Y+4.4, BROKER.z), { upd:el => {
      const html = '<span style="font:600 16px ' + SANS + ';color:#cfe9ff">Cloud</span> <span style="color:#8b97a5;font-size:12px">24 slots · queue ' + S.cloud.queue + '</span>';
      if (el.__h !== html) { el.innerHTML = html; el.__h = html; }
    } });
    mkLabel('tunnel · no coverage', () => new THREE.Vector3(7.5, 2.4, 0), { css:'color:#8b97a5;', vis:() => !S.flowActive || S.flow === 6 });
    mkLabel('', () => new THREE.Vector3(LIGHT_X, 3.6, 2.4), { vis:() => S.light.on, upd:el => { el.textContent = S.light.red ? 'red light' : 'green'; el.style.color = S.light.red ? '#ff6b6b' : '#3ddc84'; } });
    const carM = MS(COL.car, { emissive:0x3a2a00 }), cabCM = MS(0x1d222a), CV = {};
    for (const c of S.cars) {
      const g = new THREE.Group(); scene.add(g);
      const body = box(1.5,.5,.8,carM,0,.4,0,g), top = box(.8,.36,.7,cabCM,-.1,.8,0,g);
      [body, top].forEach(m => { m.userData.car = c.id; carPick.push(m); });
      if (c.id === HERO) {
        const h = new THREE.Mesh(new THREE.RingGeometry(1.15,1.4,40), MB(COL.car, { transparent:true, opacity:.7, depthWrite:false }));
        h.rotation.x = -Math.PI/2; h.position.y = .08; g.add(h); g.userData.halo = h;
      }
      CV[c.id] = g;
    }
    mkLabel('<span style="color:#ffd23f;font-weight:600">car 7</span>', () => pos3('car:'+HERO).add(new THREE.Vector3(0, 1.3, 0)), { css:'border:1px solid rgba(255,210,63,.4);' });
    const radioArr = new Float32Array(S.cars.length*6), radioGeo = new THREE.BufferGeometry(); radioGeo.setAttribute('position', new THREE.BufferAttribute(radioArr, 3));
    scene.add(new THREE.LineSegments(radioGeo, new THREE.LineBasicMaterial({ color:0xd8e2ee, transparent:true, opacity:.2 })));
    const heroArr = new Float32Array(6), heroGeo = new THREE.BufferGeometry(); heroGeo.setAttribute('position', new THREE.BufferAttribute(heroArr, 3));
    const heroLine = new THREE.Line(heroGeo, new THREE.LineDashedMaterial({ color:COL.car, transparent:true, opacity:.85, dashSize:.4, gapSize:.25 })); scene.add(heroLine);
    const pGeo = new THREE.SphereGeometry(.34, 14, 10), pMats = {}, gMats = {};
    for (const k in PCOL) { pMats[k] = MB(PCOL[k]); gMats[k] = MB(PCOL[k], { transparent:true, opacity:.25, depthWrite:false }); }
    const fMat = MB(COL.state, { transparent:true, opacity:.45 });
    const pm = new Map(), fxGeo = new THREE.RingGeometry(.4,.6,28), fxm = new Map();
    const C1 = new THREE.Color(), C2 = new THREE.Color(), v = new THREE.Vector3();
    function zoneLook(z) {
      const serv = S.flowActive && hero.serv === z.id, anchor = z.roles.has('anchor');
      if (!z.alive) return { fill:COL.dead, ring:COL.dead, a:.05 };
      const fill = anchor ? COL.anchor : serv ? COL.serving : COL.dome;
      return { fill, ring:anchor && serv ? COL.serving : fill, a:(anchor || serv) ? .15 : .06 };
    }
    function screen(id) { const w = wrap.clientWidth, h = wrap.clientHeight; v.copy(pos3(id)).project(cam); return { x:(v.x+1)/2*w, y:(1-v.y)/2*h }; }
    function render(now) {
      for (const c of S.cars) {
        const p = carPos(c), g = CV[c.id]; g.position.set(p.x, 0, p.z);
        g.rotation.y = Math.atan2(-p.dz*c.dir, p.dx*c.dir);
        if (g.userData.halo) { const s = 1 + .12*Math.sin(now*.005); g.userData.halo.scale.set(s, s, s); }
      }
      let k = 0;
      for (const c of S.cars) {
        let a, b;
        if (S.radio && c.serv && c !== hero) { a = pos3('car:'+c.id); b = pos3('ap:'+c.serv); } else a = b = new THREE.Vector3(0, -5, 0);
        radioArr.set([a.x,a.y,a.z,b.x,b.y,b.z], k); k += 6;
      }
      radioGeo.attributes.position.needsUpdate = true;
      heroLine.visible = !!hero.serv;
      if (hero.serv) { const a = pos3('car:'+HERO), b = pos3('ap:'+hero.serv); heroArr.set([a.x,a.y,a.z,b.x,b.y,b.z]); heroGeo.attributes.position.needsUpdate = true; heroLine.computeLineDistances(); }
      lg.visible = S.light.on; lampM.color.setHex(S.light.red ? 0xff4545 : 0x3ddc84);
      for (const z of zl()) {
        const V = ZV[z.id], f = z.fade, L = zoneLook(z);
        V.g.visible = V.eg.visible = f > .01; V.eg.scale.set(1, Math.max(.01, f), 1);
        V.domeMat.color.setHex(L.fill); V.domeMat.opacity = L.a*f;
        V.ringMat.color.setHex(L.ring); V.ringMat.opacity = (z.alive ? .7 : .3)*f;
        V.discMat.color.setHex(L.fill); V.discMat.opacity = L.a*.5*f;
        const col = i => !z.alive ? 0x111418 : z.failT[i] > 0 ? COL.fail : z.slots[i] > 0 ? COL.exec : COL.led;
        V.leds.forEach((m, i) => m.material.color.setHex(col(i))); V.ledsX.forEach((m, i) => m.material.color.setHex(col(i)));
        V.capM.color.setHex(!z.alive ? COL.dead : z.roles.has('exec') ? COL.exec : z.roles.has('anchor') ? COL.anchor : 0x8a96a4);
        V.headM.color.setHex(z.alive ? (S.flowActive && hero.serv === z.id ? COL.serving : 0xdfe8f2) : COL.dead);
        V.cabM.emissive.setHex(S.sel === z.id ? 0x34506a : 0);
        zWan[z.id].visible = f > .05; zWan[z.id].material.opacity = (z.alive ? .22 : .05)*f;
      }
      for (const L of links) { const zs = L.zs.map(id => S.zones[id]), f = Math.min(...zs.map(z => z.fade)); L.l.visible = f > .02; L.l.material.opacity = (zs.every(z => z.alive) ? L.base : .08)*f; }
      const hot = Math.min(1, Math.max(0, S.wanLoad)/9);
      C1.setHex(COL.wan).lerp(C2.setHex(COL.hot), hot);
      wanMats.forEach(m => m.color.copy(C1)); wgM.opacity = hot*.3; trunk.scale.set(1 + hot*1.5, 1, 1 + hot*1.5);
      for (const z of zl()) zWan[z.id].material.opacity = Math.min(.9, zWan[z.id].material.opacity + hot*.5);
      cloudLeds.forEach((m, i) => m.material.color.setHex(S.cloudSlots[i] > 0 ? COL.exec : COL.led));
      const live = new Set(S.packets);
      for (const [p, m] of pm) if (!live.has(p)) { scene.remove(m); pm.delete(p); }
      for (const p of S.packets) {
        let m = pm.get(p);
        if (!m) {
          m = new THREE.Mesh(pGeo, p.faint ? fMat : pMats[p.type]);
          if (!p.faint) { const gl = new THREE.Mesh(pGeo, gMats[p.type]); gl.scale.setScalar(2.3); m.add(gl); }
          scene.add(m); pm.set(p, m);
        }
        const [A, B, t] = along(seg3(p.ids[p.seg], p.ids[p.seg+1]), p.t, (a, b) => a.distanceTo(b));
        m.position.copy(A).lerp(B, t); m.scale.setScalar(p.size*(p.faint ? .7 : 1));
      }
      const fl = new Set(S.fx);
      for (const [f, m] of fxm) if (!fl.has(f)) { scene.remove(m); fxm.delete(f); }
      for (const f of S.fx) {
        let m = fxm.get(f);
        if (!m) { m = new THREE.Mesh(fxGeo, MB(f.color, { transparent:true, depthWrite:false, side:THREE.DoubleSide })); scene.add(m); fxm.set(f, m); m.position.copy(pos3(f.id)); }
        m.lookAt(cam.position); m.scale.setScalar(1 + f.t*5); m.material.opacity = 1 - f.t/.9;
      }
      renderer.render(scene, cam);
      const w = wrap.clientWidth, h = wrap.clientHeight;
      for (const L of labels) {
        if (L.upd) L.upd(L.el);
        if (L.vis && !L.vis()) { L.el.style.display = 'none'; continue; }
        v.copy(L.at()).project(cam);
        L.el.style.display = '';
        L.el.style.transform = 'translate(' + ((v.x+1)/2*w + L.dx) + 'px,' + ((1-v.y)/2*h) + 'px) translate(' + (L.dx ? '0' : '-50%') + ',-100%)';
      }
    }
    const el = renderer.domElement, ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
    function hit(e, list) {
      const r = el.getBoundingClientRect();
      ndc.set(((e.clientX-r.left)/r.width)*2-1, -((e.clientY-r.top)/r.height)*2+1);
      ray.setFromCamera(ndc, cam);
      const h = ray.intersectObjects(list, false); return h.length ? h[0].object : null;
    }
    const pickZone = e => { const o = hit(e, pickables.filter(m => S.zones[m.userData.zone].fade > .5)); return o ? o.userData.zone : null; };
    let drag = null;
    el.addEventListener('pointerdown', e => { drag = { x:e.clientX, y:e.clientY, moved:false }; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', e => {
      if (drag) {
        const dx = e.clientX-drag.x, dy = e.clientY-drag.y;
        if (!drag.moved && Math.abs(dx)+Math.abs(dy) > 4) drag.moved = true;
        if (drag.moved) { ctl.az -= dx*.006; ctl.el = Math.max(.2, Math.min(1.35, ctl.el + dy*.004)); drag.x = e.clientX; drag.y = e.clientY; place(); el.style.cursor = 'grabbing'; S.hover = null; }
        return;
      }
      const car = hit(e, carPick), r = root.getBoundingClientRect();
      S.hover = car ? { id:car.userData.car, x:e.clientX-r.left, y:e.clientY-r.top } : null;
      el.style.cursor = car || pickZone(e) ? 'pointer' : 'grab';
    });
    el.addEventListener('pointerleave', () => S.hover = null);
    el.addEventListener('pointerup', e => { if (drag && !drag.moved) { const z = pickZone(e); S.sel = z; call('onSelect', z); } drag = null; el.style.cursor = 'grab'; });
    el.addEventListener('wheel', e => { e.preventDefault(); ctl.zoom = Math.max(.5, Math.min(4, ctl.zoom*Math.exp(-e.deltaY*.0012))); place(); }, { passive:false });
    place();
    return { render, screen, place, show:on => wrap.style.display = on ? '' : 'none', dispose:() => renderer.dispose() };
  }

  // ---------- poster ----------
  function makePoster() {
    const NS = 'http://www.w3.org/2000/svg';
    const wrap = document.createElement('div'); wrap.style.cssText = 'position:absolute;inset:0;display:none;background:#0b0e13;';
    root.appendChild(wrap);
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 1300 1000'); svg.setAttribute('preserveAspectRatio', 'xMidYMid meet'); svg.setAttribute('font-family', MONO);
    svg.style.cssText = 'position:absolute;';
    wrap.appendChild(svg);
    const E = (t, a, p = svg) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
    const T = (x, y, txt, a = {}, p = svg) => { const e = E('text', { x, y, ...a }, p); e.textContent = txt; return e; };
    const ZX = { Z1:190, Z2:420, Z3:650, Z4:880, Z5:1110 }, RY = { A:770, B:900 };
    const mapX = wx => 90 + (wx+30)/60*1120;
    const BUS = 304, TOP = 380, PEER = 362, WANL = 342, WX = 1258;
    E('rect', { x:0, y:0, width:1300, height:1000, fill:'#0b0e13' });
    [[20,250,'CLOUD','one data centre · bottleneck is the WAN, not compute'],[326,666,'FOG','zones: AP + coverage + fog node + TOF-Broker + mobility predictor + D3QN'],[682,982,'IoT','cars talk only by radio to their serving zone · never to the broker']].forEach(([y0, y1, t, s]) => {
      E('rect', { x:20, y:y0, width:1260, height:y1-y0, rx:16, fill:'#0e131a', stroke:'#1a212b' });
      T(40, y0+32, t, { fill:'#eef3f8', 'font-size':18, 'font-weight':600, 'letter-spacing':3, 'font-family':SANS });
      T(40 + t.length*16 + 18, y0+31, s, { fill:'#6f7c8b', 'font-size':14 });
    });
    E('rect', { x:410, y:46, width:480, height:174, rx:14, fill:'#141c26', stroke:H6(COL.cloud), 'stroke-opacity':.7, 'stroke-width':2 });
    T(432, 82, 'Cloud', { fill:'#cfe9ff', 'font-size':24, 'font-weight':600, 'font-family':SANS });
    const cq = T(868, 81, 'queue 0 · 24 slots', { fill:'#7d8a99', 'font-size':14, 'text-anchor':'end' });
    const cSlots = [];
    for (let j = 0; j < 2; j++) for (let i = 0; i < 12; i++) cSlots.push(E('rect', { x:432+i*36.5, y:124+j*34, width:28, height:22, rx:4, fill:H6(COL.led) }));
    const wanEls = [];
    wanEls.push(E('line', { x1:650, y1:220, x2:650, y2:290, stroke:H6(COL.wan), 'stroke-width':3 }));
    T(660, 262, 'cloud MQTT', { fill:'#8b97a5', 'font-size':13 });
    wanEls.push(E('polyline', { points:`190,${WANL} ${WX},${WANL} ${WX},133 890,133`, fill:'none', stroke:H6(COL.wan), 'stroke-width':2.5, 'stroke-opacity':.8 }));
    const wanTxt = T(WX-8, 124, 'WAN · 50 ms', { fill:'#9fd4ff', 'font-size':15, 'text-anchor':'end' });
    T(WX-8, 236, 'task input (HTTP)', { fill:'#6f7c8b', 'font-size':13, 'text-anchor':'end' });
    E('rect', { x:60, y:290, width:1170, height:28, rx:14, fill:'#262c34', stroke:'#4b535d', 'stroke-width':1.5 });
    T(70, 282, 'MQTT broker · EMQX cluster (metro edge)', { fill:'#c9d1da', 'font-size':15, 'font-weight':500 });
    E('line', { x1:190, y1:PEER, x2:1110, y2:PEER, stroke:H6(COL.link), 'stroke-width':1.5, 'stroke-dasharray':'6 6', opacity:.7 });
    T(1120, PEER+5, 'backhaul 1–5 ms', { fill:'#6f7c8b', 'font-size':12 });
    for (const r of ['A','B']) {
      E('rect', { x:90, y:RY[r]-20, width:1120, height:40, rx:6, fill:'#171d25' });
      E('line', { x1:100, y1:RY[r], x2:1200, y2:RY[r], stroke:H6(COL.dash), 'stroke-width':1.5, 'stroke-dasharray':'14 12' });
      T(1210, RY[r]+48, r === 'A' ? 'north road' : 'south road', { fill:'#56616f', 'font-size':13, 'text-anchor':'end' });
    }
    E('rect', { x:mapX(4.5), y:RY.A-24, width:mapX(10.5)-mapX(4.5), height:48, rx:6, fill:'#0a0d11', stroke:'#3a4552', 'stroke-dasharray':'4 4' });
    T((mapX(4.5)+mapX(10.5))/2, RY.A+48, 'tunnel · no coverage', { fill:'#8b97a5', 'font-size':13, 'text-anchor':'middle' });
    const lightC = E('circle', { cx:mapX(LIGHT_X), cy:RY.A-36, r:8, fill:'#ff4545', style:'display:none' });
    const radioG = E('g', {}), PV = {};
    for (const z of zl()) {
      const cx = ZX[z.id], x0 = cx-100, y0 = TOP, ry = z.z === 0 ? RY.A : RY.B;
      const up = E('line', { x1:cx, y1:318, x2:cx, y2:TOP, stroke:H6(COL.link), 'stroke-width':2 });
      const span = E('rect', { x:mapX(z.x-z.r), y:ry-30, width:mapX(z.x+z.r)-mapX(z.x-z.r), height:60, rx:10, fill:H6(COL.dome), 'fill-opacity':.1, stroke:H6(COL.dome), 'stroke-opacity':.6, 'stroke-width':1.5 });
      const spanT = T(mapX(z.x-z.r)+8, ry-38, z.id, { fill:'#c9d3de', 'font-size':15, 'font-weight':600 });
      const g = E('g', { style:'cursor:pointer' });
      g.addEventListener('click', () => { S.sel = z.id; call('onSelect', z.id); });
      const card = E('rect', { x:x0, y:y0, width:200, height:240, rx:14, fill:'#121821', stroke:'#2a3440', 'stroke-width':2 }, g);
      T(x0+16, y0+34, z.id, { fill:'#eef3f8', 'font-size':26, 'font-weight':600, 'font-family':SANS }, g);
      const status = T(x0+186, y0+32, 'online', { fill:'#7d8a99', 'font-size':14, 'text-anchor':'end' }, g);
      const chips = { anchor:T(x0+16, y0+58, 'ANCHOR', { fill:H6(COL.anchor), 'font-size':12, 'font-weight':600, style:'display:none' }, g), serving:T(x0+80, y0+58, 'SERVING', { fill:H6(COL.serving), 'font-size':12, 'font-weight':600, style:'display:none' }, g), exec:T(x0+148, y0+58, 'EXEC', { fill:H6(COL.exec), 'font-size':12, 'font-weight':600, style:'display:none' }, g) };
      let qT;
      ['TOF-Broker → mask','D3QN agent','Priority queue'].forEach((t, i) => {
        E('rect', { x:x0+14, y:y0+70+i*34, width:172, height:28, rx:6, fill:'#19212c', stroke:'#2a3440' }, g);
        T(x0+24, y0+89+i*34, t, { fill:'#c9d3de', 'font-size':14 }, g);
        if (i === 2) qT = T(x0+176, y0+89+i*34, '0', { fill:'#7d8a99', 'font-size':13, 'text-anchor':'end' }, g);
      });
      const slots = [0,1,2,3].map(i => E('rect', { x:x0+14+i*44, y:y0+178, width:36, height:22, rx:4, fill:H6(COL.led) }, g));
      T(x0+16, y0+226, 'mobility · state · tables', { fill:'#6f7c8b', 'font-size':12 }, g);
      const ant = E('line', { x1:cx, y1:620, x2:cx, y2:640, stroke:'#8592a0', 'stroke-width':2 });
      const apC = E('circle', { cx, cy:644, r:5, fill:'#dfe8f2' });
      PV[z.id] = { g, card, status, chips, slots, qT, up, span, spanT, ant, apC };
    }
    const carEls = {};
    for (const c of S.cars) {
      const r = E('rect', { width:30, height:16, rx:4, fill:H6(COL.car), stroke:c.id === HERO ? '#ffffff' : 'none', 'stroke-width':2, style:'display:none;cursor:pointer' });
      r.addEventListener('mousemove', e => { const b = root.getBoundingClientRect(); S.hover = { id:c.id, x:e.clientX-b.left, y:e.clientY-b.top }; });
      r.addEventListener('mouseleave', () => S.hover = null);
      carEls[c.id] = { r, line:E('line', { stroke:'#d8e2ee', 'stroke-opacity':.22, 'stroke-width':1.2, style:'display:none' }, radioG) };
    }
    const heroT = T(-100, -100, 'car 7', { fill:H6(COL.car), 'font-size':14, 'font-weight':600, 'text-anchor':'middle' });
    carEls[HERO].line.setAttribute('stroke', H6(COL.car)); carEls[HERO].line.setAttribute('stroke-opacity', .8); carEls[HERO].line.setAttribute('stroke-dasharray', '5 4');
    const pktG = E('g', {}), fxG = E('g', {});
    function pos2(id) {
      const [k, v] = id.split(':');
      if (k === 'car') { const c = CAR[v], p = carPos(c); return { x:mapX(p.x), y:(p.z > -8 ? RY.A : RY.B) + (c.dir > 0 ? -8 : 8) }; }
      if (k === 'ap') return { x:ZX[v], y:644 };
      if (k === 'fog') return { x:ZX[v], y:395 };
      if (id === 'broker') return { x:650, y:BUS };
      return { x:650, y:150 };
    }
    const xOf = id => !id || id === 'broker' || id === 'cloud' ? 650 : pos2(id).x;
    function seg2(a, b, prev) {
      const ka = kind(a), kb = kind(b);
      if (ka === 'fog' && b === 'cloud') { const x = ZX[a.split(':')[1]]; return [pos2(a), { x, y:TOP }, { x, y:WANL }, { x:WX, y:WANL }, { x:WX, y:133 }, { x:890, y:133 }, pos2(b)]; }
      if (ka === 'fog' && b === 'broker') { const x = ZX[a.split(':')[1]]; return [pos2(a), { x, y:TOP }, { x, y:BUS }]; }
      if (a === 'broker' && kb === 'fog') { const x = ZX[b.split(':')[1]], px = xOf(prev); return [{ x:px, y:BUS }, { x, y:BUS }, { x, y:TOP }, pos2(b)]; }
      if (a === 'broker' && b === 'cloud') { const px = xOf(prev); return [{ x:px, y:BUS }, { x:650, y:BUS }, { x:650, y:220 }, pos2(b)]; }
      if (a === 'cloud' && b === 'broker') return [pos2(a), { x:650, y:220 }, { x:650, y:BUS }];
      if (ka === 'fog' && kb === 'fog') { const xa = ZX[a.split(':')[1]], xb = ZX[b.split(':')[1]]; return [pos2(a), { x:xa, y:TOP }, { x:xa, y:PEER }, { x:xb, y:PEER }, { x:xb, y:TOP }, pos2(b)]; }
      return [pos2(a), pos2(b)];
    }
    const pm = new Map(), fxm = new Map(), C1 = new THREE.Color(), C2 = new THREE.Color();
    function render() {
      for (const c of S.cars) {
        const p = pos2('car:'+c.id), e = carEls[c.id];
        e.r.setAttribute('x', p.x-15); e.r.setAttribute('y', p.y-8); e.r.style.display = '';
        if (c.serv && (S.radio || c.id === HERO)) { const a = pos2('ap:'+c.serv); e.line.setAttribute('x1', p.x); e.line.setAttribute('y1', p.y-8); e.line.setAttribute('x2', a.x); e.line.setAttribute('y2', a.y); e.line.style.display = ''; }
        else e.line.style.display = 'none';
        if (c.id === HERO) { heroT.setAttribute('x', p.x); heroT.setAttribute('y', p.y-16); }
      }
      lightC.style.display = S.light.on ? '' : 'none'; lightC.setAttribute('fill', S.light.red ? '#ff4545' : '#3ddc84');
      for (const z of zl()) {
        const V = PV[z.id], f = z.fade, serv = S.flowActive && hero.serv === z.id, anchor = z.roles.has('anchor');
        const col = !z.alive ? COL.dead : anchor ? COL.anchor : serv ? COL.serving : null;
        V.card.setAttribute('stroke', S.sel === z.id ? '#ffffff' : col ? H6(col) : '#2a3440');
        V.card.setAttribute('stroke-dasharray', z.joined ? 'none' : '8 6');
        V.card.setAttribute('fill', z.alive ? '#121821' : '#15181c');
        V.g.setAttribute('opacity', (z.joined ? 1 : .35)*(z.alive ? 1 : .5)*(.35 + .65*f));
        V.status.textContent = !z.joined ? 'can join later' : !z.alive ? 'offline' : 'online';
        V.status.setAttribute('fill', !z.alive ? '#ff6b6b' : '#7d8a99');
        V.chips.anchor.style.display = anchor ? '' : 'none';
        V.chips.serving.style.display = serv ? '' : 'none';
        V.chips.exec.style.display = z.roles.has('exec') ? '' : 'none';
        V.slots.forEach((s, i) => s.setAttribute('fill', !z.alive ? '#111418' : z.failT[i] > 0 ? H6(COL.fail) : z.slots[i] > 0 ? H6(COL.exec) : H6(COL.led)));
        const q = z.queue + z.qx; V.qT.textContent = q ? q + ' waiting' : '0';
        V.up.setAttribute('opacity', (z.alive ? 1 : .2)*(z.joined ? 1 : .3));
        V.up.setAttribute('stroke-dasharray', z.joined ? 'none' : '4 6');
        const sc = col || COL.dome;
        V.span.setAttribute('fill', H6(sc)); V.span.setAttribute('stroke', H6(sc));
        V.span.setAttribute('fill-opacity', (col && z.alive ? .2 : .08)*f); V.span.setAttribute('stroke-opacity', .7*f);
        V.span.setAttribute('stroke-dasharray', z.joined ? 'none' : '6 6');
        V.spanT.setAttribute('opacity', .3 + .7*f);
        V.apC.setAttribute('fill', z.alive ? (serv ? H6(COL.serving) : '#dfe8f2') : H6(COL.dead));
        V.ant.setAttribute('opacity', .3 + .7*f); V.apC.setAttribute('opacity', .3 + .7*f);
      }
      const hot = Math.min(1, Math.max(0, S.wanLoad)/9), wc = '#' + C1.setHex(COL.wan).lerp(C2.setHex(COL.hot), hot).getHexString();
      wanEls.forEach((e, i) => { e.setAttribute('stroke', wc); e.setAttribute('stroke-width', (i ? 2.5 : 3) + hot*5); });
      wanTxt.textContent = 'WAN · ' + Math.round(S.wanMs) + ' ms'; wanTxt.setAttribute('fill', hot > .5 ? '#ff6b6b' : '#9fd4ff');
      cq.textContent = 'queue ' + S.cloud.queue + ' · 24 slots';
      cSlots.forEach((s, i) => s.setAttribute('fill', S.cloudSlots[i] > 0 ? H6(COL.exec) : H6(COL.led)));
      const live = new Set(S.packets);
      for (const [p, el] of pm) if (!live.has(p)) { el.remove(); pm.delete(p); }
      for (const p of S.packets) {
        let el = pm.get(p);
        if (!el) {
          el = E('g', {}, pktG); const c = H6(PCOL[p.type]);
          if (!p.faint) E('circle', { r:18*p.size, fill:c, opacity:.22 }, el);
          E('circle', { r:(p.faint ? 5 : 9)*p.size, fill:c, opacity:p.faint ? .5 : 1 }, el);
          pm.set(p, el);
        }
        const prev = p.seg > 0 ? p.ids[p.seg-1] : p.from;
        const [A, B, t] = along(seg2(p.ids[p.seg], p.ids[p.seg+1], prev), p.t, (a, b) => Math.hypot(a.x-b.x, a.y-b.y));
        el.setAttribute('transform', 'translate(' + (A.x + (B.x-A.x)*t) + ',' + (A.y + (B.y-A.y)*t) + ')');
      }
      const fl = new Set(S.fx);
      for (const [f, el] of fxm) if (!fl.has(f)) { el.remove(); fxm.delete(f); }
      for (const f of S.fx) {
        let el = fxm.get(f);
        if (!el) { const p = pos2(f.id); el = E('circle', { cx:p.x, cy:p.y, fill:'none', stroke:H6(f.color), 'stroke-width':3 }, fxG); fxm.set(f, el); }
        el.setAttribute('r', 8 + f.t*60); el.setAttribute('opacity', 1 - f.t/.9);
      }
    }
    function screen(id) {
      const p = pos2(id), m = svg.getScreenCTM(), r = root.getBoundingClientRect();
      if (!m) return { x:-999, y:-999 };
      return { x:m.a*p.x + m.e - r.left, y:m.d*p.y + m.f - r.top };
    }
    function layout() {
      const I = S.insets;
      svg.style.left = I.left + 'px'; svg.style.top = I.top + 'px';
      svg.style.width = 'calc(100% - ' + (I.left + I.right) + 'px)'; svg.style.height = 'calc(100% - ' + (I.top + I.bottom) + 'px)';
    }
    function print() {
      const w = window.open('', '_blank'); if (!w) return;
      const c = svg.cloneNode(true); c.removeAttribute('style'); c.setAttribute('width', '100%');
      w.document.write('<!doctype html><title>PCNME — system poster</title><link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet"><style>@page{size:landscape;margin:0}html,body{margin:0;background:#0b0e13;-webkit-print-color-adjust:exact;print-color-adjust:exact}</style>' + c.outerHTML);
      w.document.close(); setTimeout(() => w.print(), 700);
    }
    return { render, screen, place:layout, print, show:on => { if (on) { layout(); render(); } wrap.style.display = on ? '' : 'none'; } };
  }

  const v3 = make3D(), vp = makePoster();
  let view = v3;
  const tagLayer = document.createElement('div'); tagLayer.style.cssText = 'position:absolute;inset:0;pointer-events:none;overflow:hidden;';
  root.appendChild(tagLayer);
  const tip = document.createElement('div');
  tip.style.cssText = 'position:absolute;left:0;top:0;display:none;pointer-events:none;white-space:nowrap;font:500 13px ' + MONO + ';color:#eef3f8;background:rgba(11,14,19,.95);border:1px solid rgba(255,210,63,.5);border-radius:8px;padding:8px 10px;line-height:1.55;';
  root.appendChild(tip);
  function renderTags() {
    const hidden = view === null;
    for (const t of S.tags) {
      if (!t.el) {
        t.el = document.createElement('div');
        t.el.style.cssText = 'position:absolute;left:0;top:0;visibility:hidden;white-space:nowrap;font:600 13px ' + MONO + ';padding:4px 9px;border-radius:6px;background:rgba(11,14,19,.92);border:1px ' + (t.dashed ? 'dashed' : 'solid') + ' ' + t.color + ';color:' + t.color + ';';
        t.el.textContent = t.text; tagLayer.appendChild(t.el);
      }
      if (t.fn) t.el.textContent = t.fn();
      if (hidden) { t.el.style.visibility = 'hidden'; continue; }
      const p = view.screen(t.id), stack = S.tags.filter(o => o.id === t.id && o.ttl > 0);
      const k = stack.indexOf(t);
      t.el.style.opacity = Math.min(1, t.ttl*2);
      t.el.style.transform = 'translate(' + p.x + 'px,' + (p.y - k*30) + 'px) translate(-50%,-150%)';
      t.el.style.visibility = 'visible';
    }
    if (S.hover && !hidden) {
      const c = CAR[S.hover.id];
      tip.innerHTML = '<b style="color:#ffd23f">car ' + c.id + '</b><br>serving <b>' + (c.serv || 'none') + '</b><br>T_exit <b>' + tExit(c) + '</b><br>in-flight tasks <b>' + S.inflight[c.id] + '</b>';
      tip.style.display = ''; tip.style.transform = 'translate(' + (S.hover.x + 16) + 'px,' + (S.hover.y + 12) + 'px)';
    } else tip.style.display = 'none';
  }
  const ro = new ResizeObserver(() => { v3.place(); vp.place(); }); ro.observe(root);
  let last = performance.now(), raf, tickT = 0, alive = true;
  function frame(now) {
    if (!alive) return;
    const dt = Math.min(.05, (now - last)/1000); last = now;
    step(dt); if (view) view.render(now); renderTags();
    tickT -= dt;
    if (tickT <= 0) {
      tickT = .25;
      const zones = {};
      for (const z of zl()) zones[z.id] = {
        id:z.id, alive:z.alive, joined:z.joined, queue:z.queue + z.qx, held:z.held, anchored:z.anchored, replay:z.replay,
        slots:z.slots.map((v, i) => z.failT[i] > 0 ? 'fail' : v > 0 ? 'busy' : 'idle'),
        roles:[...z.roles].concat(S.flowActive && hero.serv === z.id ? ['serving'] : []),
        cars:S.cars.filter(c => c.serv === z.id).map(c => c.id),
        neigh:NEIGH[z.id].map((n, i) => { const id = n && S.zones[n].joined ? n : null; return { slot:'N' + (i+1), id:id || '—', state:!id ? 'empty' : !S.zones[id].alive ? 'offline' : 'fresh' }; }),
      };
      call('onTick', { zones, wanMs:Math.round(S.wanMs), wanHot:Math.min(1, Math.max(0, S.wanLoad)/9), cloudQueue:S.cloud.queue });
    }
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);
  return {
    play:n => runFlow(n),
    reset:() => { reset(); call('onClear'); },
    pause:p => { S.paused = p; },
    next:() => { if (S.paused) S.paused = false; openGate(); },
    setStepMode:m => { S.stepMode = m; if (!m) openGate(); },
    setView:k => { view = k === 'poster' ? vp : k === '3d' ? v3 : null; v3.show(view === v3); vp.show(view === vp); },
    printPoster:() => vp.print(),
    set:o => Object.assign(S, o),
    setInsets:i => { S.insets = { ...S.insets, ...i }; v3.place(); vp.place(); },
    select:z => { S.sel = z; },
    destroy:() => { alive = false; cancelAnimationFrame(raf); ro.disconnect(); reset(); v3.dispose(); root.innerHTML = ''; },
  };
}
window.PCNME = { createPCNME, FLOWS };
})();
