/* app.jsx — V2 port-pair optimizer web frontend.
 *
 * POSTs run parameters to the Python backend (server.py), streams progress
 * lines, then renders the resulting layout to a <canvas> via Render.render
 * (defined in render.js, loaded before this file). The matplotlib PNG outputs
 * remain available as fallback views (PNG / Pareto / Infeasible).
 */

const { useState, useEffect, useCallback, useRef } = React;

const API_BASE = '';

const DEFAULTS = {
  config: 'default',
  pop_size: 500,
  n_gen: 200,
  heuristic_ratio: 0.30,
  crossover_prob: 0.9,
  mutation_prob: 0.1,
  n_workers: 16,
};

const VIEW = {
  CANVAS: 'canvas',
  CANVAS_INFEASIBLE: 'canvas_infeasible',
  SHOWCASE: 'showcase',
  TOPOLOGY: 'topology',
  PNG: 'png',
  PNG_INFEASIBLE: 'png_infeasible',
  PARETO: 'pareto',
};

function App() {
  const [tweaks, setTweak] = useTweaks(DEFAULTS);
  const [configs, setConfigs] = useState([]);
  const [catalog, setCatalog] = useState(null);
  const [runState, setRunState] = useState('idle');
  const [logLines, setLogLines] = useState([]);
  const [result, setResult] = useState(null);
  const [view, setView] = useState(VIEW.CANVAS);
  const [error, setError] = useState(null);
  const [reloadMsg, setReloadMsg] = useState(null);
  const [showcase, setShowcase] = useState(null);
  const [topologyLayouts, setTopologyLayouts] = useState(null);
  const [topologyIdx, setTopologyIdx] = useState(0);

  const loadShowcase = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/showcase`);
      const data = await resp.json();
      if (data && data.pieces) {
        setShowcase(data);
        setView(VIEW.SHOWCASE);
      }
    } catch (e) {
      setError(`Showcase load failed: ${e}`);
    }
  }, []);

  const loadTopology = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/topology-showcase`);
      const data = await resp.json();
      if (data && Array.isArray(data.layouts)) {
        setTopologyLayouts(data.layouts);
        setTopologyIdx(0);
        setView(VIEW.TOPOLOGY);
      } else if (data && data.error) {
        setError(`Topology showcase load failed: ${data.error}`);
      }
    } catch (e) {
      setError(`Topology showcase load failed: ${e}`);
    }
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/configs`)
      .then((r) => r.json())
      .then((data) => setConfigs(data.configs || []))
      .catch(() => setConfigs(['default', 'with_switches', 'with_crossing', 'compact']));
    fetch(`${API_BASE}/api/catalog`)
      .then((r) => r.json())
      .then((data) => {
        if (data && data.pieces) setCatalog(data);
      })
      .catch(() => {});
  }, []);

  const reload = useCallback(async () => {
    setReloadMsg('Reloading…');
    try {
      const resp = await fetch(`${API_BASE}/api/reload`, { method: 'POST' });
      const data = await resp.json();
      if (data.ok) {
        setReloadMsg(`Reloaded ${data.n_dropped} modules`);
      } else {
        setReloadMsg(`Reload failed: ${data.error}`);
      }
      // Refresh catalog as well — kind/port edits show up immediately.
      fetch(`${API_BASE}/api/catalog`)
        .then((r) => r.json())
        .then((d) => { if (d && d.pieces) setCatalog(d); });
    } catch (e) {
      setReloadMsg(`Reload error: ${e}`);
    }
    setTimeout(() => setReloadMsg(null), 4000);
  }, []);

  const run = useCallback(async () => {
    setRunState('running');
    setLogLines([]);
    setResult(null);
    setError(null);
    setView(VIEW.CANVAS);

    try {
      const resp = await fetch(`${API_BASE}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tweaks),
      });

      if (!resp.body) {
        const data = await resp.json();
        if (data.success) { setResult(data); setRunState('done'); }
        else { setError(data.error || 'unknown error'); setRunState('error'); }
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          let msg;
          try { msg = JSON.parse(line); } catch { continue; }
          if (msg.type === 'log') {
            setLogLines((prev) => [...prev.slice(-29), msg.line]);
          } else if (msg.type === 'result') {
            setResult(msg);
            setRunState('done');
          } else if (msg.type === 'error') {
            setError(msg.error);
            setRunState('error');
          }
        }
      }
    } catch (e) {
      setError(String(e));
      setRunState('error');
    }
  }, [tweaks]);

  const pngUrl = (() => {
    if (!result) return null;
    const ts = result.timestamp || Date.now();
    if (view === VIEW.PNG && result.best_layout) return `${API_BASE}/${result.best_layout}?t=${ts}`;
    if (view === VIEW.PNG_INFEASIBLE && result.best_infeasible) return `${API_BASE}/${result.best_infeasible}?t=${ts}`;
    if (view === VIEW.PARETO && result.pareto_front) return `${API_BASE}/${result.pareto_front}?t=${ts}`;
    return null;
  })();

  const showCanvas =
    view === VIEW.CANVAS
    || view === VIEW.CANVAS_INFEASIBLE
    || view === VIEW.SHOWCASE
    || view === VIEW.TOPOLOGY;
  const topologyLayout =
    topologyLayouts && topologyLayouts[topologyIdx]
      ? topologyLayouts[topologyIdx]
      : null;
  const layoutToShow = view === VIEW.TOPOLOGY
    ? topologyLayout
    : view === VIEW.SHOWCASE
      ? showcase
      : view === VIEW.CANVAS_INFEASIBLE
        ? (result && result.layout_infeasible)
        : (result && result.layout);
  const boundaryCaption = view === VIEW.TOPOLOGY && topologyLayout
    ? `${topologyLayout.name} — ${topologyLayout.description} · `
      + `cycles=${topologyLayout.stats.n_cycles} components=${topologyLayout.stats.n_components} `
      + `loose=${topologyLayout.stats.n_loose_ports} closure=${topologyLayout.stats.max_closure_pos_studs.toFixed(1)} studs`
    : (layoutToShow && layoutToShow.boundary_mm
        ? `${Math.round(layoutToShow.boundary_mm.width)} × ${Math.round(layoutToShow.boundary_mm.height)} mm boundary · 1 grid square = 10 studs`
        : null);

  return (
    <div className="app">
      <header className="hdr">
        <div className="logo">
          <div className="logo-mark"><div /><div /><div /></div>
          <div>
            <div className="logo-title">LEGO Track Optimizer</div>
            <div className="logo-sub">v2 port-pair · NSGA-II · python backend</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {reloadMsg && (
            <span style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'ui-monospace, monospace' }}>
              {reloadMsg}
            </span>
          )}
          <span className={`badge ${runState}`}>{runState}</span>
          <button
            className="btn"
            disabled={runState === 'running'}
            onClick={loadShowcase}
            title="Render every piece kind side-by-side (no optimization)"
          >
            Showcase
          </button>
          <button
            className="btn"
            disabled={runState === 'running'}
            onClick={loadTopology}
            title="Five hand-built layouts (circle, oval, siding, figure-8, double-crossover) proving the encoding works"
          >
            Topology
          </button>
          <button
            className="btn"
            disabled={runState === 'running'}
            onClick={reload}
            title="Reload Python backend (picks up edits to src_v2/*.py)"
          >
            ↻ Reload backend
          </button>
          <button
            className="btn primary"
            disabled={runState === 'running'}
            onClick={run}
          >
            {runState === 'running' ? 'Running…' : 'Run optimization'}
          </button>
        </div>
      </header>

      <div className="grid">
        <div className="left">
          <div className="canvas-wrap">
            {showCanvas ? (
              <LayoutCanvas
                layout={layoutToShow}
                catalog={catalog}
                runState={runState}
              />
            ) : pngUrl ? (
              <img src={pngUrl} alt={view} />
            ) : (
              <div className="empty">
                {runState === 'running'
                  ? 'Running… layout will appear when complete'
                  : 'Click "Run optimization" to start'}
              </div>
            )}
            {showCanvas && boundaryCaption && (
              <div className="canvas-caption">{boundaryCaption}</div>
            )}
            {view === VIEW.TOPOLOGY && topologyLayouts && (
              <div style={{
                position: 'absolute', left: 12, right: 12, top: 12,
                display: 'flex', gap: 6, justifyContent: 'center',
                pointerEvents: 'auto', zIndex: 10,
              }}>
                {topologyLayouts.map((L, idx) => (
                  <button
                    key={L.name}
                    className={`tab ${idx === topologyIdx ? 'active' : ''}`}
                    onClick={() => setTopologyIdx(idx)}
                  >{L.name}</button>
                ))}
              </div>
            )}
            <div className="canvas-footer">
              <div className="tabs">
                <button
                  className={`tab ${view === VIEW.CANVAS ? 'active' : ''}`}
                  disabled={!result?.layout}
                  onClick={() => setView(VIEW.CANVAS)}
                >Canvas</button>
                <button
                  className={`tab ${view === VIEW.CANVAS_INFEASIBLE ? 'active' : ''}`}
                  disabled={!result?.layout_infeasible}
                  onClick={() => setView(VIEW.CANVAS_INFEASIBLE)}
                >Canvas (infeas.)</button>
                <button
                  className={`tab ${view === VIEW.SHOWCASE ? 'active' : ''}`}
                  disabled={!showcase}
                  onClick={() => setView(VIEW.SHOWCASE)}
                >Showcase</button>
                <button
                  className={`tab ${view === VIEW.TOPOLOGY ? 'active' : ''}`}
                  disabled={!topologyLayouts}
                  onClick={() => setView(VIEW.TOPOLOGY)}
                >Topology</button>
                <button
                  className={`tab ${view === VIEW.PNG ? 'active' : ''}`}
                  disabled={!result?.best_layout}
                  onClick={() => setView(VIEW.PNG)}
                >PNG</button>
                <button
                  className={`tab ${view === VIEW.PNG_INFEASIBLE ? 'active' : ''}`}
                  disabled={!result?.best_infeasible}
                  onClick={() => setView(VIEW.PNG_INFEASIBLE)}
                >PNG (infeas.)</button>
                <button
                  className={`tab ${view === VIEW.PARETO ? 'active' : ''}`}
                  disabled={!result?.pareto_front}
                  onClick={() => setView(VIEW.PARETO)}
                >Pareto</button>
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span>{result ? `${result.elapsed_sec.toFixed(1)}s · ${tweaks.config}` : '—'}</span>
                {showCanvas && layoutToShow && (
                  <ExportPNGButton view={view} config={tweaks.config} />
                )}
              </div>
            </div>
          </div>
          <ProgressLog lines={logLines} state={runState} error={error} />
        </div>

        <div className="right">
          <ConfigCard
            tweaks={tweaks} setTweak={setTweak} configs={configs}
            disabled={runState === 'running'}
          />
          <ScoreCard result={result} />
        </div>
      </div>
    </div>
  );
}

function LayoutCanvas({ layout, catalog, runState }) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);

  // Resize the canvas to match its container, accounting for devicePixelRatio.
  const resize = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = wrap.getBoundingClientRect();
    const cssW = Math.max(200, Math.floor(rect.width));
    const cssH = Math.max(200, Math.floor(rect.height));
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
  }, []);

  useEffect(() => {
    resize();
    const obs = new ResizeObserver(() => { resize(); doRender(); });
    if (wrapRef.current) obs.observe(wrapRef.current);
    return () => obs.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const doRender = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !window.Render) return;
    if (!layout || !catalog) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#0e1620';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#7d8a99';
      ctx.font = '12px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const msg = runState === 'running'
        ? 'Running… layout will appear when complete'
        : (catalog ? 'No layout yet — click Run.' : 'Loading catalog…');
      ctx.fillText(msg, canvas.width / 2, canvas.height / 2);
      return;
    }
    window.Render.render(canvas, { layout, catalog });
  }, [layout, catalog, runState]);

  useEffect(() => { resize(); doRender(); }, [doRender, resize]);

  return (
    <div ref={wrapRef} style={{ flex: 1, minHeight: 0, display: 'flex' }}>
      <canvas ref={canvasRef} />
    </div>
  );
}

function ExportPNGButton({ view, config }) {
  const onClick = useCallback(() => {
    const canvas = document.querySelector('.canvas-wrap canvas');
    if (!canvas || !window.Render) return;
    const dataUrl = window.Render.toPNG(canvas);
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `layout_${config}_${view}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [view, config]);
  return (
    <button className="tab" onClick={onClick} title="Export current canvas to PNG">
      Export PNG
    </button>
  );
}

function ConfigCard({ tweaks, setTweak, configs, disabled }) {
  return (
    <div className="card">
      <div className="card-hd"><span>Config</span><span>{tweaks.config}</span></div>
      <div className="form-row">
        <label htmlFor="cfg-name">Config</label>
        <select
          id="cfg-name" value={tweaks.config} disabled={disabled}
          onChange={(e) => setTweak('config', e.target.value)}
        >
          {configs.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
      <div className="form-row">
        <label htmlFor="cfg-pop">Population</label>
        <input
          id="cfg-pop" type="number" min={20} step={50} disabled={disabled}
          value={tweaks.pop_size}
          onChange={(e) => setTweak('pop_size', Math.max(20, Number(e.target.value) || 0))}
        />
      </div>
      <div className="form-row">
        <label htmlFor="cfg-gen">Generations</label>
        <input
          id="cfg-gen" type="number" min={1} step={10} disabled={disabled}
          value={tweaks.n_gen}
          onChange={(e) => setTweak('n_gen', Math.max(1, Number(e.target.value) || 0))}
        />
      </div>
      <div className="form-row">
        <label htmlFor="cfg-heur">Heuristic %</label>
        <input
          id="cfg-heur" type="number" min={0} max={1} step={0.05} disabled={disabled}
          value={tweaks.heuristic_ratio}
          onChange={(e) => setTweak('heuristic_ratio', Math.max(0, Math.min(1, Number(e.target.value) || 0)))}
        />
      </div>
      <div className="form-row">
        <label htmlFor="cfg-cx">Crossover</label>
        <input
          id="cfg-cx" type="number" min={0} max={1} step={0.05} disabled={disabled}
          value={tweaks.crossover_prob}
          onChange={(e) => setTweak('crossover_prob', Math.max(0, Math.min(1, Number(e.target.value) || 0)))}
        />
      </div>
      <div className="form-row">
        <label htmlFor="cfg-mut">Mutation</label>
        <input
          id="cfg-mut" type="number" min={0} max={1} step={0.05} disabled={disabled}
          value={tweaks.mutation_prob}
          onChange={(e) => setTweak('mutation_prob', Math.max(0, Math.min(1, Number(e.target.value) || 0)))}
        />
      </div>
      <div className="form-row">
        <label htmlFor="cfg-workers">Workers</label>
        <input
          id="cfg-workers" type="number" min={1} max={32} step={1} disabled={disabled}
          value={tweaks.n_workers}
          onChange={(e) => setTweak('n_workers', Math.max(1, Math.min(32, Number(e.target.value) || 1)))}
        />
      </div>
    </div>
  );
}

function ScoreCard({ result }) {
  if (!result || !result.summary) {
    return (
      <div className="card">
        <div className="card-hd"><span>Scorecard</span></div>
        <p style={{ color: 'var(--text-dim)', fontSize: 12, margin: 0 }}>
          No result yet.
        </p>
      </div>
    );
  }
  const s = result.summary;
  const fmt = (v, d = 3) => (typeof v === 'number' ? v.toFixed(d) : '—');
  return (
    <div className="card">
      <div className="card-hd"><span>Scorecard</span><span>{s.config}</span></div>
      <div className="kv-grid">
        <span className="k">Utilization</span>
        <span className="v">{(100 * (s.best_util ?? 0)).toFixed(1)}%</span>
        <span className="k">Min speed</span>
        <span className="v">{fmt(s.best_min_speed, 2)} m/s</span>
        <span className="k">Pieces</span>
        <span className="v">{s.n_pieces ?? 0} / {s.total_inventory ?? '—'}</span>
        <span className="k">Components</span>
        <span className="v">{s.n_components ?? 0}</span>
        <span className="k">Cycles</span>
        <span className="v">{s.n_cycles ?? 0}</span>
        <span className="k">Closure (pos)</span>
        <span className="v">{fmt(s.closure_pos, 3)} studs</span>
        <span className="k">Closure (ang)</span>
        <span className="v">{fmt(s.closure_angle_deg, 3)}°</span>
        <span className="k">Feasible</span>
        <span className="v">{s.n_feasible ?? 0} / {s.pop_size ?? 0}</span>
      </div>
      {s.inventory && Object.keys(s.inventory).length > 0 && (
        <div className="pieces-used">
          {Object.entries(s.inventory).map(([pid, avail]) => {
            const used = (s.piece_counts && s.piece_counts[pid]) || 0;
            const exhausted = used >= avail;
            const unused = used === 0;
            return (
              <span key={pid} className={`pill ${exhausted ? 'pill-full' : ''} ${unused ? 'pill-empty' : ''}`}>
                <span className="pill-n">{used}/{avail}</span> {pid}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ProgressLog({ lines, state, error }) {
  if (state === 'idle' && lines.length === 0) return null;
  return (
    <div className="card" style={{ maxHeight: 220, overflowY: 'auto' }}>
      <div className="card-hd"><span>Progress</span><span>{state}</span></div>
      <div className="progress">
        {lines.map((l, i) => (
          <div key={i} className={`progress-line ${i === lines.length - 1 ? 'last' : ''}`}>{l}</div>
        ))}
        {error && <div className="progress-line" style={{ color: 'var(--bad)' }}>Error: {error}</div>}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
