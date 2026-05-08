/* app.jsx — UI glue for the Track Optimizer.
 *
 * Sections:
 *   - Theme (blueprint palette)
 *   - <SelfCheck>     — runs Tests.runAll() on mount
 *   - <Scorecard>     — closed status, utilization, lap time, bbox
 *   - <ConfigPanel>   — JSON editors for catalog & problem
 *   - <Canvas>        — renders the layout, animates during search
 *   - <App>           — search lifecycle, layout caching
 */

const { useState, useEffect, useRef, useMemo, useCallback } = React;

const THEME = {
  bg:           "#0e1620",
  panel:        "#121d2a",
  panelLight:   "#16212f",
  border:       "rgba(120,160,200,0.18)",
  text:         "#cfd9e6",
  textDim:      "#7d8a99",
  accent:       "#7fb0ff",
  accentWarm:   "#ffb47a",
  ok:           "#7be38b",
  bad:          "#ff7a8a",

  // canvas theme
  grid:         "rgba(120,160,200,0.10)",
  boundary:     "rgba(160,180,210,0.55)",
  straightFill: "rgba(127,176,255,0.18)",
  straightDim:  "rgba(127,176,255,0.07)",
  curveFill:    "rgba(255,180,122,0.18)",
  curveDim:     "rgba(255,180,122,0.07)",
  pieceStroke:  "rgba(180,200,225,0.55)",
  rail:         "rgba(220,230,245,0.95)",
  sleeper:      "rgba(160,180,210,0.55)",
  centerline:   "rgba(127,255,200,0.85)",
  start:        "#ffd166",
};

function fmtMm(v) { return (v / 10).toFixed(1) + " cm"; }
function fmtDeg(rad) { return (rad * 180 / Math.PI).toFixed(2) + "°"; }
function fmtSec(s) { return s.toFixed(2) + "s"; }

function SelfCheck({ results }) {
  const passed = results.filter(r => r.ok).length;
  const total = results.length;
  const allOk = passed === total;
  return (
    <div className="card">
      <div className="card-hd">
        <span>Self-check</span>
        <span style={{ color: allOk ? THEME.ok : THEME.bad, fontVariantNumeric: "tabular-nums" }}>
          {passed} / {total} {allOk ? "✓" : "✗"}
        </span>
      </div>
      <div className="tests">
        {results.map((r, i) => (
          <div key={i} className="test-row">
            <span className="dot" style={{ background: r.ok ? THEME.ok : THEME.bad }} />
            <span className="test-name">{r.name}</span>
            <span className="test-detail">{r.detail}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Scorecard({ layout, problem, status, nodes, elapsed, attempts }) {
  if (!layout) {
    return (
      <div className="card">
        <div className="card-hd"><span>Layout</span><span style={{ color: THEME.textDim }}>{status}</span></div>
        <div style={{ color: THEME.textDim, fontSize: 12, padding: "6px 0" }}>
          No closed loop yet. {nodes ? `Searched ${nodes.toLocaleString()} nodes across ${attempts} restarts in ${(elapsed/1000).toFixed(1)}s.` : "Press Search to begin."}
        </div>
      </div>
    );
  }
  const s = layout.score;
  const closure = {
    pos: window.Geom.poseDistance(layout.start, layout.end),
    head: window.Geom.headingDelta(layout.start, layout.end),
  };
  const bw = layout.bbox.maxX - layout.bbox.minX;
  const bh = layout.bbox.maxY - layout.bbox.minY;
  const counts = {};
  for (const p of layout.placements) counts[p.pieceId] = (counts[p.pieceId] || 0) + 1;
  return (
    <div className="card">
      <div className="card-hd">
        <span>Layout</span>
        <span style={{ color: THEME.ok }}>closed ✓</span>
      </div>
      <div className="kv-grid">
        <div className="k">utilization</div><div className="v">{(s.utilization * 100).toFixed(0)}%</div>
        <div className="k">pieces used</div><div className="v">{layout.placements.length}</div>
        <div className="k">path length</div><div className="v">{(s.length / 1000).toFixed(2)} m</div>
        <div className="k">est. lap time</div><div className="v">{fmtSec(s.lapTime)}</div>
        <div className="k">bounding box</div><div className="v">{fmtMm(bw)} × {fmtMm(bh)}</div>
        <div className="k">closure pos</div><div className="v">{closure.pos.toFixed(3)} mm</div>
        <div className="k">closure heading</div><div className="v">{fmtDeg(closure.head)}</div>
        <div className="k">score</div><div className="v">{s.total.toFixed(3)}</div>
      </div>
      <div className="pieces-used">
        {Object.entries(counts).map(([id, n]) => (
          <span key={id} className="pill">
            <span className="pill-label">{id}</span>
            <span className="pill-n">{n}</span>
          </span>
        ))}
      </div>
      {status && <div style={{ color: THEME.textDim, fontSize: 11, marginTop: 6 }}>
        {status} · {nodes.toLocaleString()} nodes · {attempts} restarts · {(elapsed/1000).toFixed(1)}s
      </div>}
    </div>
  );
}

function ConfigEditor({ label, value, onChange, rows = 10 }) {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const [error, setError] = useState(null);
  useEffect(() => { setText(JSON.stringify(value, null, 2)); }, [value]);
  return (
    <div className="card">
      <div className="card-hd">
        <span>{label}</span>
        <span style={{ color: error ? THEME.bad : THEME.textDim, fontSize: 11 }}>
          {error ? error : "JSON"}
        </span>
      </div>
      <textarea
        className="json-edit"
        rows={rows}
        value={text}
        spellCheck={false}
        onChange={(e) => {
          setText(e.target.value);
          try { const parsed = JSON.parse(e.target.value); setError(null); onChange(parsed); }
          catch (err) { setError("invalid JSON"); }
        }}
      />
    </div>
  );
}

function CanvasView({ layout, catalog, problem, onExport, dim, showGrid }) {
  const ref = useRef(null);
  const wrapRef = useRef(null);
  const [size, setSize] = useState({ w: 800, h: 600 });

  useEffect(() => {
    function fit() {
      if (!wrapRef.current) return;
      const r = wrapRef.current.getBoundingClientRect();
      setSize({ w: Math.max(320, Math.floor(r.width)), h: Math.max(240, Math.floor(r.height)) });
    }
    fit();
    const ro = new ResizeObserver(fit);
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = size.w * dpr;
    c.height = size.h * dpr;
    c.style.width = size.w + "px";
    c.style.height = size.h + "px";
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // The renderer uses canvas.width/height directly, so override:
    const fakeCanvas = { width: size.w, height: size.h, getContext: () => ctx, toDataURL: c.toDataURL.bind(c) };
    window.Render.render(fakeCanvas, {
      theme: THEME,
      boundary: problem.boundary,
      layout,
      catalog,
      showGrid: showGrid !== false,
      dimPieces: dim,
    });
  }, [layout, catalog, problem.boundary, size, dim, showGrid]);

  const exportPng = () => {
    if (!ref.current) return;
    const url = ref.current.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url; a.download = "lego-layout.png"; a.click();
  };

  return (
    <div className="canvas-wrap" ref={wrapRef}>
      <canvas ref={ref} />
      <div className="canvas-footer">
        <span>{problem.boundary.width} × {problem.boundary.height} mm boundary · 1 grid square = 10 studs</span>
        <button className="btn ghost" onClick={exportPng}>Export PNG</button>
      </div>
    </div>
  );
}

function App() {
  const [catalog, setCatalog] = useState(window.Catalog.DEFAULT_CATALOG);
  const [baseProblem, setBaseProblem] = useState(window.Catalog.DEFAULT_PROBLEM);

  const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
    "boundaryW": 3000,
    "boundaryH": 2000,
    "invCL": 16,
    "invCR": 16,
    "invS16": 4,
    "invS32": 4,
    "wUtil": 1.0,
    "wLap": 0.3,
    "wComp": 0.2,
    "budgetMs": 5000,
    "showGrid": true,
    "dimDuringSearch": true
  }/*EDITMODE-END*/;

  const [t, setTweak] = window.useTweaks(TWEAK_DEFAULTS);

  // Project tweak values into the problem object the search consumes.
  const problem = useMemo(() => ({
    ...baseProblem,
    boundary: { width: t.boundaryW, height: t.boundaryH },
    inventory: { S16: t.invS16, S32: t.invS32, CL: t.invCL, CR: t.invCR },
    weights: { utilization: t.wUtil, lapTime: t.wLap, compactness: t.wComp },
    searchBudget: { ...baseProblem.searchBudget, ms: t.budgetMs },
  }), [baseProblem, t.boundaryW, t.boundaryH, t.invS16, t.invS32, t.invCL, t.invCR, t.wUtil, t.wLap, t.wComp, t.budgetMs]);

  const setProblem = setBaseProblem; // JSON editor still edits underlying defaults

  const [layout, setLayout] = useState(null);
  const [searching, setSearching] = useState(false);
  const [status, setStatus] = useState("idle");
  const [nodes, setNodes] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const cancelRef = useRef(false);

  const tests = useMemo(() => window.Tests.runAll(), []);
  const allTestsOk = tests.every(t => t.ok);

  const totalAvailable = useMemo(
    () => Object.values(problem.inventory).reduce((a, b) => a + b, 0),
    [problem.inventory]
  );

  const handleSearch = useCallback(() => {
    if (!allTestsOk) return;
    setSearching(true);
    setStatus("searching…");
    setLayout(null);
    setNodes(0); setAttempts(0); setElapsed(0);
    cancelRef.current = false;

    const t0 = performance.now();
    window.Search.search(catalog, problem, {
      onProgress: ({ best, nodes, attempt }) => {
        if (cancelRef.current) return;
        if (best) setLayout(best);
        setNodes(nodes);
        setAttempts(attempt + 1);
        setElapsed(performance.now() - t0);
      },
      onRestart: ({ attempt, totalNodes }) => {
        if (cancelRef.current) return;
        setNodes(totalNodes);
        setAttempts(attempt);
        setElapsed(performance.now() - t0);
      },
      onDone: ({ best, nodes, elapsed: el, restarts }) => {
        setSearching(false);
        setNodes(nodes);
        setAttempts(restarts);
        setElapsed(el);
        if (best) {
          setLayout(best);
          setStatus(`done — best score ${best.score.total.toFixed(3)}`);
        } else {
          setStatus("done — no closed loop found within budget");
        }
      },
    });
  }, [catalog, problem, allTestsOk]);

  const handleStop = () => { cancelRef.current = true; setSearching(false); setStatus("stopped"); };

  return (
    <div className="app">
      <header className="hdr">
        <div className="hdr-l">
          <div className="logo">
            <div className="logo-mark"><div /><div /><div /></div>
            <div className="logo-text">
              <div className="logo-title">Track Optimizer</div>
              <div className="logo-sub">LEGO 9V · R40 lattice · 22.5° steps</div>
            </div>
          </div>
        </div>
        <div className="hdr-r">
          <button
            className={"btn " + (searching ? "ghost" : "primary")}
            disabled={!allTestsOk && !searching}
            onClick={searching ? handleStop : handleSearch}
          >
            {searching ? "Stop" : "Search"}
          </button>
        </div>
      </header>

      <main className="grid">
        <section className="left">
          <CanvasView layout={layout} catalog={catalog} problem={problem} dim={searching && t.dimDuringSearch} showGrid={t.showGrid} />
        </section>
        <aside className="right">
          <Scorecard
            layout={layout}
            problem={problem}
            status={status}
            nodes={nodes}
            elapsed={elapsed}
            attempts={attempts}
          />
          <SelfCheck results={tests} />
          <ConfigEditor label="problem.json" value={problem} onChange={setProblem} rows={14} />
          <ConfigEditor label="catalog.json" value={catalog} onChange={setCatalog} rows={10} />
          <div className="card">
            <div className="card-hd"><span>About</span><span style={{ color: THEME.textDim }}>v1</span></div>
            <p className="prose">
              A search tool for closed LEGO train-track loops. Given an inventory and a play
              area, it finds layouts you can actually build. v1 covers straights and R40
              curves; switches and crossings are deferred. The full design notes — algorithm,
              data model, what was cut and why — are in <code>README.md</code>.
            </p>
          </div>
        </aside>
      </main>

      <window.TweaksPanel>
        <window.TweakSection label="Boundary (mm)" />
        <window.TweakSlider label="Width"  value={t.boundaryW} min={800} max={6000} step={100} unit="mm"
                            onChange={(v) => setTweak('boundaryW', v)} />
        <window.TweakSlider label="Height" value={t.boundaryH} min={600} max={4000} step={100} unit="mm"
                            onChange={(v) => setTweak('boundaryH', v)} />

        <window.TweakSection label="Inventory" />
        <window.TweakSlider label="Curves left (CL)"  value={t.invCL}  min={0} max={32} step={1}
                            onChange={(v) => setTweak('invCL', v)} />
        <window.TweakSlider label="Curves right (CR)" value={t.invCR}  min={0} max={32} step={1}
                            onChange={(v) => setTweak('invCR', v)} />
        <window.TweakSlider label="Straight 16"       value={t.invS16} min={0} max={32} step={1}
                            onChange={(v) => setTweak('invS16', v)} />
        <window.TweakSlider label="Straight 32"       value={t.invS32} min={0} max={32} step={1}
                            onChange={(v) => setTweak('invS32', v)} />

        <window.TweakSection label="Score weights" />
        <window.TweakSlider label="Utilization" value={t.wUtil} min={0} max={2} step={0.05}
                            onChange={(v) => setTweak('wUtil', v)} />
        <window.TweakSlider label="Lap speed"   value={t.wLap}  min={0} max={2} step={0.05}
                            onChange={(v) => setTweak('wLap', v)} />
        <window.TweakSlider label="Compactness" value={t.wComp} min={0} max={2} step={0.05}
                            onChange={(v) => setTweak('wComp', v)} />

        <window.TweakSection label="Search" />
        <window.TweakSlider label="Time budget" value={t.budgetMs} min={500} max={15000} step={500} unit="ms"
                            onChange={(v) => setTweak('budgetMs', v)} />

        <window.TweakSection label="Display" />
        <window.TweakToggle label="Show grid"          value={t.showGrid}        onChange={(v) => setTweak('showGrid', v)} />
        <window.TweakToggle label="Dim during search"  value={t.dimDuringSearch} onChange={(v) => setTweak('dimDuringSearch', v)} />
      </window.TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
