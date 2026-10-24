import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  CloudRain,
  Droplets,
  FileImage,
  FlaskConical,
  Gauge,
  ImagePlus,
  Leaf,
  LoaderCircle,
  LockKeyhole,
  RotateCcw,
  ScanSearch,
  ShieldCheck,
  Sprout,
  Thermometer,
  UploadCloud,
  Wind,
  X,
} from 'lucide-react';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const FRUIT_SEASONS = {
  mango: ['Post-Monsoon', 'Winter'],
  apple: ['Spring', 'Summer', 'Autumn', 'Winter'],
  banana: ['Monsoon', 'Post-Monsoon', 'Summer', 'Winter'],
};

const SEASONS = FRUIT_SEASONS.mango;

const initialEnvironment = {
  temperature_c: '28',
  humidity_pct: '75',
  rainfall_mm: '5',
  soil_moisture_pct: '40',
  soil_ph: '6.5',
  season: 'Post-Monsoon',
};

function Logo() {
  return (
    <div className="brand-lockup">
      <div className="brand-mark"><Sprout size={23} strokeWidth={2.1} /></div>
      <div>
        <div className="brand-name">verdant<span>ai</span></div>
        <div className="brand-caption">FRUIT HEALTH LAB</div>
      </div>
    </div>
  );
}

function Field({ label, unit, icon: Icon, value, onChange, name, min, max, step = 'any' }) {
  return (
    <label className="field" htmlFor={name}>
      <span className="field-top"><span>{label}</span><span className="field-unit">{unit}</span></span>
      <span className="field-input-wrap">
        <Icon className="field-icon" size={17} strokeWidth={1.8} />
        <input
          id={name}
          name={name}
          type="number"
          inputMode="decimal"
          value={value}
          min={min}
          max={max}
          step={step}
          required
          onChange={(event) => onChange(name, event.target.value)}
        />
      </span>
    </label>
  );
}

function EmptyResult({ apiState }) {
  return (
    <div className="empty-result">
      <div className="empty-illustration">
        <div className="orbit orbit-one" />
        <div className="orbit orbit-two" />
        <div className="leaf-disc"><Leaf size={47} strokeWidth={1.3} /></div>
        <span className="float-dot dot-a" /><span className="float-dot dot-b" /><span className="float-dot dot-c" />
      </div>
      <p className="eyebrow">YOUR REPORT</p>
      <h3>Let’s check that leaf.</h3>
      <p className="empty-copy">Upload a clear leaf photo and add the growing conditions. Your screening result will appear here.</p>
      <div className="result-steps">
        <div><span>01</span><p>Choose a fruit</p><Check size={15} /></div>
        <div><span>02</span><p>Add image + context</p><Check size={15} /></div>
        <div className={apiState === 'online' ? 'step-ready' : ''}><span>03</span><p>Review your report</p><Activity size={15} /></div>
      </div>
      <div className="privacy-note"><LockKeyhole size={14} /> Images are sent to your configured local API.</div>
    </div>
  );
}

function ResultPanel({ result, apiState, onReset }) {
  const probabilityEntries = useMemo(() => {
    if (!result?.probabilities) return [];
    return Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  }, [result]);

  if (!result) return <EmptyResult apiState={apiState} />;

  const confidence = Number(result.confidence_percent ?? (result.confidence ? result.confidence * 100 : 0)) || 0;
  const fruitName = (result.fruit || 'fruit').toUpperCase();
  const diseaseName = result.predicted_disease || result.prediction || 'Unknown';

  return (
    <div className="result-populated">
      <div className="result-heading-row">
        <div>
          <p className="eyebrow">SCREENING REPORT</p>
          <h3>Analysis complete <CheckCircle2 size={19} /></h3>
        </div>
        <button className="icon-button" type="button" onClick={onReset} aria-label="Clear result" title="Clear result"><RotateCcw size={17} /></button>
      </div>
      <div className="diagnosis-card">
        <div className="diagnosis-topline"><span className="status-pill"><span className="status-dot" /> Model response received</span><span className="class-count">{fruitName} · MULTIMODAL</span></div>
        <div className="diagnosis-main">
          <div className="diagnosis-leaf"><Leaf size={27} /></div>
          <div className="diagnosis-name-wrap">
            <span className="small-label">PREDICTED CONDITION</span>
            <h4>{diseaseName}</h4>
          </div>
        </div>
        <div className="confidence-row"><span>Model confidence</span><strong>{confidence.toFixed(2)}%</strong></div>
        <div className="confidence-track"><span style={{ width: `${Math.min(100, Math.max(0, confidence))}%` }} /></div>
        {result.severity !== undefined && result.severity !== null && (
          <div className="confidence-row" style={{ marginTop: '0.45rem' }}>
            <span>Estimated severity</span>
            <strong>{(Number(result.severity) * 100).toFixed(1)}% <small>({Number(result.severity).toFixed(2)})</small></strong>
          </div>
        )}
        <p className="confidence-disclaimer">Confidence is the model’s score for this input, not a guarantee of real-world accuracy.</p>
      </div>

      <div className="probability-section">
        <div className="section-heading-line"><h4>Class breakdown</h4><span>{probabilityEntries.length} classes</span></div>

        <div className="probability-list">
          {probabilityEntries.map(([name, score], index) => (
            <div className={`probability-row ${index === 0 ? 'probability-top' : ''}`} key={name}>
              <span className="probability-label">{name}</span>
              <div className="probability-track"><span style={{ width: `${Math.min(100, Math.max(0, Number(score) || 0))}%` }} /></div>
              <span className="probability-value">{Number(score).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </div>

      {result.gradcam_image && (
        <div className="gradcam-section">
          <div className="section-heading-line">
            <h4>Grad-CAM Visual Attention</h4>
            <span><ScanSearch size={14} /> Attention Map</span>
          </div>
          <div className="gradcam-card">
            <div className="gradcam-image-frame">
              <img src={result.gradcam_image} alt="Grad-CAM Saliency Overlay" className="gradcam-img" />
              <div className="gradcam-badge">
                <Leaf size={13} /> Saliency Heatmap
              </div>
            </div>
            <div className="gradcam-legend-bar">
              <span className="legend-txt">Low focus</span>
              <div className="legend-spectrum" />
              <span className="legend-txt">High symptom focus</span>
            </div>
            <p className="gradcam-explainer">
              Heatmap highlights the specific leaf patterns and lesion regions that most strongly influenced the Vision Transformer's diagnosis.
            </p>
          </div>
        </div>
      )}

      <div className="environment-report">

        <div className="section-heading-line"><h4>Submitted conditions</h4><span><CheckCircle2 size={14} /> Included</span></div>
        <div className="environment-chips">
          {result.environment && Object.entries(result.environment).map(([key, value]) => (
            <div className="environment-chip" key={key}>
              <span>{key.replaceAll('_', ' ')}</span>
              <strong>{value}{key === 'temperature_c' ? ' °C' : key.endsWith('_pct') ? '%' : key === 'rainfall_mm' ? ' mm' : ''}</strong>
            </div>
          ))}
        </div>
      </div>
      <div className="result-footnote"><ShieldCheck size={16} /><span>Use this result as a screening aid. Confirm uncertain or serious symptoms with a qualified plant-health professional before treatment.</span></div>
    </div>
  );
}

export default function App() {
  const [fruit, setFruit] = useState('mango');
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [environment, setEnvironment] = useState(initialEnvironment);
  const [apiState, setApiState] = useState('checking');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/health`)
      .then((response) => {
        if (!response.ok) throw new Error('API is unavailable');
        return response.json();
      })
      .then(() => active && setApiState('online'))
      .catch(() => active && setApiState('offline'));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl('');
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function updateEnvironment(name, value) {
    setEnvironment((current) => ({ ...current, [name]: value }));
  }

  function acceptFile(nextFile) {
    setError('');
    setResult(null);
    if (!nextFile) return;
    if (!nextFile.type.startsWith('image/')) {
      setError('Choose a valid image file such as JPG, PNG, or WEBP.');
      return;
    }
    if (nextFile.size > MAX_FILE_BYTES) {
      setError('This image is larger than 10 MB. Choose a smaller image.');
      return;
    }
    setFile(nextFile);
  }

  function clearFile() {
    setFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  function validateEnvironment() {
    const numericKeys = ['temperature_c', 'humidity_pct', 'rainfall_mm', 'soil_moisture_pct', 'soil_ph'];
    for (const key of numericKeys) {
      if (environment[key] === '' || !Number.isFinite(Number(environment[key]))) {
        return `Enter a valid value for ${key.replaceAll('_', ' ')}.`;
      }
    }
    const values = Object.fromEntries(numericKeys.map((key) => [key, Number(environment[key])]));
    if (values.humidity_pct < 0 || values.humidity_pct > 100) return 'Humidity must be between 0 and 100%. ';
    if (values.soil_moisture_pct < 0 || values.soil_moisture_pct > 100) return 'Soil moisture must be between 0 and 100%. ';
    if (values.rainfall_mm < 0) return 'Rainfall cannot be negative.';
    if (values.soil_ph < 0 || values.soil_ph > 14) return 'Soil pH must be between 0 and 14.';
    return '';
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError('');
    setResult(null);

    if (fruit !== 'mango' && fruit !== 'apple' && fruit !== 'banana') {
      setError('Please choose a supported fruit: Mango, Apple, or Banana.');
      return;
    }
    if (!file) {
      setError('Upload a leaf image before running the screening.');
      return;
    }
    const validationError = validateEnvironment();
    if (validationError) {
      setError(validationError.trim());
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);
    formData.append('temperature_c', environment.temperature_c);
    formData.append('humidity_pct', environment.humidity_pct);
    formData.append('rainfall_mm', environment.rainfall_mm);
    formData.append('soil_moisture_pct', environment.soil_moisture_pct);
    formData.append('soil_ph', environment.soil_ph);
    formData.append('season', environment.season);

    const endpoint = `${API_BASE}/predict/${fruit}`;


    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        let msg = 'Prediction failed. Check backend.';
        if (data && data.detail) {
          msg = typeof data.detail === 'object' ? (data.detail.message || JSON.stringify(data.detail)) : data.detail;
        } else {
          msg = `Request failed (${response.status}).`;
        }
        setResult(null);
        setError(msg);
        setApiState('online');
        return;
      }
      setError('');
      setResult(data);
      setApiState('online');
    } catch (err) {
      setResult(null);
      setError(err.message || 'Unable to reach the API. Check that FastAPI is running on port 8000.');
      setApiState('offline');
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setResult(null);
    setError('');
    clearFile();
    setEnvironment(initialEnvironment);
  }

  const apiBadge = apiState === 'online'
    ? { text: 'API connected', className: 'api-online' }
    : apiState === 'checking'
      ? { text: 'Checking API…', className: 'api-checking' }
      : { text: 'API offline', className: 'api-offline' };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Logo />
        <div className="sidebar-label">WORKSPACE</div>
        <a className="side-link side-link-active" href="#diagnose"><ScanSearch size={18} /><span>New diagnosis</span><span className="side-link-indicator" /></a>
        <a className="side-link" href="#how-it-works"><Activity size={18} /><span>How it works</span></a>
        <a className="side-link" href="#model-status"><FlaskConical size={18} /><span>Model status</span></a>
        <div className="sidebar-spacer" />
        <div className="sidebar-help">
          <div className="help-icon"><CircleHelp size={17} /></div>
          <strong>New to leaf screening?</strong>
          <p>Use a clear, well-lit photo showing the affected leaf.</p>
          <a href="#how-it-works">See how it works <ArrowRight size={14} /></a>
        </div>
        <div className="sidebar-bottom"><span className="mini-leaf"><Leaf size={14} /></span><span>Built for healthier growing</span></div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="mobile-brand"><Logo /></div>
          <div className="breadcrumb"><span>Workspace</span><span className="breadcrumb-slash">/</span><strong>New diagnosis</strong></div>
          <div className={`api-badge ${apiBadge.className}`}><span className="api-status-dot" />{apiBadge.text}</div>
        </header>

        <div className="content-wrap">
          <section className="welcome-section">
            <div className="welcome-copy">
              <div className="welcome-kicker"><span className="kicker-line" /> PLANT HEALTH, WITH MORE CONTEXT</div>
              <h1>A clearer picture of<br className="desktop-break" /> <em>leaf health.</em></h1>
              <p className="welcome-description">AI-assisted disease screening that brings leaf imagery and growing conditions into one place.</p>
              <div className="welcome-meta"><span><ShieldCheck size={15} /> Research project</span><span className="meta-dot">·</span><span>Image + environment</span></div>
            </div>
            <div className="hero-art" aria-hidden="true">
              <div className="hero-glow" />
              <div className="hero-ring ring-back" />
              <div className="hero-ring ring-front" />
              <div className="hero-plant"><div className="plant-stem" /><div className="plant-leaf leaf-left" /><div className="plant-leaf leaf-right" /><div className="plant-leaf leaf-top" /><div className="plant-leaf leaf-small" /></div>
              <div className="hero-float float-confidence"><span className="float-check"><Check size={13} /></span><div><strong>Context-aware</strong><small>Screening workflow</small></div></div>
              <div className="hero-float float-scan"><ScanSearch size={16} /><span>Leaf scan</span></div>
              <div className="hero-sparkle sparkle-one">✳</div><div className="hero-sparkle sparkle-two">✧</div>
            </div>
          </section>

          <section className="stats-grid" id="model-status" aria-label="Project overview">
            <div className="stat-card"><div className="stat-icon stat-icon-green"><Leaf size={19} /></div><div className="stat-text"><span>Fruit models</span><strong>3 <small>available</small></strong><p>Mango, Apple &amp; Banana</p></div><span className="stat-corner"><ArrowRight size={16} /></span></div>
            <div className="stat-card"><div className="stat-icon stat-icon-yellow"><ScanSearch size={19} /></div><div className="stat-text"><span>Active classes</span><strong>19</strong><p>8 Mango + 4 Apple + 7 Banana</p></div><span className="stat-corner"><ArrowRight size={16} /></span></div>
            <div className="stat-card"><div className="stat-icon stat-icon-blue"><Wind size={19} /></div><div className="stat-text"><span>Input modalities</span><strong>02</strong><p>Image + environment</p></div><span className="stat-corner"><ArrowRight size={16} /></span></div>
          </section>

          <section className="workspace-grid" id="diagnose">
            <div className="form-panel panel-card">
              <div className="panel-heading">
                <div className="step-number">01</div>
                <div><p className="eyebrow">START A SCREENING</p><h2>Leaf health check</h2><p className="panel-subtitle">Choose a fruit, add a photo, then share its environment.</p></div>
                <span className="panel-secure"><LockKeyhole size={14} /> Private session</span>
              </div>

              <form onSubmit={handleSubmit}>
                <div className="form-section">
                  <div className="section-title-row"><h3>Which fruit are we checking?</h3><span className="required-label">Required</span></div>
                  <div className="fruit-options">
                    <button
                      className={`fruit-option ${fruit === 'apple' ? 'fruit-selected' : ''}`}
                      type="button"
                      onClick={() => {
                        setFruit('apple');
                        setError('');
                        if (!FRUIT_SEASONS.apple.includes(environment.season)) {
                          setEnvironment((prev) => ({ ...prev, season: 'Spring' }));
                        }
                      }}
                    >
                      <span className="fruit-emoji fruit-apple">🍎</span>
                      <span className="fruit-name">Apple</span>
                      <span className="fruit-state fruit-state-ready"><span /> Ready to use</span>
                    </button>
                    <button
                      className={`fruit-option ${fruit === 'mango' ? 'fruit-selected' : ''}`}
                      type="button"
                      onClick={() => {
                        setFruit('mango');
                        setError('');
                        if (!FRUIT_SEASONS.mango.includes(environment.season)) {
                          setEnvironment((prev) => ({ ...prev, season: 'Post-Monsoon' }));
                        }
                      }}
                    >
                      <span className="fruit-emoji fruit-mango">🥭</span>
                      <span className="fruit-name">Mango</span>
                      <span className="fruit-state fruit-state-ready"><span /> Ready to use</span>
                    </button>
                    <button
                      className={`fruit-option ${fruit === 'banana' ? 'fruit-selected' : ''}`}
                      type="button"
                      onClick={() => {
                        setFruit('banana');
                        setError('');
                        if (!FRUIT_SEASONS.banana.includes(environment.season)) {
                          setEnvironment((prev) => ({ ...prev, season: 'Monsoon' }));
                        }
                      }}
                    >
                      <span className="fruit-emoji fruit-banana">🍌</span>
                      <span className="fruit-name">Banana</span>
                      <span className="fruit-state fruit-state-ready"><span /> Ready to use</span>
                    </button>
                  </div>
                  <p className="field-hint"><span className="hint-dot" /> Mango, Apple, and Banana models are connected and ready with Grad-CAM explainability.</p>
                </div>



                <div className="form-divider" />

                <div className="form-section">
                  <div className="section-title-row"><h3>Upload a leaf image</h3><span className="required-label">Required</span></div>
                  {!file ? (
                    <div
                      className={`upload-zone ${dragging ? 'upload-dragging' : ''}`}
                      role="button"
                      tabIndex={0}
                      onClick={() => fileInputRef.current?.click()}
                      onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') fileInputRef.current?.click(); }}
                      onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                      onDragLeave={() => setDragging(false)}
                      onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files?.[0]); }}
                    >
                      <input ref={fileInputRef} type="file" accept="image/*" className="visually-hidden" onChange={(event) => acceptFile(event.target.files?.[0])} />
                      <div className="upload-icon-wrap"><UploadCloud size={24} strokeWidth={1.8} /></div>
                      <div className="upload-title">Drop your leaf photo here</div>
                      <p>or <span>browse files</span> on your device</p>
                      <div className="upload-formats"><span><FileImage size={13} /> JPG, PNG, WEBP</span><i /> <span>Max 10 MB</span></div>
                    </div>
                  ) : (
                    <div className="file-preview-card">
                      {previewUrl ? <img src={previewUrl} alt="Selected leaf preview" /> : null}
                      <div className="file-preview-info"><strong>{file.name}</strong><span>{(file.size / (1024 * 1024)).toFixed(2)} MB · Ready to analyze</span><span className="file-ready"><CheckCircle2 size={13} /> Image selected</span></div>
                      <button className="icon-button file-remove" type="button" onClick={clearFile} aria-label="Remove selected image"><X size={16} /></button>
                    </div>
                  )}
                  <p className="field-hint upload-hint"><ImagePlus size={14} /> Best results: focus on one leaf, use natural light, avoid heavy blur.</p>
                </div>

                <div className="form-divider" />

                <div className="form-section">
                  <div className="section-title-row"><div><h3>Growing conditions</h3><p className="section-description">Use measured values where possible.</p></div><span className="context-tag"><Sprout size={13} /> Environment</span></div>
                  <div className="example-note"><span className="example-note-icon"><CircleHelp size={15} /></span><p>Sample values are filled in to make testing easier. Replace them with the conditions for your plant.</p></div>
                  <div className="fields-grid">
                    <Field label="Temperature" unit="°C" icon={Thermometer} name="temperature_c" value={environment.temperature_c} onChange={updateEnvironment} min="-10" max="60" />
                    <Field label="Humidity" unit="%" icon={Droplets} name="humidity_pct" value={environment.humidity_pct} onChange={updateEnvironment} min="0" max="100" />
                    <Field label="Rainfall" unit="mm" icon={CloudRain} name="rainfall_mm" value={environment.rainfall_mm} onChange={updateEnvironment} min="0" />
                    <Field label="Soil moisture" unit="%" icon={Droplets} name="soil_moisture_pct" value={environment.soil_moisture_pct} onChange={updateEnvironment} min="0" max="100" />
                    <Field label="Soil pH" unit="pH" icon={FlaskConical} name="soil_ph" value={environment.soil_ph} onChange={updateEnvironment} min="0" max="14" />
                    <label className="field" htmlFor="season"><span className="field-top"><span>Season</span><span className="field-unit">Required</span></span><span className="field-input-wrap select-wrap"><Sprout className="field-icon" size={17} strokeWidth={1.8} /><select id="season" name="season" value={environment.season} onChange={(event) => updateEnvironment('season', event.target.value)} required>{(FRUIT_SEASONS[fruit] || SEASONS).map((season) => <option value={season} key={season}>{season}</option>)}</select><ChevronDown className="select-chevron" size={16} /></span></label>

                  </div>
                </div>

                {error && <div className="error-message" role="alert"><CircleHelp size={17} /><span>{error}</span></div>}

                <div className="form-actions">
                  <button className="primary-button" type="submit" disabled={loading || apiState === 'checking'}>
                    {loading ? <><LoaderCircle size={18} className="spin" /> Analyzing leaf…</> : <><ScanSearch size={18} /> Analyze leaf <ArrowRight size={17} /></>}
                  </button>
                  <button className="secondary-button" type="button" onClick={resetForm} disabled={loading}><RotateCcw size={16} /> Reset</button>
                </div>
                <p className="form-legal"><ShieldCheck size={14} /> Screening support only — not a substitute for expert diagnosis.</p>
              </form>
            </div>

            <aside className="result-panel panel-card" aria-live="polite">
              <div className="result-panel-top"><div><p className="eyebrow">YOUR INSIGHTS</p><h2>Diagnosis report</h2></div><span className={`result-live-dot ${apiState === 'online' ? 'live' : ''}`} title={apiBadge.text} /></div>
              <ResultPanel result={result} apiState={apiState} onReset={() => setResult(null)} />
            </aside>
          </section>

          <section className="how-section" id="how-it-works">
            <div className="how-heading"><p className="eyebrow">SIMPLE BY DESIGN</p><h2>From leaf photo to useful insight.</h2><p>One guided workflow combines a visual signal with the growing context you provide.</p></div>
            <div className="how-grid">
              <div className="how-card"><div className="how-card-icon icon-cream"><ImagePlus size={21} /></div><span className="how-step">STEP 01</span><h3>Capture</h3><p>Upload a focused, well-lit image of the leaf you want to assess.</p></div>
              <div className="how-card"><div className="how-card-icon icon-sage"><Wind size={21} /></div><span className="how-step">STEP 02</span><h3>Add context</h3><p>Share temperature, humidity, rainfall, soil measurements and season.</p></div>
              <div className="how-card"><div className="how-card-icon icon-lilac"><Gauge size={21} /></div><span className="how-step">STEP 03</span><h3>Review</h3><p>See the model’s predicted class and probability breakdown for review.</p></div>
            </div>
          </section>

          <footer className="footer"><Logo /><div className="footer-note">A student research project · AI-assisted plant health screening</div><a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">API documentation <ArrowDownToLine size={14} /></a></footer>
        </div>
      </main>
    </div>
  );
}
