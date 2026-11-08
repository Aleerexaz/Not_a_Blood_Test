import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { predictImage, health, API_BASE, scannerStatus, captureAndPredict, openScannerDemo, consumeLatestAndPredict, captureMultiAndPredict } from './api'
import FeedbackPanel from './components/FeedbackPanel'

export default function PredictPage() {
  const navigate = useNavigate()
  const [ready, setReady] = useState(false)
  const [scannerReady, setScannerReady] = useState(false)
  const [file, setFile] = useState(null)
  const [files, setFiles] = useState([]) // multi-upload files
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [scanLoading, setScanLoading] = useState(false)
  const [bridgeLoading, setBridgeLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [labels, setLabels] = useState([])
  const [multiLoading, setMultiLoading] = useState(false)
  const [scanSamples, setScanSamples] = useState([]) // collected samples from scanner
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false)
  const [feedbackStats, setFeedbackStats] = useState(null)

  const checkHealth = useCallback(() => {
    setError(null)
    return health()
      .then((h) => { 
        setLabels(h.labels || [])
        setReady(true)
        // Check if scanner is mentioned in health response
        if (h.scanner_available !== undefined) {
          setScannerReady(h.scanner_available)
        }
      })
      .catch(() => setError(`Backend not reachable at ${API_BASE}. Start the API server.`))
  }, [])

  const checkScanner = useCallback(() => {
    scannerStatus()
      .then((s) => setScannerReady(!!s.available))
      .catch(() => setScannerReady(false))
  }, [])

  useEffect(() => { 
    checkHealth()
    checkScanner()
  }, [checkHealth, checkScanner])

  function onSelect(e) {
    const picked = Array.from(e.target.files || [])
    setResult(null)
    setError(null)
    setFiles(picked)
    const first = picked[0] || null
    setFile(first)
    if (first) {
      const url = URL.createObjectURL(first)
      setPreview(url)
    } else {
      setPreview(null)
    }
  }

  // Drag & drop
  function prevent(e){ e.preventDefault(); e.stopPropagation() }
  function onDrop(e){
    prevent(e)
    const picked = Array.from(e.dataTransfer.files || [])
    if (picked.length) {
      setResult(null)
      setError(null)
      setFiles(picked)
      setFile(picked[0])
      const url = URL.createObjectURL(picked[0])
      setPreview(url)
    }
  }

  async function onPredict() {
    // If user has collected scanner samples, aggregate those
    if (scanSamples.length > 0) {
      return onPredictFromSamples()
    }
    if (!file) return
    setLoading(true)
    setError(null)
    resetFeedback()
    try {
      // Start prediction and minimum animation time in parallel
      const [res] = await Promise.all([
        predictImage(file),
        new Promise(resolve => setTimeout(resolve, 2500)) // 2.5s minimum for animation
      ])
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function onPredictMultiUploads() {
    if (!files || files.length === 0) return
    setLoading(true)
    setError(null)
    setResult(null)
    resetFeedback()
    try {
      const used = files.slice(0, 3)
      const labels = new Set()
      const probsSum = {}
      const samples = []
      for (const f of used) {
        const r = await predictImage(f)
        // accumulate
        if (r?.probs) {
          for (const [k, v] of Object.entries(r.probs)) {
            probsSum[k] = (probsSum[k] || 0) + (typeof v === 'number' ? v : 0)
            labels.add(k)
          }
        }
        samples.push({
          predicted: r?.predicted,
          confidence: r?.confidence,
          sorted: r?.sorted,
          corrections: r?.corrections,
          extended: r?.extended,
          features: r?.features,
        })
      }
      const n = samples.length || 1
      const avg = {}
      for (const k of labels) avg[k] = (probsSum[k] || 0) / n
      const finalSorted = Object.entries(avg).sort((a,b)=>b[1]-a[1])
      // majority vote tie-break
      const votes = {}
      for (const s of samples) {
        const lbl = s?.predicted
        if (!lbl) continue
        votes[lbl] = (votes[lbl] || 0) + 1
      }
      let predicted = finalSorted[0]?.[0]
      const topVote = Object.entries(votes).sort((a,b)=> b[1]-a[1] || (avg[b[0]]||0)-(avg[a[0]]||0))[0]?.[0]
      if (topVote && votes[topVote] > (votes[predicted]||0)) predicted = topVote
      const extended = aggregateExtended(samples.map(s => s.extended).filter(Boolean))
      // choose representative features by max skeleton_pixels
      const rep = samples.reduce((best, s) => {
        const sk = (s?.features && typeof s.features.skeleton_pixels === 'number') ? s.features.skeleton_pixels : -1
        const bsk = (best?.features && typeof best.features.skeleton_pixels === 'number') ? best.features.skeleton_pixels : -1
        return sk > bsk ? s : best
      }, null)
      setResult({
        predicted,
        confidence: avg[predicted] || 0,
        probs: avg,
        sorted: finalSorted,
        samples,
        extended,
        features: rep?.features || null,
        source: 'upload-multi',
        feedback_enabled: true  // Enable feedback for multi-upload
      })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function onScanAndPredict() {
    setLoading(true)
    setError(null)
    setResult(null)
    resetFeedback()
    try {
      // Capture from scanner and predict in one call
      const [res] = await Promise.all([
        captureAndPredict(),
        new Promise(resolve => setTimeout(resolve, 2500)) // 2.5s minimum for animation
      ])
      setResult(res)
      // Optionally show the captured image (if you want to display it)
      // The backend could return the image, or we fetch it separately
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function onMultiScan() {
    setMultiLoading(true)
    setError(null)
    setResult(null)
    resetFeedback()
    try {
      const [res] = await Promise.all([
        captureMultiAndPredict(3),
        new Promise(resolve => setTimeout(resolve, 2500)),
      ])
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setMultiLoading(false)
    }
  }

  // Predict from collected scanner samples (aggregate up to 3)
  async function onPredictFromSamples() {
    if (!scanSamples.length) return
    setLoading(true)
    setError(null)
    setResult(null)
    resetFeedback()
    try {
      const used = scanSamples.slice(0, 3)
      const labels = new Set()
      const probsSum = {}
      const samples = []
      for (const s of used) {
        const r = await predictImage(s.file)
        if (r?.probs) {
          for (const [k, v] of Object.entries(r.probs)) {
            probsSum[k] = (probsSum[k] || 0) + (typeof v === 'number' ? v : 0)
            labels.add(k)
          }
        }
        samples.push({
          predicted: r?.predicted,
          confidence: r?.confidence,
          sorted: r?.sorted,
          corrections: r?.corrections,
          extended: r?.extended,
          features: r?.features,
        })
      }
      const n = samples.length || 1
      const avg = {}
      for (const k of labels) avg[k] = (probsSum[k] || 0) / n
      const finalSorted = Object.entries(avg).sort((a,b)=>b[1]-a[1])
      const votes = {}
      for (const s of samples) {
        const lbl = s?.predicted
        if (!lbl) continue
        votes[lbl] = (votes[lbl] || 0) + 1
      }
      let predicted = finalSorted[0]?.[0]
      const topVote = Object.entries(votes).sort((a,b)=> b[1]-a[1] || (avg[b[0]]||0)-(avg[a[0]]||0))[0]?.[0]
      if (topVote && votes[topVote] > (votes[predicted]||0)) predicted = topVote
      const extended = aggregateExtended(samples.map(s => s.extended).filter(Boolean))
      const rep = samples.reduce((best, s) => {
        const sk = (s?.features && typeof s.features.skeleton_pixels === 'number') ? s.features.skeleton_pixels : -1
        const bsk = (best?.features && typeof best.features.skeleton_pixels === 'number') ? best.features.skeleton_pixels : -1
        return sk > bsk ? s : best
      }, null)
      setResult({
        predicted,
        confidence: avg[predicted] || 0,
        probs: avg,
        sorted: finalSorted,
        samples,
        extended,
        features: rep?.features || null,
        source: 'scanner-samples',
        feedback_enabled: true  // Enable feedback for scanner samples
      })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function onOpenDemo() {
    setBridgeLoading(true)
    setError(null)
    try {
      const res = await openScannerDemo()
      // Show hint to user
      setError(`Demo opened. Save the captured image into: ${res.save_dir}. Then click 'Consume Latest & Predict'.`)
    } catch (e) {
      setError(e.message)
    } finally {
      setBridgeLoading(false)
    }
  }

  async function onConsumeLatest() {
    setLoading(true)
    setError(null)
    setResult(null)
    resetFeedback()
    try {
      const [res] = await Promise.all([
        consumeLatestAndPredict(),
        new Promise(resolve => setTimeout(resolve, 1500)),
      ])
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Scan only, show preview, and stage the image for prediction
  function dataURLToFile(dataUrl, filename) {
    try {
      const arr = dataUrl.split(',')
      const mime = (arr[0].match(/:(.*?);/) || [,'image/png'])[1]
      const bstr = atob(arr[1])
      let n = bstr.length
      const u8arr = new Uint8Array(n)
      while (n--) u8arr[n] = bstr.charCodeAt(n)
      return new File([u8arr], filename, { type: mime })
    } catch {
      return null
    }
  }

  async function onScanPreview() {
    setScanLoading(true)
    setError(null)
    setResult(null)
    try {
      const resp = await fetch(`${API_BASE}/scanner/capture`, { method: 'POST' })
      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(text || 'Scan failed')
      }
      const data = await resp.json()
      if (data?.image) {
        setPreview(data.image)
        const f = dataURLToFile(data.image, data?.filename || 'scanner_capture.png')
        if (f) setFile(f)
      } else {
        throw new Error('No image returned from scanner')
      }
    } catch (e) {
      setError(e.message || 'Scan failed')
    } finally {
      setScanLoading(false)
    }
  }

  // Scan and add to samples gallery (without predicting yet)
  async function onScanAddSample() {
    setScanLoading(true)
    setError(null)
    setResult(null)
    try {
      const resp = await fetch(`${API_BASE}/scanner/capture`, { method: 'POST' })
      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(text || 'Scan failed')
      }
      const data = await resp.json()
      if (data?.image) {
        const f = dataURLToFile(data.image, data?.filename || `scanner_sample_${scanSamples.length+1}.png`)
        const sample = { url: data.image, file: f }
        setScanSamples(prev => [...prev, sample])
        setPreview(data.image)
        if (f) setFile(f)
      } else {
        throw new Error('No image returned from scanner')
      }
    } catch (e) {
      setError(e.message || 'Scan failed')
    } finally {
      setScanLoading(false)
    }
  }

  function onRemoveSample(idx) {
    setScanSamples(prev => prev.filter((_, i) => i !== idx))
  }

  function onClearSamples() {
    setScanSamples([])
  }

  const bars = useMemo(() => {
    if (!result?.probs) return []
    const entries = Object.entries(result.probs)
    entries.sort((a, b) => b[1] - a[1])
    return entries
  }, [result])

  const fmt = (v, d = 3) => (typeof v === 'number' ? v.toFixed(d) : '—')
  const [showExplain, setShowExplain] = useState(false)

  // Handle feedback submission
  const handleFeedbackSubmitted = (stats) => {
    setFeedbackSubmitted(true)
    setFeedbackStats(stats)
  }

  // Reset feedback state when new prediction is made
  const resetFeedback = () => {
    setFeedbackSubmitted(false)
    setFeedbackStats(null)
  }

  // Aggregate extended metrics across multiple sample results
  function aggregateExtended(exts) {
    if (!Array.isArray(exts) || exts.length === 0) return null
    // pattern
    const pattVotes = {}
    const pattConfs = []
    for (const e of exts) {
      const l = e?.pattern_type?.label
      const c = typeof e?.pattern_type?.confidence === 'number' ? e.pattern_type.confidence : 0
      if (l) pattVotes[l] = (pattVotes[l] || 0) + 1
      pattConfs.push(c)
    }
  const pattLabel = Object.entries(pattVotes).sort((a,b)=>b[1]-a[1])[0]?.[0]
  const pattConf = pattConfs.length ? pattConfs.reduce((a,b)=>a+b,0)/pattConfs.length : 0
  const pattBasis = `votes=${JSON.stringify(pattVotes)}, mean_conf=${pattConf.toFixed(2)}, samples=${exts.length}`

    // sex
    const sexVotes = { male:0, female:0 }
    const sexProbs = { male:[], female:[] }
    for (const e of exts) {
      const l = e?.sex?.label
      const p = typeof e?.sex?.probability === 'number' ? e.sex.probability : 0
      if (l === 'male' || l === 'female') {
        sexVotes[l] += 1
        sexProbs[l].push(p)
      }
    }
    let sexLabel = 'undetermined', sexProb = 0.5
    if (sexVotes.male || sexVotes.female) {
      sexLabel = sexVotes.female > sexVotes.male ? 'female' : (sexVotes.male > sexVotes.female ? 'male' : 'undetermined')
      if (sexLabel !== 'undetermined') {
        const arr = sexProbs[sexLabel]
        sexProb = arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0.5
      }
    }
    const sexBasis = `votes=${JSON.stringify(sexVotes)}, mean_prob(${sexLabel})=${sexProb.toFixed(2)}, samples=${exts.length}`

    // health index
    const hiScores = exts.map(e => e?.health_index?.score).filter(v => typeof v === 'number')
    const hiScore = hiScores.length ? Math.round(hiScores.reduce((a,b)=>a+b,0)/hiScores.length) : null
  const hiInterp = (s => s==null ? null : s>=70 ? 'good' : s>=40 ? 'average' : 'low')(hiScore)
  const hiBasis = hiScore == null ? 'no valid scores' : `avg_score over ${hiScores.length} samples`

    // rh factor
    const rhVotes = { positive:0, negative:0 }
    const rhProbs = { positive:[], negative:[] }
    for (const e of exts) {
      const l = e?.rh_factor?.label
      const p = typeof e?.rh_factor?.probability === 'number' ? e.rh_factor.probability : 0
      if (l === 'positive' || l === 'negative') {
        rhVotes[l] += 1
        rhProbs[l].push(p)
      }
    }
    let rhLabel = 'undetermined', rhProb = 0.5
    if (rhVotes.positive || rhVotes.negative) {
      rhLabel = rhVotes.positive > rhVotes.negative ? 'positive' : (rhVotes.negative > rhVotes.positive ? 'negative' : 'undetermined')
      if (rhLabel !== 'undetermined') {
        const arr = rhProbs[rhLabel]
        rhProb = arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0.5
      }
    }
    const rhBasis = `votes=${JSON.stringify(rhVotes)}, mean_prob(${rhLabel})=${rhProb.toFixed(2)}, samples=${exts.length}`

    // bp tendency
    const bpVotes = {}
    const bpScores = []
    for (const e of exts) {
      const l = e?.bp_tendency?.label
      const s = e?.bp_tendency?.score
      if (l) bpVotes[l] = (bpVotes[l] || 0) + 1
      if (typeof s === 'number') bpScores.push(s)
    }
  const bpLabel = Object.entries(bpVotes).sort((a,b)=>b[1]-a[1])[0]?.[0]
  const bpScore = bpScores.length ? bpScores.reduce((a,b)=>a+b,0)/bpScores.length : null
  const bpBasis = (bpScores.length || Object.keys(bpVotes).length) ? `votes=${JSON.stringify(bpVotes)}, mean_score=${bpScore!=null?bpScore.toFixed(2):'—'} over ${bpScores.length} samples` : 'no data'

    // stress index
    const stScores = exts.map(e => e?.stress_index?.score).filter(v => typeof v === 'number')
    const stScore = stScores.length ? Math.round(stScores.reduce((a,b)=>a+b,0)/stScores.length) : null
  const stInterp = (s => s==null ? null : s>=70 ? 'elevated irregularity' : s>=40 ? 'mild irregularity' : 'normal')(stScore)
  const stBasis = stScore == null ? 'no valid scores' : `avg_score over ${stScores.length} samples`

    return {
      pattern_type: { label: pattLabel, confidence: pattConf, basis: pattBasis },
      sex: { label: sexLabel, probability: sexProb, basis: sexBasis },
      health_index: { score: hiScore, interpretation: hiInterp, basis: hiBasis },
      rh_factor: { label: rhLabel, probability: rhProb, basis: rhBasis },
      bp_tendency: { label: bpLabel, score: bpScore, basis: bpBasis },
      stress_index: { score: stScore, interpretation: stInterp, basis: stBasis },
    }
  }

  return (
    <div className="page">
      <nav className="nav">
        <button className="back-btn" onClick={() => navigate('/')}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Home
        </button>
        <div className="brand">ABO Predictor</div>
        <div className="nav-right">
          <span className={`badge ${ready ? 'ok' : 'bad'}`}>{ready ? 'API: Online' : 'API: Offline'}</span>
          <button className="link" onClick={checkHealth}>Check</button>
        </div>
      </nav>

      <main className="container">
        <header className="hero">
          <h1>Blood Group Analysis</h1>
          <p>Upload a clear fingerprint scan or photo for instant analysis with detailed insights.</p>
        </header>

        {/* Two-column layout for upload + result */}
        <div className="main-grid">
          {/* Left column: Upload & Preview */}
          <div className="left-column">
            <section className="card fade-in">
              <h2>1. Provide Fingerprint</h2>
              <div className="dropzone" onDragOver={prevent} onDragEnter={prevent} onDrop={onDrop}>
                <div className="dz-inner">
                  <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <p className="muted">Drag & drop or click to select an image</p>
                  <input type="file" accept="image/*" multiple onChange={onSelect} />
                </div>
              </div>
              {files?.length > 1 && (
                <p className="muted">Selected {files.length} images (using up to 3)</p>
              )}
              {preview && (
                <div className="preview slide-up">
                  <img src={preview} alt="preview" />
                </div>
              )}
              {scanSamples.length > 0 && (
                <div className="samples-gallery" style={{ marginTop: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <h3 style={{ margin: 0, fontSize: '1rem' }}>Collected samples ({scanSamples.length})</h3>
                    <button className="link" onClick={onClearSamples}>Clear</button>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
                    {scanSamples.map((s, i) => (
                      <div key={i} className="thumb" style={{ position: 'relative' }}>
                        <img src={s.url} alt={`sample-${i+1}`} style={{ width: 80, height: 80, objectFit: 'cover', borderRadius: 6, border: '1px solid #ddd' }} />
                        <button
                          className="chip"
                          onClick={() => onRemoveSample(i)}
                          style={{ position: 'absolute', top: 4, right: 4 }}
                          title="Remove sample"
                        >×</button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            <section className="card fade-in delay-1">
              <h2>2. Analyze</h2>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button className="primary" onClick={onPredict} disabled={!file || loading || !ready}>
                  {loading ? (
                    <>
                      <span className="spinner-small"></span>
                      Analyzing…
                    </>
                  ) : 'Predict Blood Group'}
                </button>
                <button className="secondary" onClick={onPredictMultiUploads} disabled={!files?.length || loading || !ready}>
                  Predict from Selected (Multi)
                </button>
                
                {(
                  <>
                    <button className="secondary" onClick={onScanAddSample} disabled={scanLoading || !ready}>
                      {scanLoading ? (
                        <>
                          <span className="spinner-small"></span>
                          Scanning…
                        </>
                      ) : (
                        'Scan (Add sample)'
                      )}
                    </button>
                    <button className="secondary" onClick={onScanPreview} disabled={scanLoading || !ready}>
                      {scanLoading ? (
                        <>
                          <span className="spinner-small"></span>
                          Scanning…
                        </>
                      ) : (
                        <>
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style={{ width: '20px', height: '20px' }}>
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v1m6 11h2m-6 0h-2v4m0-11v3m0 0h.01M12 12h4.01M16 20h4M4 12h4m12 0h.01M5 8h2a1 1 0 001-1V5a1 1 0 00-1-1H5a1 1 0 00-1 1v2a1 1 0 001 1zm12 0h2a1 1 0 001-1V5a1 1 0 00-1-1h-2a1 1 0 00-1 1v2a1 1 0 001 1zM5 20h2a1 1 0 001-1v-2a1 1 0 00-1-1H5a1 1 0 00-1 1v2a1 1 0 001 1z" />
                          </svg>
                          Scan (Preview)
                        </>
                      )}
                    </button>
                    <button className="ghost" onClick={onScanAndPredict} disabled={loading || !ready}>
                      Quick Scan & Predict
                    </button>
                    <button className="ghost" onClick={onMultiScan} disabled={multiLoading || !ready}>
                      {multiLoading ? 'Multi-scan…' : 'Multi-scan x3 & Predict'}
                    </button>
                    <button className="secondary" onClick={onOpenDemo} disabled={bridgeLoading || !ready}>
                      {bridgeLoading ? 'Opening demo…' : 'Open Demo (fallback)'}
                    </button>
                    <button className="ghost" onClick={onConsumeLatest} disabled={loading || !ready}>
                      Consume Latest & Predict
                    </button>
                  </>
                )}
              </div>
              {!ready && (<p className="warn">Backend not reachable at {API_BASE}. Start it or click Check.</p>)}
              {scannerReady ? (
                <p className="success-note">✓ Scanner detected: Suprema BioMini Slim 2</p>
              ) : (
                <p className="warn">Scanner not detected. You can still use the demo fallback: open demo, capture, save into temp folder, then consume. <button className="link" onClick={checkScanner}>Check scanner</button></p>
              )}
              {error && <p className="error">{error}</p>}
            </section>
          </div>

          {/* Right column: Result */}
          {result && (
            <div className="right-column">
              <section className="card result-enter">
                <h2>Result</h2>
                <div className="result-main">
                  <div className="predicted">{result.predicted}</div>
                  <div className="confidence">{(result.confidence * 100).toFixed(1)}% confidence</div>
                </div>
                <div className="bars">
                  {bars.map(([k, v]) => {
                    const pct = (v * 100).toFixed(1)
                    const width = Math.max(2, Math.min(100, Math.round(v * 100)))
                    return (
                      <div key={k} className="barrow">
                        <div className="barlabel">{k}</div>
                        <div className="bartrack">
                          <div className={`barfill ${k === result.predicted ? 'top' : ''}`} style={{ width: `${width}%` }} />
                        </div>
                        <div className="barpct">{pct}%</div>
                      </div>
                    )
                  })}
                </div>
              </section>
              {Array.isArray(result.samples) && result.samples.length > 0 && (
                <section className="card result-enter" style={{ marginTop: '1rem' }}>
                  <h3>Per-sample predictions</h3>
                  <ul>
                    {result.samples.map((s, i) => (
                      <li key={i}>Sample {i+1}: {s.predicted} ({typeof s.confidence==='number' ? (s.confidence*100).toFixed(1)+'%' : '—'})</li>
                    ))}
                  </ul>
                </section>
              )}
            </div>
          )}
        </div>

        {/* Full-width details below */}
        {result && (
          <>
            <section className="card result-enter delay-1">
              <h2>Detailed Metrics</h2>
              {result.features && (
                <div className="metrics">
                  <div className="metric">
                    <div className="m-label">Energy</div>
                    <div className="m-value">{fmt(result.features.energy)}</div>
                  </div>
                  <div className="metric">
                    <div className="m-label">Homogeneity</div>
                    <div className="m-value">{fmt(result.features.homogeneity)}</div>
                  </div>
                  <div className="metric">
                    <div className="m-label">Contrast</div>
                    <div className="m-value">{fmt(result.features.contrast)}</div>
                  </div>
                  <div className="metric">
                    <div className="m-label">Entropy</div>
                    <div className="m-value">{fmt(result.features.entropy)}</div>
                  </div>
                  <div className="metric">
                    <div className="m-label">Ridge density</div>
                    <div className="m-value">{fmt(result.features.ridge_density)}</div>
                  </div>
                  <div className="metric">
                    <div className="m-label">Correlation</div>
                    <div className="m-value">{fmt(result.features.correlation)}</div>
                  </div>
                  <div className="metric">
                    <div className="m-label">Skeleton pixels</div>
                    <div className="m-value">{result.features.skeleton_pixels ?? '—'}</div>
                  </div>
                </div>
              )}

              <details style={{ marginTop: '0.5rem' }}>
                <summary>What these metrics mean</summary>
                <ul className="muted">
                  <li><strong>Energy:</strong> texture uniformity (higher can mean smoother ridges).</li>
                  <li><strong>Homogeneity:</strong> similarity of neighboring pixels (higher means more even ridge flow).</li>
                  <li><strong>Contrast:</strong> intensity variation (higher means stronger ridge-valley differences).</li>
                  <li><strong>Entropy:</strong> randomness in orientation histogram (lower suggests more ordered flow).</li>
                  <li><strong>Ridge density:</strong> ridges per area (varies by finger and sex; quality-gated usage).</li>
                  <li><strong>Correlation:</strong> linear dependence of gray levels; auxiliary stability cue.</li>
                  <li><strong>Skeleton pixels:</strong> number of pixels in thinned ridge skeleton (proxy for coverage).</li>
                </ul>
              </details>

              {result.corrections?.length ? (
                <div className="notes">
                  <h3>Pattern Analysis</h3>
                  <ul>
                    {result.corrections.map((n, i) => (<li key={i}>{n}</li>))}
                  </ul>
                </div>
              ) : null}

              <details>
                <summary>Raw Features</summary>
                <pre>{JSON.stringify(result.features, null, 2)}</pre>
              </details>
            </section>

            {result.extended && (
              <section className="card result-enter delay-1">
                <h2>Extended Analysis</h2>
              <div className="metrics">
                <div className="metric">
                  <div className="m-label">Pattern</div>
                  <div className="m-value">{result.extended.pattern_type?.label ?? '—'}</div>
                  <div className="muted">conf: {fmt(result.extended.pattern_type?.confidence, 2)}</div>
                </div>
                <div className="metric">
                  <div className="m-label">Sex (heuristic)</div>
                  <div className="m-value">{(result.extended.sex?.label === 'undetermined') ? '—' : (result.extended.sex?.label ?? '—')}</div>
                  <div className="muted">
                    {(() => {
                      const p = result.extended.sex?.probability
                      if (result.extended.sex?.label === 'undetermined') return 'undetermined'
                      if (typeof p !== 'number') return 'p=—'
                      const d = Math.abs(p - 0.5)
                      const conf = d < 0.07 ? 'low' : d < 0.12 ? 'medium' : 'high'
                      return `p=${fmt(p, 2)} · confidence: ${conf}`
                    })()}
                  </div>
                </div>
                <div className="metric">
                  <div className="m-label">Health Index (HI)</div>
                  <div className="m-value">{result.extended.health_index?.score ?? '—'}</div>
                  <div className="muted">{result.extended.health_index?.interpretation ?? ''}</div>
                </div>
                <div className="metric">
                  <div className="m-label">Rh Factor</div>
                  <div className="m-value">{(result.extended.rh_factor?.label === 'undetermined') ? '—' : (result.extended.rh_factor?.label ?? '—')}</div>
                  <div className="muted">
                    {(() => {
                      const p = result.extended.rh_factor?.probability
                      if (result.extended.rh_factor?.label === 'undetermined') return 'undetermined'
                      if (typeof p !== 'number') return 'p=—'
                      const d = Math.abs(p - 0.5)
                      const conf = d < 0.07 ? 'low' : d < 0.12 ? 'medium' : 'high'
                      return `p=${fmt(p, 2)} · confidence: ${conf}`
                    })()}
                  </div>
                </div>
                <div className="metric">
                  <div className="m-label">BP tendency</div>
                  <div className="m-value">{result.extended.bp_tendency?.label ?? '—'}</div>
                  <div className="muted">score={fmt(result.extended.bp_tendency?.score, 2)}</div>
                </div>
                <div className="metric">
                  <div className="m-label">Stress index</div>
                  <div className="m-value">{result.extended.stress_index?.score ?? '—'}</div>
                  <div className="muted">{result.extended.stress_index?.interpretation ?? ''}</div>
                </div>
              </div>

              <div className="explain">
                <button className="link" onClick={() => setShowExplain(s => !s)}>
                  {showExplain ? 'Hide explanations' : 'Show explanations'}
                </button>
                {showExplain && (
                  <div className="explain-body">
                    <ul>
                      <li>
                        <strong>Pattern:</strong> {result.extended.pattern_type?.basis}
                      </li>
                      <li>
                        <strong>Sex:</strong> {result.extended.sex?.basis}
                        {result.extended.sex?.citation?.length ? (
                          <div className="cite">Refs: {result.extended.sex.citation.join('; ')}</div>
                        ) : null}
                      </li>
                      <li>
                        <strong>BP tendency:</strong> {result.extended.bp_tendency?.basis}
                        {result.extended.bp_tendency?.citation?.length ? (
                          <div className="cite">Refs: {result.extended.bp_tendency.citation.join('; ')}</div>
                        ) : null}
                      </li>
                      <li>
                        <strong>Stress index:</strong> {result.extended.stress_index?.basis}
                      </li>
                      <li>
                        <strong>Rh factor:</strong> {result.extended.rh_factor?.basis}
                      </li>
                      <li>
                        <strong>Aggregation:</strong> When multiple images are used, values are combined by majority vote (labels) and averages (scores/probabilities). See bases above for vote counts and sample sizes.
                      </li>
                    </ul>
                  </div>
                )}
              </div>
              </section>
            )}

            {/* Feedback Panel - continuous learning */}
            {!feedbackSubmitted && (
              <FeedbackPanel 
                predictionResult={result} 
                onFeedbackSubmitted={handleFeedbackSubmitted}
                sampleId={result.source || 'single'}
                isMultiSample={result.samples && result.samples.length > 0}
                sampleCount={result.samples ? result.samples.length : 1}
              />
            )}

            {/* Feedback stats after submission */}
            {feedbackSubmitted && feedbackStats && !feedbackStats.skipped && (
              <section className="card result-enter delay-2" style={{ textAlign: 'center', background: '#f0fdf4', border: '2px solid #10b981' }}>
                <h3 style={{ color: '#047857' }}>🚀 Strong Learning Applied!</h3>
                <p style={{ color: '#065f46', fontSize: '0.95rem', marginBottom: '0.5rem' }}>
                  Model learned this pattern AGGRESSIVELY (85% weight) - next scan will be much better!
                </p>
                {feedbackStats.knn_enabled && (
                  <p style={{ color: '#047857', fontSize: '0.85rem', fontWeight: '500' }}>
                    💪 Aggressive Learning Active • {feedbackStats.knn_training_size} samples • {Math.round(feedbackStats.knn_weight * 100)}% influence
                  </p>
                )}
                {!feedbackStats.knn_enabled && feedbackStats.knn_training_size < 5 && (
                  <p style={{ color: '#0891b2', fontSize: '0.85rem' }}>
                    🔄 Building strength... ({feedbackStats.knn_training_size}/5 samples for full power)
                  </p>
                )}
              </section>
            )}
          </>
        )}

        <footer className="foot">
          <small>Calibration labels: {labels.join(', ') || '—'}</small>
        </footer>
      </main>

      {loading && (
        <div className="spinner-overlay">
          <div className="fingerprint-loader">
            <svg viewBox="0 0 200 200" className="fp-svg">
              {/* Fingerprint ridges - animated drawing */}
              <ellipse cx="100" cy="100" rx="70" ry="85" className="fp-ridge ridge-1" />
              <ellipse cx="100" cy="100" rx="60" ry="73" className="fp-ridge ridge-2" />
              <ellipse cx="100" cy="100" rx="50" ry="61" className="fp-ridge ridge-3" />
              <ellipse cx="100" cy="100" rx="40" ry="49" className="fp-ridge ridge-4" />
              <ellipse cx="100" cy="100" rx="30" ry="37" className="fp-ridge ridge-5" />
              <ellipse cx="100" cy="100" rx="20" ry="25" className="fp-ridge ridge-6" />
              <ellipse cx="100" cy="100" rx="10" ry="13" className="fp-ridge ridge-7" />
              <circle cx="100" cy="100" r="3" className="fp-center" />
            </svg>
            <p className="fp-text">Analyzing fingerprint...</p>
          </div>
        </div>
      )}
    </div>
  )
}
