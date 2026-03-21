import { useMemo, useRef, useState } from 'react';
import { BrowserRouter, Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';
const HISTORY_KEY = 'truevoice_prediction_history_v1';
const LAST_RESULT_KEY = 'truevoice_last_result_v1';
const MAX_HISTORY_ITEMS = 20;
const SUPPORTED_EXTENSIONS = ['wav', 'mp3', 'flac', 'ogg', 'm4a'];

function toPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function readHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function writeHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function saveResult(payload) {
  localStorage.setItem(LAST_RESULT_KEY, JSON.stringify(payload));
  const current = readHistory();
  const updated = [payload, ...current].slice(0, MAX_HISTORY_ITEMS);
  writeHistory(updated);
}

function readLastResult() {
  try {
    const raw = localStorage.getItem(LAST_RESULT_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function Shell({ children }) {
  const location = useLocation();

  const links = [
    { to: '/', label: 'Home' },
    { to: '/predict', label: 'Prediction' },
    { to: '/result', label: 'Result' },
    { to: '/history', label: 'History' },
  ];

  return (
    <div className="page">
      <div className="background-decoration"></div>

      <header className="top-nav">
        <div className="brand">TrueVoice</div>
        <nav className="nav-links">
          {links.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className={`nav-link ${location.pathname === link.to ? 'active' : ''}`}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </header>

      <main className="card">{children}</main>

      <footer className="footer">
        <p>ResNet-18 • 50 epochs • 97.33% validation accuracy</p>
      </footer>
    </div>
  );
}

function HomePage() {
  return (
    <>
      <section className="header-section">
        <div className="logo-badge">TV</div>
        <p className="eyebrow">Deepfake Audio Detection</p>
        <h1>TrueVoice Review Demo</h1>
        <p className="subtext">
          Detect synthetic speech using a ResNet-18 model trained on 56,654 audio samples. Use the flow:
          predict, review output, and track history.
        </p>
      </section>

      <section className="overview-grid">
        <article className="overview-item">
          <p className="result-label">Dataset</p>
          <p className="overview-value">56,654</p>
          <p className="subtext">Real and spoofed audio clips across train/val/test splits.</p>
        </article>
        <article className="overview-item">
          <p className="result-label">Backbone</p>
          <p className="overview-value">ResNet-18</p>
          <p className="subtext">Mel-spectrogram based binary classification.</p>
        </article>
        <article className="overview-item">
          <p className="result-label">Best Validation</p>
          <p className="overview-value">97.33%</p>
          <p className="subtext">Best model checkpoint selected from epoch training run.</p>
        </article>
      </section>

      <section className="home-actions">
        <Link className="primary-btn action-btn" to="/predict">
          Start Prediction
        </Link>
        <Link className="status-btn action-btn" to="/history">
          Open Prediction History
        </Link>
      </section>
    </>
  );
}

function PredictionPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState(null);
  const audioRef = useRef(null);
  const navigate = useNavigate();
  const acceptTypes = SUPPORTED_EXTENSIONS.map((ext) => `.${ext}`).join(',');

  const fileInfo = useMemo(() => {
    if (!selectedFile) {
      return 'No file selected';
    }
    return `${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`;
  }, [selectedFile]);

  function handlePlay() {
    if (!selectedFile || !audioRef.current) return;
    const objectUrl = URL.createObjectURL(selectedFile);
    audioRef.current.src = objectUrl;
    audioRef.current.play();
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null;

    if (!file) {
      setSelectedFile(null);
      return;
    }

    const extension = file.name.includes('.') ? file.name.split('.').pop().toLowerCase() : '';
    if (!SUPPORTED_EXTENSIONS.includes(extension)) {
      setSelectedFile(null);
      setError('Unsupported file type selected. Please choose a supported audio file.');
      return;
    }

    setError('');
    setSelectedFile(file);
  }

  async function checkHealth() {
    try {
      setError('');
      const response = await fetch(`${API_BASE_URL}/api/health`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Backend health check failed');
      }
      setHealth(data);
    } catch (err) {
      setHealth(null);
      setError(err.message || 'Failed to reach backend');
    }
  }

  async function handleAnalyze(event) {
    event.preventDefault();
    if (!selectedFile) {
      setError('Please choose an audio file first.');
      return;
    }

    const formData = new FormData();
    formData.append('audio', selectedFile);

    try {
      setIsLoading(true);
      setError('');

      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      const payload = {
        id: `${Date.now()}`,
        fileName: selectedFile.name,
        fileSizeMB: Number((selectedFile.size / 1024 / 1024).toFixed(2)),
        analyzedAt: new Date().toISOString(),
        result: data,
      };

      saveResult(payload);
      navigate('/result', { state: { payload } });
    } catch (err) {
      setError(err.message || 'Prediction request failed');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <>
      <section className="header-section compact-header">
        <h1>Prediction</h1>
        <p className="subtext">Upload an audio file to run live deepfake detection.</p>
      </section>

      <div className="backend-section">
        <button type="button" className="status-btn" onClick={checkHealth}>
          {health ? 'Backend Online' : 'Check Backend'}
        </button>
        <span className={`status-indicator ${health ? 'healthy' : 'unknown'}`}>
          {health ? (
            <>
              <span className="pulse"></span>
              Running on {health.device}
            </>
          ) : (
            'Status unknown'
          )}
        </span>
      </div>

      <form onSubmit={handleAnalyze} className="upload-form">
        <div className="form-group">
          <label className="file-label" htmlFor="audioFile">
            Select Audio File
          </label>
          <div className="file-input-wrapper">
            <input
              id="audioFile"
              type="file"
              accept={acceptTypes}
              onChange={handleFileChange}
              className="file-input"
            />
            <span className="file-input-placeholder">
              {selectedFile
                ? 'File selected'
                : 'Choose a supported audio file'}
            </span>
          </div>
          <p className="file-info">{fileInfo}</p>
        </div>

        {selectedFile && (
          <div className="audio-controls">
            <button type="button" className="play-btn" onClick={handlePlay}>
              Play Preview
            </button>
            <audio ref={audioRef} className="audio-element" controls />
          </div>
        )}

        <button type="submit" className="primary-btn" disabled={isLoading}>
          {isLoading ? (
            <>
              <span className="spinner"></span> Analyzing...
            </>
          ) : (
            'Analyze Audio'
          )}
        </button>
      </form>

      {error && (
        <div className="error-box">
          <span className="error-icon">!</span>
          <p>{error}</p>
        </div>
      )}
    </>
  );
}

function ResultContent({ payload }) {
  const { result, fileName, analyzedAt, fileSizeMB } = payload;

  return (
    <section className="result-box result-page-box">
      <div className="result-header">
        <h2>Prediction Result</h2>
        <p className="subtext">
          {fileName} • {fileSizeMB} MB • {new Date(analyzedAt).toLocaleString()}
        </p>
      </div>

      <div className="result-grid">
        <div className={`result-item prediction-${result.prediction.toLowerCase()}`}>
          <p className="result-label">Classification</p>
          <p className="result-value">{result.prediction}</p>
        </div>

        <div className="result-item confidence">
          <p className="result-label">Confidence</p>
          <div className="confidence-bar">
            <div className="confidence-fill" style={{ width: `${result.confidence * 100}%` }}></div>
          </div>
          <p className="result-value">{toPercent(result.confidence)}</p>
        </div>
      </div>

      <div className="scores-section">
        <p className="scores-title">Detailed Scores:</p>
        <div className="score-items">
          <div className="score-item real">
            <span className="score-label">Real (Bona-fide)</span>
            <div className="score-bar">
              <div className="score-fill real-fill" style={{ width: `${result.scores.real * 100}%` }}></div>
            </div>
            <span className="score-value">{toPercent(result.scores.real)}</span>
          </div>

          <div className="score-item fake">
            <span className="score-label">Fake (Spoof)</span>
            <div className="score-bar">
              <div className="score-fill fake-fill" style={{ width: `${result.scores.fake * 100}%` }}></div>
            </div>
            <span className="score-value">{toPercent(result.scores.fake)}</span>
          </div>
        </div>
      </div>

      <div className="result-actions">
        <Link to="/predict" className="primary-btn action-btn">
          Predict Again
        </Link>
        <Link to="/history" className="status-btn action-btn">
          View History
        </Link>
      </div>
    </section>
  );
}

function ResultPage() {
  const location = useLocation();
  const payload = location.state?.payload || readLastResult();

  if (!payload || !payload.result) {
    return (
      <section className="header-section compact-header">
        <h1>Result</h1>
        <p className="subtext">No recent prediction found. Run a prediction first.</p>
        <div className="result-actions">
          <Link to="/predict" className="primary-btn action-btn">
            Go to Prediction
          </Link>
          <Link to="/history" className="status-btn action-btn">
            View History
          </Link>
        </div>
      </section>
    );
  }

  return (
    <>
      <section className="header-section compact-header">
        <h1>Result</h1>
      </section>
      <ResultContent payload={payload} />
    </>
  );
}

function HistoryPage() {
  const [history, setHistory] = useState(() => readHistory());

  function clearHistory() {
    writeHistory([]);
    localStorage.removeItem(LAST_RESULT_KEY);
    setHistory([]);
  }

  return (
    <>
      <section className="header-section compact-header">
        <h1>Prediction History</h1>
        <p className="subtext">Recent analyses saved locally in this browser.</p>
      </section>

      <div className="history-actions">
        <Link to="/predict" className="primary-btn action-btn">
          Predict New Audio
        </Link>
        <button type="button" className="status-btn action-btn" onClick={clearHistory}>
          Clear History
        </button>
      </div>

      {history.length === 0 ? (
        <div className="empty-history">
          <p>No predictions yet. Run your first analysis on the Prediction page.</p>
        </div>
      ) : (
        <div className="history-list">
          {history.map((item) => (
            <article key={item.id} className="history-item">
              <div>
                <p className="history-title">{item.fileName}</p>
                <p className="history-meta">{new Date(item.analyzedAt).toLocaleString()}</p>
              </div>
              <div className="history-stats">
                <span className={`history-pill ${item.result.prediction === 'Real' ? 'ok' : 'alert'}`}>
                  {item.result.prediction}
                </span>
                <span>{toPercent(item.result.confidence)}</span>
              </div>
            </article>
          ))}
        </div>
      )}
    </>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Shell>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/predict" element={<PredictionPage />} />
          <Route path="/result" element={<ResultPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Shell>
    </BrowserRouter>
  );
}
