import { useMemo, useRef, useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

function toPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const audioRef = useRef(null);

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
      setResult(null);

      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      setResult(data);
    } catch (err) {
      setError(err.message || 'Prediction request failed');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="background-decoration"></div>
      
      <main className="card">
        <div className="header-section">
          <div className="logo-badge">🎙️</div>
          <p className="eyebrow">TrueVoice • Deepfake Detection</p>
          <h1>Is It Real?</h1>
          <p className="subtext">
            Advanced AI-powered deepfake detection using ResNet-18 trained on 56,654 authentic audio samples
          </p>
        </div>

        <div className="backend-section">
          <button type="button" className="status-btn" onClick={checkHealth}>
            🔌 {health ? 'Backend Online' : 'Check Backend'}
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
              📁 Select Audio File
            </label>
            <div className="file-input-wrapper">
              <input
                id="audioFile"
                type="file"
                accept=".wav,.mp3,.flac,.ogg,.m4a"
                onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
                className="file-input"
              />
              <span className="file-input-placeholder">
                {selectedFile ? '✓ File selected' : 'Choose WAV, MP3, FLAC, OGG, or M4A'}
              </span>
            </div>
            <p className="file-info">{fileInfo}</p>
          </div>

          {selectedFile && (
            <div className="audio-controls">
              <button
                type="button"
                className="play-btn"
                onClick={handlePlay}
                title="Preview selected audio"
              >
                ▶️ Play Preview
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
              '🔬 Analyze Audio'
            )}
          </button>
        </form>

        {error && (
          <div className="error-box">
            <span className="error-icon">⚠️</span>
            <p>{error}</p>
          </div>
        )}

        {result && (
          <section className="result-box">
            <div className="result-header">
              <h2>🎯 Analysis Result</h2>
            </div>
            
            <div className="result-grid">
              <div className={`result-item prediction-${result.prediction.toLowerCase()}`}>
                <p className="result-label">Classification</p>
                <p className="result-value">{result.prediction}</p>
              </div>

              <div className="result-item confidence">
                <p className="result-label">Confidence</p>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
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
                    <div 
                      className="score-fill real-fill"
                      style={{ width: `${result.scores.real * 100}%` }}
                    ></div>
                  </div>
                  <span className="score-value">{toPercent(result.scores.real)}</span>
                </div>

                <div className="score-item fake">
                  <span className="score-label">Fake (Spoof)</span>
                  <div className="score-bar">
                    <div 
                      className="score-fill fake-fill"
                      style={{ width: `${result.scores.fake * 100}%` }}
                    ></div>
                  </div>
                  <span className="score-value">{toPercent(result.scores.fake)}</span>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>ResNet-18 • 50 epochs • 97.33% validation accuracy</p>
      </footer>
    </div>
  );
}
