import React from 'react';
import { useNavigate } from 'react-router-dom';
import './homepage.css';

function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="homepage">
      <div className="hero-container">
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">
              <span className="gradient-text">Blood Group</span>
              <br />
              Intelligence System
            </h1>
            <p className="hero-subtitle">
              Advanced fingerprint analysis powered by calibration-aware algorithms
            </p>
            <div className="feature-highlights">
              <div className="feature-item">
                <div className="feature-icon">🔬</div>
                <div className="feature-content">
                  <h3>Instant Analysis</h3>
                  <p>ABO prediction with explainable results in seconds</p>
                </div>
              </div>
              <div className="feature-item">
                <div className="feature-icon">🧬</div>
                <div className="feature-content">
                  <h3>Extended Insights</h3>
                  <p>Pattern type, Rh factor, health metrics & more</p>
                </div>
              </div>
              <div className="feature-item">
                <div className="feature-icon">✨</div>
                <div className="feature-content">
                  <h3>Transparent & Explainable</h3>
                  <p>Every decision backed by clear reasoning</p>
                </div>
              </div>
            </div>
            <button 
              className="cta-button"
              onClick={() => navigate('/predict')}
            >
              <span>Start Predicting</span>
              <svg className="arrow-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </button>
            <p className="disclaimer">
              <strong>Note:</strong> This is a research demonstration tool, not a medical device.
            </p>
          </div>
          <div className="hero-image">
            <div className="fingerprint-visual">
              <svg viewBox="0 0 200 200" className="fingerprint-svg">
                <defs>
                  <linearGradient id="gradientBlue" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style={{ stopColor: '#0066cc', stopOpacity: 1 }} />
                    <stop offset="100%" style={{ stopColor: '#00ccff', stopOpacity: 1 }} />
                  </linearGradient>
                </defs>
                {/* Fingerprint ridges - decorative */}
                <ellipse cx="100" cy="100" rx="70" ry="85" fill="none" stroke="url(#gradientBlue)" strokeWidth="2" opacity="0.3" />
                <ellipse cx="100" cy="100" rx="60" ry="73" fill="none" stroke="url(#gradientBlue)" strokeWidth="2" opacity="0.4" />
                <ellipse cx="100" cy="100" rx="50" ry="61" fill="none" stroke="url(#gradientBlue)" strokeWidth="2" opacity="0.5" />
                <ellipse cx="100" cy="100" rx="40" ry="49" fill="none" stroke="url(#gradientBlue)" strokeWidth="2" opacity="0.6" />
                <ellipse cx="100" cy="100" rx="30" ry="37" fill="none" stroke="url(#gradientBlue)" strokeWidth="2" opacity="0.7" />
                <ellipse cx="100" cy="100" rx="20" ry="25" fill="none" stroke="url(#gradientBlue)" strokeWidth="2" opacity="0.8" />
                <ellipse cx="100" cy="100" rx="10" ry="13" fill="none" stroke="url(#gradientBlue)" strokeWidth="2" opacity="0.9" />
                <circle cx="100" cy="100" r="3" fill="url(#gradientBlue)" />
              </svg>
              <div className="pulse-ring"></div>
              <div className="pulse-ring delay-1"></div>
              <div className="pulse-ring delay-2"></div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="info-section">
        <div className="info-container">
          <h2 className="section-title">How It Works</h2>
          <div className="workflow-steps">
            <div className="workflow-step">
              <div className="step-number">1</div>
              <h3>Upload Image</h3>
              <p>Provide a clear fingerprint scan or photo</p>
            </div>
            <div className="workflow-arrow">→</div>
            <div className="workflow-step">
              <div className="step-number">2</div>
              <h3>Feature Extraction</h3>
              <p>Analyze ridge patterns, density & orientation</p>
            </div>
            <div className="workflow-arrow">→</div>
            <div className="workflow-step">
              <div className="step-number">3</div>
              <h3>Calibrated Prediction</h3>
              <p>Compare against reference database</p>
            </div>
            <div className="workflow-arrow">→</div>
            <div className="workflow-step">
              <div className="step-number">4</div>
              <h3>Detailed Results</h3>
              <p>View predictions with confidence & reasoning</p>
            </div>
          </div>
        </div>
      </div>

      <footer className="homepage-footer">
        <p>Fingerprint-Based ABO Screening System • Research & Educational Use Only</p>
      </footer>
    </div>
  );
}

export default HomePage;
