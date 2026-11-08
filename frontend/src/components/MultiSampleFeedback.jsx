import { useState } from 'react';
import './MultiSampleFeedback.css';

const BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'];

export default function MultiSampleFeedback({ 
  samples, 
  onFeedbackSubmitted 
}) {
  const [feedbackStates, setFeedbackStates] = useState(
    samples.map(() => ({ state: 'asking', correction: '' }))
  );
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [allSubmitted, setAllSubmitted] = useState(false);

  if (!samples || samples.length === 0) {
    return null;
  }

  const handleCorrectClick = (index) => {
    const newStates = [...feedbackStates];
    newStates[index] = { ...newStates[index], state: 'correct' };
    setFeedbackStates(newStates);
    submitFeedback(index, true, samples[index].predicted);
  };

  const handleIncorrectClick = (index) => {
    const newStates = [...feedbackStates];
    newStates[index] = { ...newStates[index], state: 'correcting' };
    setFeedbackStates(newStates);
  };

  const handleCorrectionChange = (index, value) => {
    const newStates = [...feedbackStates];
    newStates[index] = { ...newStates[index], correction: value };
    setFeedbackStates(newStates);
  };

  const handleCorrectionSubmit = (index, e) => {
    e.preventDefault();
    const correction = feedbackStates[index].correction;
    if (!correction) {
      setErrorMessage('Please select the correct blood group');
      return;
    }
    submitFeedback(index, false, correction);
  };

  const submitFeedback = async (index, isCorrect, trueLabel) => {
    setIsSubmitting(true);
    setErrorMessage('');

    try {
      const apiBase = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';
      
      const sample = samples[index];
      const feedbackData = {
        features: sample.features,
        predicted_label: sample.predicted,
        true_label: trueLabel,
        is_correct: isCorrect,
        timestamp: new Date().toISOString(),
        sample_id: `multi_${Date.now()}_${index}`
      };

      const response = await fetch(`${apiBase}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData)
      });

      if (!response.ok) {
        throw new Error(`Failed to submit feedback: ${response.statusText}`);
      }

      const result = await response.json();
      
      const newStates = [...feedbackStates];
      newStates[index] = { ...newStates[index], state: 'submitted' };
      setFeedbackStates(newStates);

      // Check if all samples have been processed
      const allDone = newStates.every(s => s.state === 'submitted' || s.state === 'correct');
      if (allDone) {
        setAllSubmitted(true);
        if (onFeedbackSubmitted) {
          onFeedbackSubmitted(result);
        }
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      setErrorMessage(error.message || 'Failed to submit feedback. Please try again.');
      
      const newStates = [...feedbackStates];
      newStates[index] = { ...newStates[index], state: 'error' };
      setFeedbackStates(newStates);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (allSubmitted) {
    return (
      <div className="multi-feedback-panel">
        <div className="feedback-success">
          <div className="feedback-icon">✓</div>
          <h3>All Feedback Submitted!</h3>
          <p>Thank you for helping improve the model with feedback on all samples.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="multi-feedback-panel">
      <h3>Provide Feedback for Each Sample</h3>
      <p className="muted">Help improve the model by confirming or correcting each sample prediction.</p>
      
      {errorMessage && (
        <div className="error-banner">{errorMessage}</div>
      )}

      <div className="samples-feedback-grid">
        {samples.map((sample, index) => {
          const state = feedbackStates[index];
          
          if (state.state === 'submitted' || state.state === 'correct') {
            return (
              <div key={index} className="sample-card submitted">
                <div className="sample-header">
                  <span className="sample-number">Sample {index + 1}</span>
                  <span className="status-badge success">✓ Submitted</span>
                </div>
                <div className="sample-prediction">{sample.predicted}</div>
              </div>
            );
          }

          if (state.state === 'error') {
            return (
              <div key={index} className="sample-card error">
                <div className="sample-header">
                  <span className="sample-number">Sample {index + 1}</span>
                  <span className="status-badge error">⚠ Error</span>
                </div>
                <div className="sample-prediction">{sample.predicted}</div>
                <button 
                  className="btn-retry-small"
                  onClick={() => {
                    const newStates = [...feedbackStates];
                    newStates[index] = { state: 'asking', correction: '' };
                    setFeedbackStates(newStates);
                  }}
                >
                  Try Again
                </button>
              </div>
            );
          }

          if (state.state === 'correcting') {
            return (
              <div key={index} className="sample-card correcting">
                <div className="sample-header">
                  <span className="sample-number">Sample {index + 1}</span>
                  <span className="status-badge">Correcting</span>
                </div>
                <div className="sample-prediction-small">Predicted: {sample.predicted}</div>
                
                <form onSubmit={(e) => handleCorrectionSubmit(index, e)}>
                  <div className="correction-grid-compact">
                    {BLOOD_GROUPS.map(group => (
                      <label key={group} className="correction-option-compact">
                        <input
                          type="radio"
                          name={`correction-${index}`}
                          value={group}
                          checked={state.correction === group}
                          onChange={(e) => handleCorrectionChange(index, e.target.value)}
                          disabled={isSubmitting}
                        />
                        <span className={`option-label-compact ${state.correction === group ? 'selected' : ''}`}>
                          {group}
                        </span>
                      </label>
                    ))}
                  </div>
                  
                  <div className="sample-actions">
                    <button 
                      type="button"
                      onClick={() => {
                        const newStates = [...feedbackStates];
                        newStates[index] = { state: 'asking', correction: '' };
                        setFeedbackStates(newStates);
                      }}
                      className="btn-cancel-small"
                      disabled={isSubmitting}
                    >
                      Cancel
                    </button>
                    <button 
                      type="submit"
                      className="btn-submit-small"
                      disabled={isSubmitting || !state.correction}
                    >
                      {isSubmitting ? 'Submitting...' : 'Submit'}
                    </button>
                  </div>
                </form>
              </div>
            );
          }

          // Default: asking state
          return (
            <div key={index} className="sample-card asking">
              <div className="sample-header">
                <span className="sample-number">Sample {index + 1}</span>
              </div>
              <div className="sample-prediction">{sample.predicted}</div>
              <div className="sample-confidence">
                {typeof sample.confidence === 'number' 
                  ? `${(sample.confidence * 100).toFixed(1)}% confidence`
                  : '—'
                }
              </div>
              <div className="sample-buttons">
                <button 
                  onClick={() => handleCorrectClick(index)}
                  className="btn-correct-small"
                  disabled={isSubmitting}
                >
                  ✓ Correct
                </button>
                <button 
                  onClick={() => handleIncorrectClick(index)}
                  className="btn-incorrect-small"
                  disabled={isSubmitting}
                >
                  ✗ Incorrect
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
