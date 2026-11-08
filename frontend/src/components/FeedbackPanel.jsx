import { useState } from 'react';
import './FeedbackPanel.css';

const BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'];

export default function FeedbackPanel({ 
  predictionResult, 
  onFeedbackSubmitted,
  sampleId = null,
  isMultiSample = false,
  sampleCount = 1
}) {
  const [feedbackState, setFeedbackState] = useState('asking'); // 'asking', 'correcting', 'submitted', 'error', 'skipped'
  const [selectedCorrection, setSelectedCorrection] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  if (!predictionResult || !predictionResult.feedback_enabled) {
    return null;
  }

  const handleCorrectClick = async () => {
    // User says prediction is correct
    await submitFeedback(true, predictionResult.predicted);
  };

  const handleIncorrectClick = () => {
    // Show correction form
    setFeedbackState('correcting');
  };

  const handleSkipClick = () => {
    // User chooses to skip feedback
    setFeedbackState('skipped');
    if (onFeedbackSubmitted) {
      onFeedbackSubmitted({ skipped: true });
    }
  };

  const handleCorrectionSubmit = async (e) => {
    e.preventDefault();
    if (!selectedCorrection) {
      setErrorMessage('Please select the correct blood group');
      return;
    }
    await submitFeedback(false, selectedCorrection);
  };

  const submitFeedback = async (isCorrect, trueLabel) => {
    setIsSubmitting(true);
    setErrorMessage('');

    try {
      const apiBase = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';
      
      // For multi-sample, submit feedback for aggregate result
      const feedbackData = {
        features: predictionResult.features,
        predicted_label: predictionResult.predicted,
        true_label: trueLabel,
        is_correct: isCorrect,
        timestamp: new Date().toISOString(),
        sample_id: sampleId,
        sample_count: isMultiSample ? sampleCount : 1
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
      
      setFeedbackState('submitted');
      
      if (onFeedbackSubmitted) {
        onFeedbackSubmitted(result);
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      setErrorMessage(error.message || 'Failed to submit feedback. Please try again.');
      setFeedbackState('error');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (feedbackState === 'submitted') {
    return (
      <div className="feedback-panel feedback-success">
        <div className="feedback-icon">✓</div>
        <h3>Thank You!</h3>
        <p>
          {isMultiSample 
            ? `Your feedback for ${sampleCount} samples helps improve the model's accuracy.`
            : 'Your feedback helps improve the model\'s accuracy.'
          }
        </p>
      </div>
    );
  }

  if (feedbackState === 'skipped') {
    return (
      <div className="feedback-panel feedback-skipped">
        <div className="feedback-icon">⊘</div>
        <h3>Feedback Skipped</h3>
        <p>You can provide feedback on the next prediction.</p>
      </div>
    );
  }

  if (feedbackState === 'error') {
    return (
      <div className="feedback-panel feedback-error">
        <div className="feedback-icon">⚠</div>
        <h3>Submission Failed</h3>
        <p>{errorMessage}</p>
        <button 
          onClick={() => setFeedbackState('asking')} 
          className="btn-retry"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (feedbackState === 'correcting') {
    return (
      <div className="feedback-panel feedback-correcting">
        <h3>What is the correct blood group?</h3>
        {isMultiSample && (
          <p className="muted">This will be applied to all {sampleCount} samples</p>
        )}
        <form onSubmit={handleCorrectionSubmit}>
          <div className="correction-grid">
            {BLOOD_GROUPS.map(group => (
              <label key={group} className="correction-option">
                <input
                  type="radio"
                  name="correction"
                  value={group}
                  checked={selectedCorrection === group}
                  onChange={(e) => setSelectedCorrection(e.target.value)}
                  disabled={isSubmitting}
                />
                <span className={`option-label ${selectedCorrection === group ? 'selected' : ''}`}>
                  {group}
                </span>
              </label>
            ))}
          </div>
          
          {errorMessage && (
            <div className="error-message">{errorMessage}</div>
          )}

          <div className="correction-actions">
            <button 
              type="button"
              onClick={() => setFeedbackState('asking')}
              className="btn-secondary"
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button 
              type="submit"
              className="btn-primary"
              disabled={isSubmitting || !selectedCorrection}
            >
              {isSubmitting ? 'Submitting...' : 'Submit Correction'}
            </button>
          </div>
        </form>
      </div>
    );
  }

  // Default: asking state
  return (
    <div className="feedback-panel feedback-asking">
      <h3>Was this prediction correct?</h3>
      {isMultiSample && (
        <p className="muted">Based on {sampleCount} samples analyzed</p>
      )}
      <p className="prediction-summary">
        We predicted: <strong>{predictionResult.predicted}</strong>
        {predictionResult.extended?.rh_factor?.label && 
          ` (Rh: ${predictionResult.extended.rh_factor.label})`
        }
      </p>
      <div className="feedback-buttons">
        <button 
          onClick={handleCorrectClick}
          className="btn-correct"
          disabled={isSubmitting}
        >
          ✓ Yes, Correct
        </button>
        <button 
          onClick={handleIncorrectClick}
          className="btn-incorrect"
          disabled={isSubmitting}
        >
          ✗ No, Incorrect
        </button>
        <button 
          onClick={handleSkipClick}
          className="btn-skip"
          disabled={isSubmitting}
        >
          Skip
        </button>
      </div>
    </div>
  );
}
