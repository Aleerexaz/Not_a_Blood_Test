import React, { useState, useEffect } from 'react';
import './TrainingStatusIndicator.css';

const TrainingStatusIndicator = () => {
  const [status, setStatus] = useState({
    color: 'green',
    status: 'idle',
    feedback_count: 0,
    threshold: 3,
    training_in_progress: false,
    last_train_time: null
  });

  useEffect(() => {
    // Initial fetch
    fetchStatus();

    // Poll every 3 seconds
    const interval = setInterval(fetchStatus, 3000);

    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/training/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
      }
    } catch (error) {
      console.error('Failed to fetch training status:', error);
    }
  };

  const getTooltip = () => {
    const { status: st, feedback_count, threshold, training_in_progress, last_train_time } = status;
    
    if (training_in_progress) {
      return 'Model training in progress...';
    }
    if (st === 'ready' || st === 'idle') {
      if (last_train_time) {
        return `Model ready (last trained in ${last_train_time.toFixed(1)}s)`;
      }
      return 'Model ready';
    }
    if (st === 'collecting') {
      return `Collecting feedback (${feedback_count}/${threshold})`;
    }
    if (st === 'pending') {
      return `Training queued (${feedback_count} samples)`;
    }
    return 'Model status';
  };

  const getColorClass = () => {
    return `status-dot status-${status.color}`;
  };

  return (
    <div className="training-status-indicator" title={getTooltip()}>
      <div className={getColorClass()}>
        {status.training_in_progress && <div className="pulse-ring"></div>}
      </div>
    </div>
  );
};

export default TrainingStatusIndicator;
