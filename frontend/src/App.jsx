import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import HomePage from './HomePage'
import PredictPage from './PredictPage'
import TrainingStatusIndicator from './components/TrainingStatusIndicator'

export default function App() {
  return (
    <BrowserRouter>
      <TrainingStatusIndicator />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/predict" element={<PredictPage />} />
      </Routes>
    </BrowserRouter>
  )
}
