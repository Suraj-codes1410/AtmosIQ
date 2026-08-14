import React, { useEffect, useState } from 'react';
import {
  DecisionSupportResponse,
  HealthResponse
} from './types';
import { getHealth, getDecisionSupport } from './api/decisionSupport';
import { Header } from './components/Header';
import { ScientificDisclaimer } from './components/ScientificDisclaimer';
import { PredictionCard } from './components/PredictionCard';
import { AttributionPanel } from './components/AttributionPanel';
import { FeatureImportanceChart } from './components/FeatureImportanceChart';
import { EnvironmentalEvidence } from './components/EnvironmentalEvidence';
import { CounterEvidenceAlert } from './components/CounterEvidenceAlert';
import { CounterfactualSimulator } from './components/CounterfactualSimulator';
import { ConfidenceBadge } from './components/ConfidenceBadge';
import { EventExplorer } from './components/EventExplorer';
import { PollutionTimeline } from './components/PollutionTimeline';
import { SeasonalAnalysis } from './components/SeasonalAnalysis';
import { MethodologyPage } from './components/MethodologyPage';
import { AlertCircle, RefreshCw } from 'lucide-react';

export const App: React.FC = () => {
  const [selectedDate, setSelectedDate] = useState<string>('2024-11-16');
  const [activeTab, setActiveTab] = useState<'dashboard' | 'events' | 'timeline' | 'seasonal' | 'methodology'>('dashboard');

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [dsData, setDsData] = useState<DecisionSupportResponse | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Initial Health Check
  useEffect(() => {
    getHealth()
      .then((h) => setHealth(h))
      .catch((err) => console.warn('Health check failed:', err));
  }, []);

  // Fetch Decision Support Data on Date Change
  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    getDecisionSupport(selectedDate)
      .then((data) => {
        if (isMounted) {
          setDsData(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [selectedDate]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Header
        selectedDate={selectedDate}
        onDateChange={setSelectedDate}
        health={health}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-6 space-y-6">
        {/* Main Tab Routing */}
        {activeTab === 'events' ? (
          <EventExplorer />
        ) : activeTab === 'timeline' ? (
          <PollutionTimeline
            onSelectDate={(d) => {
              setSelectedDate(d);
              setActiveTab('dashboard');
            }}
          />
        ) : activeTab === 'seasonal' ? (
          <SeasonalAnalysis />
        ) : activeTab === 'methodology' ? (
          <MethodologyPage />
        ) : (
          /* Dashboard View */
          <div className="space-y-6">
            {/* System / Data Integrity Error Banner */}
            {health && health.integrity_check !== 'PASS' && (
              <div className="bg-rose-950 border border-rose-800 p-4 rounded-xl text-rose-200 text-xs flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
                <div>
                  <strong className="block font-bold text-rose-100">Model Integrity Error:</strong>
                  The underlying SHA-256 artifact verification failed. Results should not be interpreted until resolved.
                </div>
              </div>
            )}

            {/* Error State */}
            {error && (
              <div className="bg-rose-950/60 border border-rose-800 p-4 rounded-xl text-rose-200 text-xs flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-400" />
                  <span>Failed to load data for {selectedDate}: {error}</span>
                </div>
                <button
                  onClick={() => setSelectedDate(selectedDate)}
                  className="bg-slate-900 border border-slate-700 px-3 py-1 rounded text-slate-200 hover:bg-slate-800 flex items-center gap-1"
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Retry
                </button>
              </div>
            )}

            {/* Loading Spinner */}
            {loading && !error && (
              <div className="py-24 text-center space-y-3">
                <RefreshCw className="w-8 h-8 text-sky-400 animate-spin mx-auto" />
                <p className="text-xs text-slate-400 font-mono">
                  Loading AtmosIQ TreeSHAP attributions &amp; sensitivities for {selectedDate}...
                </p>
              </div>
            )}

            {/* Loaded Dashboard View */}
            {!loading && dsData && (
              <div className="space-y-6">
                {/* 1. Prediction Card */}
                <PredictionCard prediction={dsData.prediction} />

                {/* 2. Counter-Evidence Warning Banner (Surfaced if present) */}
                {dsData.validation.has_counter_evidence && (
                  <CounterEvidenceAlert conflicts={dsData.validation.counter_evidence_conflicts} />
                )}

                {/* 3. Attribution & Feature Importance Row */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <AttributionPanel attribution={dsData.attribution} />
                  <ConfidenceBadge confidence={dsData.confidence} />
                </div>

                {/* 4. Feature Importance & Environmental Evidence */}
                <FeatureImportanceChart attribution={dsData.attribution} />
                <EnvironmentalEvidence validation={dsData.validation} />

                {/* 5. Counterfactual Simulator */}
                <CounterfactualSimulator
                  scenarios={dsData.counterfactual_scenarios}
                  date={selectedDate}
                />

                {/* 6. Scientific Limitations Disclaimer */}
                <ScientificDisclaimer />
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="bg-slate-900 border-t border-slate-800 py-4 px-4 text-center text-xs text-slate-500 font-mono">
        AtmosIQ Research Platform &bull; Delhi NCR PM2.5 Forecasting &amp; Source Attribution Engine &bull; Version 1.0.0 (Phase 4F)
      </footer>
    </div>
  );
};
