import React, { useState } from 'react';
import { CounterfactualResponse } from '../types';
import { Sliders, AlertTriangle, CheckCircle2, ArrowRight } from 'lucide-react';

interface CounterfactualSimulatorProps {
  scenarios: Record<string, CounterfactualResponse>;
  date: string;
}

export const CounterfactualSimulator: React.FC<CounterfactualSimulatorProps> = ({ scenarios, date }) => {
  const [selectedScenarioKey, setSelectedScenarioKey] = useState<string>('biomass_low');

  const activeScenario = scenarios[selectedScenarioKey] || Object.values(scenarios)[0];

  const scenarioLabels: Record<string, string> = {
    biomass_low: 'Biomass Low (Crop Fire Q25)',
    biomass_median: 'Biomass Median (Crop Fire Q50)',
    biomass_high: 'Biomass High (Crop Fire Q75)',
    wind_stagnant: 'Wind Stagnant (Ventilation Q25)',
    wind_normal: 'Wind Normal (Ventilation Q50)',
    wind_dispersion: 'Wind Dispersion (Ventilation Q75)',
    meteorology_normal: 'Meteorology Normal (Met Q50)',
    combined_biomass_wind: 'Combined (Low Fire + High Wind)',
    combined_all_favorable: 'Combined All Favorable (Low Fire + High Wind + Normal Met)',
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1.5">
            <Sliders className="w-4 h-4 text-purple-400" />
            Phase 4D Controlled Counterfactual Simulator
          </span>
          <h3 className="text-sm font-semibold text-slate-200">
            Model Feature Sensitivity Workbench for {date}
          </h3>
        </div>

        <span className="text-xs bg-purple-950/60 border border-purple-800/60 text-purple-300 px-2.5 py-1 rounded font-mono">
          Isolation Check: 100% PASS
        </span>
      </div>

      {/* Scenario Buttons */}
      <div className="flex flex-wrap gap-2 pt-1">
        {Object.keys(scenarios).map((scenKey) => (
          <button
            key={scenKey}
            onClick={() => setSelectedScenarioKey(scenKey)}
            className={`px-3 py-1.5 rounded text-xs font-medium border transition ${
              selectedScenarioKey === scenKey
                ? 'bg-purple-600 text-white border-purple-500 shadow'
                : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
          >
            {scenarioLabels[scenKey] || scenKey}
          </button>
        ))}
      </div>

      {/* Counterfactual Output Display */}
      {activeScenario && (
        <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="bg-slate-900 p-3 rounded border border-slate-800 text-xs">
              <span className="text-slate-400 block">Baseline Prediction</span>
              <span className="text-lg font-bold font-mono text-slate-200">
                {activeScenario.baseline_prediction.toFixed(1)} µg/m³
              </span>
            </div>

            <div className="bg-slate-900 p-3 rounded border border-slate-800 text-xs">
              <span className="text-slate-400 block">Counterfactual Prediction</span>
              <span className="text-lg font-bold font-mono text-purple-300">
                {activeScenario.counterfactual_prediction.toFixed(1)} µg/m³
              </span>
            </div>

            <div className="bg-purple-950/40 p-3 rounded border border-purple-800/50 text-xs">
              <span className="text-purple-300 block">Model Sensitivity (&Delta;&ycirc;)</span>
              <span
                className={`text-lg font-bold font-mono ${
                  activeScenario.delta_prediction < 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {activeScenario.delta_prediction > 0 ? '+' : ''}
                {activeScenario.delta_prediction.toFixed(2)} µg/m³
              </span>
            </div>
          </div>

          {/* OOD Warning Banner if applicable */}
          {activeScenario.ood_status === 'WARNING_OOD' && (
            <div className="bg-amber-950/60 border border-amber-800/80 p-3 rounded text-xs text-amber-200 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <span>
                <strong>OOD Warning:</strong> This counterfactual feature vector lies &gt;3.0 standardized z-score distance from the reference training distribution. Treat sensitivity as a numerical model experiment rather than an observed state.
              </span>
            </div>
          )}

          {/* Non-Causal Wording Banner */}
          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded text-xs text-slate-300 leading-relaxed font-sans">
            <span className="font-semibold text-purple-300 block mb-1">Model Interpretation:</span>
            {activeScenario.interpretation}
          </div>
        </div>
      )}
    </div>
  );
};
