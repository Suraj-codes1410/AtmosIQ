import React from 'react';
import { PredictionResponse } from '../types';
import { Target, TrendingUp, AlertCircle, Clock } from 'lucide-react';

interface PredictionCardProps {
  prediction: PredictionResponse;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({ prediction }) => {
  const getCategoryColor = (cat: string) => {
    switch (cat) {
      case 'Extreme Episode':
      case 'Severe+':
        return 'bg-rose-950/80 border-rose-800 text-rose-300';
      case 'Very Poor':
        return 'bg-amber-950/80 border-amber-800 text-amber-300';
      case 'Poor':
        return 'bg-yellow-950/80 border-yellow-800 text-yellow-300';
      case 'Moderate':
        return 'bg-blue-950/80 border-blue-800 text-blue-300';
      default:
        return 'bg-emerald-950/80 border-emerald-800 text-emerald-300';
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden">
      {/* Background Subtle Gradient */}
      <div className="absolute -right-12 -top-12 w-32 h-32 bg-sky-500/5 rounded-full blur-2xl pointer-events-none" />

      <div className="flex items-center justify-between mb-4">
        <div>
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
            Daily PM2.5 Forecast
          </span>
          <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            Target Date: <span className="font-mono text-sky-400">{prediction.date}</span>
          </h2>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-xs font-semibold border ${getCategoryColor(
            prediction.pollution_category
          )}`}
        >
          {prediction.pollution_category}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 border-t border-slate-800/80 pt-4">
        {/* Observed Value */}
        <div className="bg-slate-950/60 p-3.5 rounded-lg border border-slate-800">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
            <Target className="w-3.5 h-3.5 text-emerald-400" />
            Observed Surface PM2.5
          </div>
          <div className="text-2xl font-bold text-slate-100 font-mono">
            {prediction.observed_pm25 !== null
              ? `${prediction.observed_pm25.toFixed(1)}`
              : 'N/A'}
            <span className="text-xs text-slate-400 ml-1 font-sans">µg/m³</span>
          </div>
          <p className="text-[10px] text-slate-500 mt-1">Ground monitoring benchmark</p>
        </div>

        {/* AtmosIQ Model Prediction */}
        <div className="bg-sky-950/40 p-3.5 rounded-lg border border-sky-800/50">
          <div className="flex items-center gap-1.5 text-xs text-sky-300 mb-1">
            <TrendingUp className="w-3.5 h-3.5 text-sky-400" />
            AtmosIQ Model Prediction
          </div>
          <div className="text-2xl font-bold text-sky-200 font-mono">
            {prediction.predicted_pm25.toFixed(1)}
            <span className="text-xs text-sky-400 ml-1 font-sans">µg/m³</span>
          </div>
          <p className="text-[10px] text-sky-400/80 mt-1">
            Random Forest (450 trees, lag &ge; 1d)
          </p>
        </div>

        {/* Persistence Baseline */}
        <div className="bg-slate-950/60 p-3.5 rounded-lg border border-slate-800">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
            <Clock className="w-3.5 h-3.5 text-amber-400" />
            Persistence Baseline (t-1d)
          </div>
          <div className="text-2xl font-bold text-slate-200 font-mono">
            {prediction.persistence_baseline !== null
              ? `${prediction.persistence_baseline.toFixed(1)}`
              : 'N/A'}
            <span className="text-xs text-slate-400 ml-1 font-sans">µg/m³</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-1 flex items-center gap-1 font-mono">
            <span>Error:</span>
            <span
              className={
                prediction.prediction_error && prediction.prediction_error > 0
                  ? 'text-rose-400'
                  : 'text-emerald-400'
              }
            >
              {prediction.prediction_error !== null
                ? `${prediction.prediction_error > 0 ? '+' : ''}${prediction.prediction_error.toFixed(1)} µg/m³`
                : 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
