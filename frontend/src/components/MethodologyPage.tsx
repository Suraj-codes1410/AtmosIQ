import React from 'react';
import { Layers, Database, ShieldCheck, Sliders, Cpu, CheckCircle } from 'lucide-react';
import { ScientificDisclaimer } from './ScientificDisclaimer';

export const MethodologyPage: React.FC = () => {
  const steps = [
    { num: '1', title: 'Multi-Source Environmental Data Ingestion', desc: 'CPCB ground PM2.5, IMD surface meteorology, ERA5 atmospheric reanalysis, MODIS/VIIRS satellite fire hotspots (2020–2024).' },
    { num: '2', title: 'Feature Engineering & Leakage Prevention', desc: '261 raw columns engineered into 147 prediction-safe features (lags 1–14d, rolling windows 3–14d, atmospheric ventilation indices).' },
    { num: '3', title: 'Optuna Model Selection & Freezing', desc: 'Evaluated Baselines, Linear, LightGBM, XGBoost, and Random Forest. Frozen Random Forest Regressor (450 trees, max depth 9).' },
    { num: '4', title: 'Phase 4B TreeSHAP Feature Attribution', desc: 'TreeSHAP exact feature & group attribution (<10⁻¹² µg/m³ additivity error). Grouped into 5 environmental categories.' },
    { num: '5', title: 'Phase 4C Environmental Validation Framework', desc: 'Validated SHAP against independent satellite fire counts & surface wind speeds across 110 cataloged extreme episodes.' },
    { num: '6', title: 'Phase 4D Controlled Counterfactual Engine', desc: 'Simulates model sensitivity (Δŷ) under controlled feature interventions (biomass_low, wind_dispersion, combined).' },
    { num: '7', title: 'Phase 4E Decision Support API & Pipeline', desc: 'RESTful API serving unified predictions, attributions, environmental evidence, counter-evidence, and confidence ratings.' },
    { num: '8', title: 'Phase 4F Interactive Dashboard & Workbench', desc: 'Research-grade presentation layer bringing transparent decision support to policymakers and atmospheric researchers.' },
  ];

  return (
    <div className="space-y-6">
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-2">
        <span className="text-xs uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1.5">
          <Layers className="w-4 h-4 text-sky-400" />
          AtmosIQ System Architecture &amp; Methodology
        </span>
        <h2 className="text-xl font-bold text-slate-100">
          End-to-End Atmospheric Forecasting &amp; Source Attribution Pipeline
        </h2>
        <p className="text-xs text-slate-400 leading-relaxed max-w-3xl">
          AtmosIQ is an advanced machine-learning system for next-day PM2.5 forecasting and environmental attribution in Delhi NCR. Below is the complete pipeline provenance from raw data ingestion to user interaction.
        </p>
      </div>

      {/* Methodology Pipeline Flow */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {steps.map((st) => (
          <div
            key={st.num}
            className="bg-slate-900 border border-slate-800 p-4 rounded-xl space-y-2 relative overflow-hidden"
          >
            <span className="text-xs font-mono font-bold bg-sky-500/20 text-sky-300 border border-sky-500/30 px-2 py-0.5 rounded">
              Step {st.num}
            </span>
            <h3 className="text-xs font-bold text-slate-200">{st.title}</h3>
            <p className="text-[11px] text-slate-400 leading-relaxed font-sans">{st.desc}</p>
          </div>
        ))}
      </div>

      {/* Key Provenance Specifications */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-3">
        <h3 className="text-sm font-bold text-slate-200">
          Immutable Model &amp; Dataset Provenance Specifications
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
          <div className="bg-slate-950 p-3 rounded border border-slate-800 space-y-1">
            <span className="text-slate-400 block text-[10px]">Dataset v2 Hash</span>
            <span className="text-sky-300 font-bold break-all">
              e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301
            </span>
          </div>

          <div className="bg-slate-950 p-3 rounded border border-slate-800 space-y-1">
            <span className="text-slate-400 block text-[10px]">Frozen Model Hash</span>
            <span className="text-purple-300 font-bold break-all">
              55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162
            </span>
          </div>

          <div className="bg-slate-950 p-3 rounded border border-slate-800 space-y-1">
            <span className="text-slate-400 block text-[10px]">Model Configuration</span>
            <span className="text-slate-200 font-bold">
              RandomForest (450 trees, max_depth=9)
            </span>
          </div>
        </div>
      </div>

      <ScientificDisclaimer />
    </div>
  );
};
