import React from 'react';
import { Activity, Calendar, ShieldCheck, Database, Layers } from 'lucide-react';
import { HealthResponse } from '../types';

interface HeaderProps {
  selectedDate: string;
  onDateChange: (date: string) => void;
  health: HealthResponse | null;
  activeTab: 'dashboard' | 'events' | 'timeline' | 'seasonal' | 'methodology';
  onTabChange: (tab: 'dashboard' | 'events' | 'timeline' | 'seasonal' | 'methodology') => void;
}

export const Header: React.FC<HeaderProps> = ({
  selectedDate,
  onDateChange,
  health,
  activeTab,
  onTabChange,
}) => {
  const isHealthy = health?.status === 'healthy' && health?.integrity_check === 'PASS';

  return (
    <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 py-3 flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Title */}
        <div className="flex items-center gap-3">
          <div className="bg-sky-500/10 border border-sky-500/30 p-2 rounded-lg text-sky-400">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-lg text-slate-100 flex items-center gap-2">
              AtmosIQ <span className="text-xs bg-sky-500/20 text-sky-300 border border-sky-500/30 px-2 py-0.5 rounded font-mono">Phase 4F</span>
            </h1>
            <p className="text-xs text-slate-400">
              Delhi NCR PM2.5 Forecasting &amp; Source Attribution Engine
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => onTabChange('dashboard')}
            className={`px-3 py-1.5 rounded transition ${
              activeTab === 'dashboard'
                ? 'bg-sky-600 text-white font-medium shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Dashboard
          </button>
          <button
            onClick={() => onTabChange('events')}
            className={`px-3 py-1.5 rounded transition ${
              activeTab === 'events'
                ? 'bg-sky-600 text-white font-medium shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Extreme Episodes (110)
          </button>
          <button
            onClick={() => onTabChange('timeline')}
            className={`px-3 py-1.5 rounded transition ${
              activeTab === 'timeline'
                ? 'bg-sky-600 text-white font-medium shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            2020–2024 Timeline
          </button>
          <button
            onClick={() => onTabChange('seasonal')}
            className={`px-3 py-1.5 rounded transition ${
              activeTab === 'seasonal'
                ? 'bg-sky-600 text-white font-medium shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Seasonal Regimes
          </button>
          <button
            onClick={() => onTabChange('methodology')}
            className={`px-3 py-1.5 rounded transition ${
              activeTab === 'methodology'
                ? 'bg-sky-600 text-white font-medium shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Methodology
          </button>
        </nav>

        {/* Date Picker & System Health Badge */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-xs">
            <Calendar className="w-4 h-4 text-sky-400" />
            <input
              type="date"
              value={selectedDate}
              min="2020-01-01"
              max="2024-12-31"
              onChange={(e) => onDateChange(e.target.value)}
              className="bg-transparent text-slate-200 focus:outline-none cursor-pointer font-mono"
            />
          </div>

          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs border font-medium ${
              isHealthy
                ? 'bg-emerald-950/60 border-emerald-800/60 text-emerald-300'
                : 'bg-rose-950/60 border-rose-800/60 text-rose-300'
            }`}
            title="System Operational & Artifact Integrity Check"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'
              }`}
            />
            {isHealthy ? 'Operational' : 'System Degraded'}
          </div>
        </div>
      </div>
    </header>
  );
};
