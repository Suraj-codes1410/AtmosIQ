import React, { useState } from 'react';
import { Activity, Calendar } from 'lucide-react';

interface PollutionTimelineProps {
  onSelectDate: (date: string) => void;
}

export const PollutionTimeline: React.FC<PollutionTimelineProps> = ({ onSelectDate }) => {
  const [activeYear, setActiveYear] = useState<string>('2024');

  // Key historical benchmark episodes
  const landmarkEpisodes = [
    { date: '2024-11-16', pm25: 385.4, event: '2024 Severe Crop-Burning Episode', group: 'Biomass Burning' },
    { date: '2023-12-25', pm25: 412.1, event: '2023 Winter Stagnation Inversion', group: 'Wind Ventilation' },
    { date: '2024-01-15', pm25: 340.5, event: '2024 Thermal Inversion', group: 'Meteorology' },
    { date: '2023-11-12', pm25: 445.0, event: '2023 Mixed Diwali Festival Peak', group: 'Combined Sources' },
    { date: '2020-11-09', pm25: 420.2, event: '2020 Post-Monsoon Extreme Episode', group: 'Biomass Burning' },
    { date: '2021-11-13', pm25: 460.0, event: '2021 Winter Extreme Stagnation', group: 'Wind Ventilation' },
    { date: '2022-11-04', pm25: 395.8, event: '2022 Post-Monsoon Biomass Surge', group: 'Biomass Burning' },
  ];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-sky-400" />
            5-Year Historical Dataset v2 PM2.5 Timeline (2020–2024)
          </span>
          <h2 className="text-lg font-bold text-slate-100">
            Delhi NCR Daily Air Quality Observations (1,827 Days)
          </h2>
        </div>

        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          {['2020', '2021', '2022', '2023', '2024'].map((yr) => (
            <button
              key={yr}
              onClick={() => setActiveYear(yr)}
              className={`px-3 py-1 rounded transition font-mono ${
                activeYear === yr
                  ? 'bg-sky-600 text-white font-bold shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {yr}
            </button>
          ))}
        </div>
      </div>

      {/* Threshold Banner */}
      <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex items-center justify-between text-xs font-mono">
        <span className="text-slate-400">Extreme Pollution Threshold (Phase 4C):</span>
        <span className="text-rose-400 font-bold">306.81 µg/m³ (110 Episodes Total)</span>
      </div>

      {/* Representative Extreme Episodes for Selected Year */}
      <div className="space-y-3">
        <h3 className="text-xs uppercase tracking-wider text-slate-400 font-medium">
          Key Extreme Episodes in {activeYear}
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {landmarkEpisodes
            .filter((ep) => ep.date.startsWith(activeYear))
            .map((ep) => (
              <div
                key={ep.date}
                onClick={() => onSelectDate(ep.date)}
                className="bg-slate-950 border border-slate-800 hover:border-sky-500/50 p-3.5 rounded-lg cursor-pointer transition space-y-2 group"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono text-sky-400 font-semibold">{ep.date}</span>
                  <span className="text-[10px] bg-rose-950/80 text-rose-300 border border-rose-800/80 px-2 py-0.5 rounded font-mono">
                    {ep.pm25.toFixed(1)} µg/m³
                  </span>
                </div>
                <div className="text-xs font-medium text-slate-200 group-hover:text-sky-300 transition">
                  {ep.event}
                </div>
                <div className="text-[10px] text-slate-400 uppercase font-mono">
                  Dominant Signal: {ep.group}
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};
