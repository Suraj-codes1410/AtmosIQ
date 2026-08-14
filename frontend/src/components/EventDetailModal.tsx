import React, { useEffect, useState } from 'react';
import { EventResponse } from '../types';
import { getEventDetail } from '../api/decisionSupport';
import { X, Flame, Wind, Layers, Calendar, AlertTriangle, ShieldCheck } from 'lucide-react';

interface EventDetailModalProps {
  eventId: string;
  onClose: () => void;
}

export const EventDetailModal: React.FC<EventDetailModalProps> = ({ eventId, onClose }) => {
  const [eventDetail, setEventDetail] = useState<EventResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    getEventDetail(eventId)
      .then((data) => {
        if (isMounted) {
          setEventDetail(data);
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
  }, [eventId]);

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto p-6 space-y-4 relative shadow-2xl">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-slate-200 bg-slate-800 p-1.5 rounded-full"
        >
          <X className="w-4 h-4" />
        </button>

        {loading ? (
          <div className="py-12 text-center text-slate-400 text-sm">
            Loading episode metadata for {eventId}...
          </div>
        ) : error ? (
          <div className="py-12 text-center text-rose-400 text-sm">
            Failed to load episode details: {error}
          </div>
        ) : eventDetail ? (
          <div className="space-y-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono bg-sky-500/20 text-sky-300 border border-sky-500/30 px-2.5 py-0.5 rounded font-semibold">
                  {eventDetail.event_id}
                </span>
                <span className="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-medium">
                  {eventDetail.seasonal_regime} Regime
                </span>
              </div>
              <h2 className="text-lg font-bold text-slate-100 mt-1">
                Extreme Pollution Episode ({eventDetail.start_date} &rarr; {eventDetail.end_date})
              </h2>
            </div>

            {/* Metrics Row */}
            <div className="grid grid-cols-3 gap-3 bg-slate-950 p-4 rounded-lg border border-slate-800 text-xs">
              <div>
                <span className="text-slate-400 block">Peak PM2.5 ({eventDetail.peak_date})</span>
                <span className="text-xl font-bold font-mono text-rose-400">
                  {eventDetail.peak_pm25.toFixed(1)} µg/m³
                </span>
              </div>
              <div>
                <span className="text-slate-400 block">Episode Duration</span>
                <span className="text-xl font-bold font-mono text-slate-200">
                  {eventDetail.duration_days} Days
                </span>
              </div>
              <div>
                <span className="text-slate-400 block">Dominant Group</span>
                <span className="text-sm font-bold text-sky-400 font-mono capitalize">
                  {eventDetail.dominant_group.replace('_', ' ')}
                </span>
              </div>
            </div>

            {/* Counterfactual Sensitivity */}
            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-2 text-xs">
              <span className="font-semibold text-purple-300 uppercase tracking-wider text-[11px] block">
                Episode Counterfactual Model Sensitivity (&Delta;&ycirc;)
              </span>

              <div className="grid grid-cols-3 gap-2 font-mono">
                <div className="bg-slate-900 p-2.5 rounded border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Biomass Low</span>
                  <span className="text-emerald-400 font-bold">
                    {eventDetail.biomass_cf_delta.toFixed(1)} µg/m³
                  </span>
                </div>

                <div className="bg-slate-900 p-2.5 rounded border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Wind Dispersion</span>
                  <span className="text-emerald-400 font-bold">
                    {eventDetail.wind_cf_delta.toFixed(1)} µg/m³
                  </span>
                </div>

                <div className="bg-purple-950/40 p-2.5 rounded border border-purple-800/50">
                  <span className="text-purple-300 block text-[10px]">Combined Favorable</span>
                  <span className="text-emerald-400 font-bold">
                    {eventDetail.combined_cf_delta.toFixed(1)} µg/m³
                  </span>
                </div>
              </div>
            </div>

            {/* Disclaimer */}
            <p className="text-[11px] text-slate-400 bg-slate-950 p-3 rounded border border-slate-800 leading-relaxed font-sans">
              <strong>Non-Causal Note:</strong> Counterfactual deltas reflect internal model response under simulated feature states. They do not constitute a physical emission reduction estimate.
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
};
