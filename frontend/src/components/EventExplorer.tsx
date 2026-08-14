import React, { useEffect, useState } from 'react';
import { EventCatalogItem } from '../types';
import { getEvents } from '../api/decisionSupport';
import { EventDetailModal } from './EventDetailModal';
import { Flame, Wind, Filter, Search, Calendar, ChevronRight } from 'lucide-react';

export const EventExplorer: React.FC = () => {
  const [events, setEvents] = useState<EventCatalogItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedYear, setSelectedYear] = useState<string>('All');
  const [selectedGroup, setSelectedGroup] = useState<string>('All');
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);

  useEffect(() => {
    getEvents()
      .then((data) => {
        setEvents(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const filteredEvents = events.filter((evt) => {
    const yr = evt.event_start.substring(0, 4);
    if (selectedYear !== 'All' && yr !== selectedYear) return false;
    if (selectedGroup !== 'All' && evt.dominant_attribution_group !== selectedGroup) return false;
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Title & Filters */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1.5">
            <Flame className="w-4 h-4 text-rose-400" />
            Phase 4C Extreme Pollution Event Catalog
          </span>
          <h2 className="text-lg font-bold text-slate-100">
            Historical Extreme Episodes Catalog ({filteredEvents.length} / {events.length} Episodes)
          </h2>
        </div>

        {/* Filter Controls */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1 bg-slate-950 px-3 py-1.5 rounded border border-slate-800">
            <Filter className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-slate-400">Year:</span>
            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(e.target.value)}
              className="bg-transparent text-slate-200 focus:outline-none cursor-pointer"
            >
              <option value="All" className="bg-slate-900">All Years (2020–2024)</option>
              <option value="2020" className="bg-slate-900">2020</option>
              <option value="2021" className="bg-slate-900">2021</option>
              <option value="2022" className="bg-slate-900">2022</option>
              <option value="2023" className="bg-slate-900">2023</option>
              <option value="2024" className="bg-slate-900">2024</option>
            </select>
          </div>

          <div className="flex items-center gap-1 bg-slate-950 px-3 py-1.5 rounded border border-slate-800">
            <span className="text-slate-400">Dominant Group:</span>
            <select
              value={selectedGroup}
              onChange={(e) => setSelectedGroup(e.target.value)}
              className="bg-transparent text-slate-200 focus:outline-none cursor-pointer"
            >
              <option value="All" className="bg-slate-900">All Groups</option>
              <option value="pm25_persistence" className="bg-slate-900">PM2.5 Persistence</option>
              <option value="biomass_burning" className="bg-slate-900">Biomass Burning</option>
              <option value="wind_ventilation" className="bg-slate-900">Wind Ventilation</option>
              <option value="meteorology" className="bg-slate-900">Meteorology</option>
            </select>
          </div>
        </div>
      </div>

      {/* Events Table / Grid */}
      {loading ? (
        <div className="py-16 text-center text-slate-400 text-sm">
          Loading 110 extreme pollution episodes...
        </div>
      ) : error ? (
        <div className="py-16 text-center text-rose-400 text-sm">
          Failed to load event catalog: {error}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredEvents.map((evt) => (
            <div
              key={evt.event_id}
              onClick={() => setSelectedEventId(evt.event_id)}
              className="bg-slate-900 border border-slate-800 hover:border-sky-500/50 rounded-xl p-4 cursor-pointer transition space-y-3 group shadow-md"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold bg-sky-500/10 text-sky-300 border border-sky-500/30 px-2 py-0.5 rounded">
                  {evt.event_id}
                </span>
                <span className="text-xs font-mono text-slate-400 flex items-center gap-1">
                  <Calendar className="w-3.5 h-3.5" />
                  {evt.duration_days} Days
                </span>
              </div>

              <div>
                <span className="text-xs text-slate-400 block">Peak PM2.5 on {evt.peak_date}</span>
                <span className="text-xl font-bold font-mono text-rose-400">
                  {evt.peak_pm25.toFixed(1)} µg/m³
                </span>
              </div>

              <div className="flex items-center justify-between text-xs pt-2 border-t border-slate-800/80">
                <span className="text-slate-400 font-mono text-[11px] capitalize">
                  {evt.dominant_attribution_group.replace('_', ' ')}
                </span>
                <span className="text-sky-400 group-hover:translate-x-1 transition flex items-center font-medium">
                  Details <ChevronRight className="w-3.5 h-3.5" />
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modal Detail View */}
      {selectedEventId && (
        <EventDetailModal
          eventId={selectedEventId}
          onClose={() => setSelectedEventId(null)}
        />
      )}
    </div>
  );
};
