"use client";

import { Droplets, Waves, MapPin, ShieldCheck, Activity } from "lucide-react";
import PredictionForm from "@/components/PredictionForm";
import RecentPredictions from "@/components/RecentPredictions";
import { useState } from "react";

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'predict'>('dashboard');

  return (
    <main className="min-h-screen bg-slate-50">
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-2">
              <div className="bg-blue-600 p-2 rounded-lg">
                <Waves className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900 leading-tight">AquaGuard</h1>
                <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">Flood Early Warning</p>
              </div>
            </div>

            <div className="flex bg-slate-100 p-1 rounded-xl">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${activeTab === 'dashboard' ? 'bg-white shadow-sm text-blue-600' : 'text-slate-500 hover:text-slate-700'
                  }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('predict')}
                className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${activeTab === 'predict' ? 'bg-white shadow-sm text-blue-600' : 'text-slate-500 hover:text-slate-700'
                  }`}
              >
                Risk Predictor
              </button>
            </div>

            <div className="hidden md:flex items-center gap-4 text-slate-500 text-sm">
              <div className="flex items-center gap-1">
                <MapPin size={16} />
                <span>Abuja, Nigeria</span>
              </div>
              <div className="flex items-center gap-1 text-green-600 bg-green-50 px-2 py-1 rounded-full">
                <ShieldCheck size={16} />
                <span className="font-medium">System Online</span>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <header className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-end gap-4">
            <div>
              <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">
                {activeTab === 'dashboard' ? 'Coastal Risk Overview' : ' Flood Risk Assessment'}
              </h2>
              <p className="mt-2 text-slate-500 max-w-2xl font-medium">
                Intelligent prediction system monitoring Nigerian coastal cities using satellite rainfall data
                and local tidal patterns.
              </p>
            </div>
            {activeTab === 'dashboard' && (
              <div className="flex items-center gap-2 text-sm text-slate-400 bg-slate-50 px-3 py-1 rounded-lg border border-slate-100">
                <Activity size={14} />
                <span>Last scan: Today, {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {activeTab === 'dashboard' ? <RecentPredictions /> : <PredictionForm />}
      </div>

      <footer className="mt-20 border-t border-slate-200 py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-slate-400 text-sm">
          <p>© 2026 AquaGuard Project. Designed for Nigerian Coastal Resilience.</p>
          <p className="mt-1">Coastal Early Warning System powered by AI Classifier & LSTM Networks.</p>
        </div>
      </footer>
    </main>
  );
}
