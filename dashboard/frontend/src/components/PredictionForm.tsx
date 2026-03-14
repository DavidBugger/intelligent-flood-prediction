"use client";

import { useState } from "react";
import { predictFlood, PredictionInput, PredictionResponse } from "@/lib/api";
import { AlertTriangle, Droplets, Waves, Calendar, Loader2 } from "lucide-react";

export default function PredictionForm() {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<PredictionResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const [formData, setFormData] = useState<PredictionInput>({
        rainfall_mm: 80,
        rain_7day: 250,
        month: new Date().getMonth() + 1,
        sea_level_m_lagos: 7.1,
        tide_level_m: 1.4,
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        try {
            const data = await predictFlood(formData);
            setResult(data);
        } catch (err) {
            setError("Failed to connect to the prediction server. Make sure the backend is running.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFormData((prev) => ({
            ...prev,
            [name]: parseFloat(value) || 0,
        }));
    };

    return (
        <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 glass card-hover transition-all duration-300">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                    <Droplets className="text-blue-600" size={24} />
                    Manual Risk Assessment
                </h2>
                <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-slate-700">Today's Rainfall (mm)</label>
                        <input
                            type="number"
                            name="rainfall_mm"
                            value={formData.rainfall_mm}
                            onChange={handleChange}
                            className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                            required
                        />
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-slate-700">7-Day Accumulated Rain (mm)</label>
                        <input
                            type="number"
                            name="rain_7day"
                            value={formData.rain_7day}
                            onChange={handleChange}
                            className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                            required
                        />
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-slate-700">Sea Level (m)</label>
                        <input
                            type="number"
                            step="0.1"
                            name="sea_level_m_lagos"
                            value={formData.sea_level_m_lagos}
                            onChange={handleChange}
                            className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                        />
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-slate-700">Tide Level (m)</label>
                        <input
                            type="number"
                            step="0.1"
                            name="tide_level_m"
                            value={formData.tide_level_m}
                            onChange={handleChange}
                            className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                        />
                    </div>
                    <div className="md:col-span-2 space-y-2">
                        <label className="text-sm font-medium text-slate-700">Month</label>
                        <select
                            name="month"
                            value={formData.month}
                            onChange={handleChange}
                            className="w-full px-4 py-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                        >
                            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((m) => (
                                <option key={m} value={m}>
                                    {new Date(2024, m - 1).toLocaleString("default", { month: "long" })}
                                </option>
                            ))}
                        </select>
                    </div>
                    <button
                        type="submit"
                        disabled={loading}
                        className="md:col-span-2 bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                    >
                        {loading ? <Loader2 className="animate-spin" size={20} /> : <Waves size={20} />}
                        Generate Warning
                    </button>
                </form>
            </div>

            {error && (
                <div className="p-4 bg-red-50 border border-red-100 text-red-700 rounded-xl flex items-center gap-3">
                    <AlertTriangle size={20} />
                    {error}
                </div>
            )}

            {result && (
                <div
                    className="p-8 rounded-2xl shadow-lg border-2 animate-in fade-in slide-in-from-bottom-4 duration-500 glass card-hover"
                    style={{ borderColor: result.color, backgroundColor: `${result.color}10` }}
                >
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                        <div>
                            <div className="flex items-center gap-3 mb-1">
                                <span className="text-4xl">{result.emoji}</span>
                                <h3 className="text-2xl font-bold" style={{ color: result.color }}>
                                    {result.label}
                                </h3>
                            </div>
                            <p className="text-slate-600 font-medium">Flood Probability: {(result.probability * 100).toFixed(1)}%</p>
                        </div>
                        <div className="bg-white px-4 py-2 rounded-full border border-slate-100 shadow-sm">
                            <span className="font-bold text-slate-700">Status: {result.alert_level}</span>
                        </div>
                    </div>
                    <div className="mt-6 p-4 bg-white rounded-xl border border-slate-100 shadow-sm">
                        <h4 className="font-bold text-slate-900 mb-2">Recommended Actions:</h4>
                        <p className="text-slate-600">{result.action}</p>
                    </div>
                </div>
            )}
        </div>
    );
}
