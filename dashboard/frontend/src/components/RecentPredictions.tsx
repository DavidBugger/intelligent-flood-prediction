"use client";

import { useEffect, useState } from "react";
import { getPredictionHistory, HistoryItem } from "@/lib/api";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area,
} from "recharts";
import { Clock, TrendingUp, AlertCircle, CheckCircle2 } from "lucide-react";

export default function RecentPredictions() {
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const data = await getPredictionHistory();
                setHistory(data);
            } catch (err) {
                console.error("Failed to fetch history", err);
            } finally {
                setLoading(false);
            }
        };
        fetchHistory();
    }, []);

    if (loading) {
        return (
            <div className="h-96 flex items-center justify-center bg-white rounded-2xl border border-slate-100 italic text-slate-400">
                Loading historical data...
            </div>
        );
    }

    if (history.length === 0) {
        return (
            <div className="h-96 flex items-center justify-center bg-white rounded-2xl border border-slate-100 italic text-slate-400">
                No recent prediction data available.
            </div>
        );
    }

    const latest = history[history.length - 1];

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm flex items-center gap-4 glass card-hover transition-all duration-300">
                    <div className="p-3 bg-blue-50 rounded-xl text-blue-600">
                        <TrendingUp size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-slate-500 font-medium">Avg. Probability</p>
                        <p className="text-2xl font-bold">
                            {(history.reduce((acc, curr) => acc + curr.flood_probability, 0) / history.length * 100).toFixed(1)}%
                        </p>
                    </div>
                </div>
                <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm flex items-center gap-4 glass card-hover transition-all duration-300">
                    <div className="p-3 bg-red-50 rounded-xl text-red-600">
                        <AlertCircle size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-slate-500 font-medium">Risk Alerts (30d)</p>
                        <p className="text-2xl font-bold">
                            {history.filter((h) => h.alert_level === "RED").length}
                        </p>
                    </div>
                </div>
                <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm flex items-center gap-4 glass card-hover transition-all duration-300">
                    <div className="p-3 bg-green-50 rounded-xl text-green-600">
                        <CheckCircle2 size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-slate-500 font-medium">Safe Days (30d)</p>
                        <p className="text-2xl font-bold">
                            {history.filter((h) => h.alert_level === "GREEN").length}
                        </p>
                    </div>
                </div>
            </div>

            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm glass">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                        <Clock className="text-blue-600" size={24} />
                        Risk Probability Trend
                    </h2>
                    <div className="text-sm font-medium text-slate-500">Last 30 Days</div>
                </div>
                <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={history}>
                            <defs>
                                <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                            <XAxis
                                dataKey="date"
                                hide
                            />
                            <YAxis
                                domain={[0, 1]}
                                tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                                tick={{ fontSize: 12, fill: '#64748b' }}
                                axisLine={false}
                                tickLine={false}
                            />
                            <Tooltip
                                contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                                formatter={(val: any) => [`${(parseFloat(val) * 100).toFixed(1)}%`, 'Probability']}
                            />
                            <Area
                                type="monotone"
                                dataKey="flood_probability"
                                stroke="#3b82f6"
                                strokeWidth={3}
                                fillOpacity={1}
                                fill="url(#colorProb)"
                                animationDuration={2000}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden glass">
                <div className="p-6 border-b border-slate-50">
                    <h2 className="text-xl font-semibold">Detailed Log</h2>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-left">
                        <thead className="bg-slate-50 text-slate-500 text-sm font-medium">
                            <tr>
                                <th className="px-6 py-3">Date</th>
                                <th className="px-6 py-3">Rain (mm)</th>
                                <th className="px-6 py-3">Probability</th>
                                <th className="px-6 py-3">Alert</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-50">
                            {[...history].reverse().slice(0, 10).map((item, idx) => (
                                <tr key={idx} className="hover:bg-slate-50 transition-colors">
                                    <td className="px-6 py-4 font-medium text-slate-700">{item.date}</td>
                                    <td className="px-6 py-4 text-slate-600">{item.rainfall_mm} mm</td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-2">
                                            <div className="flex-1 h-2 bg-slate-100 rounded-full max-w-[60px] overflow-hidden">
                                                <div
                                                    className="h-full bg-blue-500 rounded-full"
                                                    style={{ width: `${item.flood_probability * 100}%` }}
                                                />
                                            </div>
                                            <span className="text-sm text-slate-500 font-medium">
                                                {(item.flood_probability * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2 py-1 rounded-full text-xs font-bold ${item.alert_level === 'RED' ? 'bg-red-100 text-red-600' :
                                            item.alert_level === 'AMBER' ? 'bg-amber-100 text-amber-600' :
                                                'bg-green-100 text-green-600'
                                            }`}>
                                            {item.alert_level}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
