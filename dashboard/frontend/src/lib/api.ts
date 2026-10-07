const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PredictionInput {
    rainfall_mm: number;
    rain_7day: number;
    rain_3day?: number;
    rain_14day?: number;
    rain_30day?: number;
    heavy_rain_flag?: number;
    tide_level_m?: number;
    sea_level_m_lagos?: number;
    sea_level_m_port_harcourt?: number;
    month: number;
    day_of_year?: number;
    wet_season?: number;
}

export interface PredictionResponse {
    probability: number;
    alert_level: string;
    emoji: string;
    label: string;
    action: string;
    color: string;
}

export interface HistoryItem {
    date: string;
    rainfall_mm: number;
    flood_probability: number;
    alert_level: string;
    [key: string]: any;
}

export async function predictFlood(data: PredictionInput): Promise<PredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    });

    if (!response.ok) {
        let errorMessage = `Prediction request failed with status ${response.status}`;
        try {
            const errData = await response.json();
            if (errData?.detail) {
                errorMessage = typeof errData.detail === "string" ? errData.detail : JSON.stringify(errData.detail);
            }
        } catch {
            // ignore JSON parse error
        }
        throw new Error(errorMessage);
    }

    return response.json();
}

export async function getPredictionHistory(limit: number = 30): Promise<HistoryItem[]> {
    const response = await fetch(`${API_BASE_URL}/history?limit=${limit}`);

    if (!response.ok) {
        let errorMessage = `Failed to fetch history with status ${response.status}`;
        try {
            const errData = await response.json();
            if (errData?.detail) {
                errorMessage = typeof errData.detail === "string" ? errData.detail : JSON.stringify(errData.detail);
            }
        } catch {
            // ignore JSON parse error
        }
        throw new Error(errorMessage);
    }

    return response.json();
}
