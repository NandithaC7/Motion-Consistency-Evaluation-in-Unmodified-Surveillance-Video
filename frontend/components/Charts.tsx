"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  accuracyPoints,
  confusion,
  lossPoints,
  prPoints,
  rocPoints,
} from "@/lib/chart-data";

const axisTick = {
  fill: "#6B7280",
  fontSize: 12,
  fontFamily: "Inter, sans-serif",
};

const tooltipStyle = {
  border: "1px solid #DADADA",
  borderRadius: 0,
  background: "#FFFFFF",
  boxShadow: "none",
  fontSize: 12,
  fontFamily: "Inter, sans-serif",
};

function ChartShell({
  children,
  minWidth = 560,
}: {
  children: React.ReactNode;
  minWidth?: number;
}) {
  return (
    <div className="chart-scroll">
      <div className="h-[380px] md:h-[420px]" style={{ minWidth }}>
        {children}
      </div>
    </div>
  );
}

export function ConfusionMatrix() {
  const max = Math.max(...confusion.matrix.flat());

  return (
    <div className="chart-scroll">
      <div className="min-w-[520px]">
        <div className="grid grid-cols-[88px_1fr] gap-4 items-start">
          <div />
          <div className="grid grid-cols-3">
            {confusion.labels.map((label) => (
              <p
                key={label}
                className="text-center text-[12px] tracking-[0.08em] uppercase text-muted pb-3"
              >
                Pred. {label}
              </p>
            ))}
          </div>
          {confusion.matrix.map((row, r) => (
            <div key={confusion.labels[r]} className="contents">
              <p className="text-[12px] tracking-[0.08em] uppercase text-muted self-center">
                True {confusion.labels[r]}
              </p>
              <div className="grid grid-cols-3 border-l border-t border-line">
                {row.map((value, c) => {
                  const intensity = 0.12 + (value / max) * 0.88;
                  const dark = intensity > 0.55;
                  return (
                    <div
                      key={`${r}-${c}`}
                      className="aspect-square border-r border-b border-line flex items-center justify-center text-[18px] tabular-nums"
                      style={{
                        background: `rgba(46, 125, 50, ${intensity})`,
                        color: dark ? "#FFFFFF" : "#111111",
                      }}
                    >
                      {value}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-[13px] text-muted">
          Green intensity encodes count. Demonstration placeholder.
        </p>
      </div>
    </div>
  );
}

export function RocCurve() {
  return (
    <ChartShell>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rocPoints} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="#DADADA" strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            stroke="#111111"
            tick={axisTick}
            tickLine={false}
            label={{
              value: "False positive rate",
              position: "insideBottom",
              offset: -2,
              fill: "#6B7280",
              fontSize: 12,
            }}
          />
          <YAxis
            dataKey="tpr"
            type="number"
            domain={[0, 1]}
            stroke="#111111"
            tick={axisTick}
            tickLine={false}
            width={40}
          />
          <Tooltip contentStyle={tooltipStyle} />
          <Line
            dataKey="tpr"
            type="monotone"
            stroke="#2E7D32"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function PrecisionRecallCurve() {
  return (
    <ChartShell>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={prPoints} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="#DADADA" strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="recall"
            type="number"
            domain={[0, 1]}
            stroke="#111111"
            tick={axisTick}
            tickLine={false}
            label={{
              value: "Recall",
              position: "insideBottom",
              offset: -2,
              fill: "#6B7280",
              fontSize: 12,
            }}
          />
          <YAxis
            dataKey="precision"
            type="number"
            domain={[0.6, 1]}
            stroke="#111111"
            tick={axisTick}
            tickLine={false}
            width={40}
          />
          <Tooltip contentStyle={tooltipStyle} />
          <Line
            dataKey="precision"
            type="monotone"
            stroke="#2E7D32"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function TrainingLossCurve() {
  return (
    <ChartShell>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={lossPoints} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="#DADADA" strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="epoch"
            stroke="#111111"
            tick={axisTick}
            tickLine={false}
            label={{
              value: "Epoch",
              position: "insideBottom",
              offset: -2,
              fill: "#6B7280",
              fontSize: 12,
            }}
          />
          <YAxis stroke="#111111" tick={axisTick} tickLine={false} width={48} />
          <Tooltip contentStyle={tooltipStyle} />
          <Line
            dataKey="streamA"
            name="Stream A"
            type="monotone"
            stroke="#111111"
            strokeWidth={1.5}
            dot={false}
          />
          <Line
            dataKey="streamB"
            name="Stream B"
            type="monotone"
            stroke="#2E7D32"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function AccuracyCurve() {
  return (
    <ChartShell>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={accuracyPoints}
          margin={{ top: 12, right: 16, left: 0, bottom: 8 }}
        >
          <CartesianGrid stroke="#DADADA" strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="epoch"
            stroke="#111111"
            tick={axisTick}
            tickLine={false}
            label={{
              value: "Epoch",
              position: "insideBottom",
              offset: -2,
              fill: "#6B7280",
              fontSize: 12,
            }}
          />
          <YAxis
            domain={[60, 100]}
            stroke="#111111"
            tick={axisTick}
            tickLine={false}
            width={40}
          />
          <Tooltip contentStyle={tooltipStyle} />
          <Line
            dataKey="train"
            name="Train"
            type="monotone"
            stroke="#111111"
            strokeWidth={1.5}
            dot={false}
          />
          <Line
            dataKey="validation"
            name="Validation"
            type="monotone"
            stroke="#2E7D32"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}
