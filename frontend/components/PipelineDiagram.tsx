import { FULL_PIPELINE } from "@/lib/site";

export function PipelineDiagram() {
  const width = 1280;
  const nodeW = 132;
  const gap = 24;
  const startX = 20;
  const y = 70;

  return (
    <div className="overflow-x-auto border border-line bg-bg">
      <svg
        viewBox={`0 0 ${width} 200`}
        className="w-full min-w-[1100px] h-auto"
        role="img"
        aria-label="Full evaluation pipeline"
      >
        {FULL_PIPELINE.map((label, i) => {
          const x = startX + i * (nodeW + gap);
          const accent = i === 0 || i === FULL_PIPELINE.length - 1 || label === "Temporal Model";
          return (
            <g key={label}>
              {i < FULL_PIPELINE.length - 1 ? (
                <line
                  x1={x + nodeW}
                  y1={y + 28}
                  x2={x + nodeW + gap}
                  y2={y + 28}
                  stroke={accent ? "#2E7D32" : "#111111"}
                  strokeWidth="1"
                />
              ) : null}
              <rect
                x={x}
                y={y}
                width={nodeW}
                height={56}
                fill="#FFFFFF"
                stroke={accent ? "#2E7D32" : "#111111"}
                strokeWidth="1"
              />
              <text
                x={x + nodeW / 2}
                y={y + 34}
                textAnchor="middle"
                fontSize="12"
                fontFamily="Inter, sans-serif"
                fill="#111111"
              >
                {label}
              </text>
              <text
                x={x + nodeW / 2}
                y={y + 84}
                textAnchor="middle"
                fontSize="11"
                fontFamily="Inter, sans-serif"
                fill="#6B7280"
              >
                {String(i + 1).padStart(2, "0")}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
