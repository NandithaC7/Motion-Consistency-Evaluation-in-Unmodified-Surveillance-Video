import { FadeIn } from "@/components/FadeIn";

const NODES = [
  { id: "in", label: "Input Video", x: 40, y: 110, w: 120, accent: false },
  { id: "g", label: "Guard G1 / G2", x: 200, y: 110, w: 130, accent: true },
  { id: "pp", label: "Preprocess", x: 370, y: 110, w: 120, accent: false },
  { id: "a", label: "Stream A · ΔF", x: 540, y: 48, w: 140, accent: false },
  { id: "b", label: "Stream B · Flow", x: 540, y: 172, w: 140, accent: false },
  { id: "ae", label: "Dual LSTM-AE", x: 730, y: 110, w: 140, accent: true },
  { id: "sc", label: "Divergence Score", x: 910, y: 110, w: 150, accent: false },
  { id: "th", label: "Adaptive θ", x: 1100, y: 110, w: 120, accent: false },
  { id: "out", label: "R / Class", x: 1260, y: 110, w: 100, accent: true },
];

export function Architecture() {
  return (
    <section className="border-b border-line">
      <div className="shell py-20 md:py-28">
        <FadeIn>
          <p className="type-label mb-4">Architecture preview</p>
          <h2 className="type-section max-w-2xl">
            Dual motion streams, one reliability decision.
          </h2>
          <p className="mt-5 max-w-2xl text-[16px] text-muted">
            Frozen and near-static feeds exit at the guard layer. Remaining
            clips are reconstructed independently on frame difference and
            optical flow, then scored by their disagreement.
          </p>
        </FadeIn>
      </div>

      <div className="border-t border-line overflow-x-auto">
        <div className="min-w-[1400px] px-8 py-16">
          <svg
            viewBox="0 0 1400 280"
            className="w-full h-auto"
            role="img"
            aria-label="System architecture from input video to reliability classification"
          >
            <line
              x1="160"
              y1="140"
              x2="200"
              y2="140"
              stroke="#111111"
              strokeWidth="1"
            />
            <line
              x1="330"
              y1="140"
              x2="370"
              y2="140"
              stroke="#111111"
              strokeWidth="1"
            />
            <line
              x1="490"
              y1="140"
              x2="540"
              y2="78"
              stroke="#111111"
              strokeWidth="1"
            />
            <line
              x1="490"
              y1="140"
              x2="540"
              y2="202"
              stroke="#111111"
              strokeWidth="1"
            />
            <line
              x1="680"
              y1="78"
              x2="730"
              y2="140"
              stroke="#2E7D32"
              strokeWidth="1.25"
            />
            <line
              x1="680"
              y1="202"
              x2="730"
              y2="140"
              stroke="#2E7D32"
              strokeWidth="1.25"
            />
            <line
              x1="870"
              y1="140"
              x2="910"
              y2="140"
              stroke="#111111"
              strokeWidth="1"
            />
            <line
              x1="1060"
              y1="140"
              x2="1100"
              y2="140"
              stroke="#111111"
              strokeWidth="1"
            />
            <line
              x1="1220"
              y1="140"
              x2="1260"
              y2="140"
              stroke="#111111"
              strokeWidth="1"
            />

            {NODES.map((node) => (
              <g key={node.id}>
                <rect
                  x={node.x}
                  y={node.y}
                  width={node.w}
                  height={60}
                  fill="#FFFFFF"
                  stroke={node.accent ? "#2E7D32" : "#111111"}
                  strokeWidth="1"
                />
                <text
                  x={node.x + node.w / 2}
                  y={node.y + 35}
                  textAnchor="middle"
                  fill="#111111"
                  fontSize="13"
                  fontFamily="Inter, sans-serif"
                >
                  {node.label}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>
    </section>
  );
}
