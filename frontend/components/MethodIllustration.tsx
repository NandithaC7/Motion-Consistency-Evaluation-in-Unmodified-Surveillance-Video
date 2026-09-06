export function MethodIllustration() {
  const steps = [
    { n: "01", t: "Acquire" },
    { n: "02", t: "Extract" },
    { n: "03", t: "Represent" },
    { n: "04", t: "Analyse" },
    { n: "05", t: "Score" },
    { n: "06", t: "Output" },
  ];

  return (
    <div className="overflow-x-auto border-t border-line">
      <svg
        viewBox="0 0 960 160"
        className="w-full min-w-[720px] h-auto"
        role="img"
        aria-label="Six-step method line illustration"
      >
        <line x1="48" y1="80" x2="912" y2="80" stroke="#DADADA" strokeWidth="1" />
        <line x1="48" y1="80" x2="320" y2="80" stroke="#2E7D32" strokeWidth="1.25" />
        {steps.map((step, i) => {
          const x = 48 + i * 172;
          const accent = i === 0 || i === 5;
          return (
            <g key={step.n}>
              <rect
                x={x}
                y={52}
                width="56"
                height="56"
                fill="#FFFFFF"
                stroke={accent ? "#2E7D32" : "#111111"}
                strokeWidth="1"
              />
              <text
                x={x + 28}
                y={84}
                textAnchor="middle"
                fontSize="13"
                fontFamily="Inter, sans-serif"
                fill="#111111"
              >
                {step.n}
              </text>
              <text
                x={x + 28}
                y="136"
                textAnchor="middle"
                fontSize="12"
                fontFamily="Inter, sans-serif"
                fill="#6B7280"
                letterSpacing="1.2"
              >
                {step.t.toUpperCase()}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
