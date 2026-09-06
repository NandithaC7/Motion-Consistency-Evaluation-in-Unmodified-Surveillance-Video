export function DashboardMockup() {
  return (
    <div className="h-full min-h-[420px] lg:min-h-[640px] border border-line bg-bg flex flex-col">
      <div className="h-10 px-4 border-b border-line flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="type-label !tracking-[0.16em]">Cam-04</span>
          <span className="text-line">/</span>
          <span className="text-[13px] text-muted">Lot B · Live</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 bg-accent" />
          <span className="text-[13px] tracking-[0.12em] uppercase text-accent">
            Healthy
          </span>
        </div>
      </div>

      <div className="grid grid-cols-12 flex-1 min-h-0">
        <div className="col-span-12 lg:col-span-8 grid grid-cols-2 border-b lg:border-b-0 lg:border-r border-line">
          <FeedCell
            id="01"
            label="Approach"
            variant="lot"
            className="border-r border-b border-line"
          />
          <FeedCell
            id="02"
            label="Intersection"
            variant="street"
            className="border-b border-line"
          />
          <FeedCell
            id="03"
            label="Corridor"
            variant="hall"
            className="border-r border-line"
          />
          <FeedCell id="04" label="Motion field" variant="flow" />
        </div>

        <aside className="col-span-12 lg:col-span-4 flex flex-col">
          <div className="p-5 border-b border-line">
            <p className="type-label mb-3">Reliability</p>
            <p className="text-[44px] leading-none tracking-[-0.04em] font-medium">
              0.962
            </p>
            <p className="mt-2 text-[13px] text-muted">Feed score R ∈ [0, 1]</p>
          </div>
          <div className="p-5 border-b border-line">
            <p className="type-label mb-4">Consistency</p>
            <div className="h-1.5 w-full bg-section border border-line">
              <div className="h-full w-[96%] bg-accent" />
            </div>
            <div className="mt-4 space-y-2 text-[13px]">
              <Row k="eA · frame diff" v="0.00087" />
              <Row k="eB · optical flow" v="0.03168" />
              <Row k="λ" v="2.0" />
            </div>
          </div>
          <div className="p-5 flex-1">
            <p className="type-label mb-4">Stream state</p>
            <ul className="space-y-3 text-[13px]">
              <li className="flex items-center justify-between">
                <span>Guard G1 / G2</span>
                <span className="text-accent">Clear</span>
              </li>
              <li className="flex items-center justify-between">
                <span>Window T</span>
                <span>16 frames</span>
              </li>
              <li className="flex items-center justify-between">
                <span>Class</span>
                <span>Healthy</span>
              </li>
            </ul>
          </div>
        </aside>
      </div>

      <div className="h-12 border-t border-line px-4 flex items-center gap-4">
        <span className="text-[12px] text-muted tabular-nums">00:00</span>
        <div className="relative flex-1 h-px bg-line">
          <span className="absolute left-[38%] top-1/2 -translate-y-1/2 h-2 w-2 bg-accent" />
        </div>
        <span className="text-[12px] text-muted tabular-nums">00:24</span>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-muted">{k}</span>
      <span className="tabular-nums">{v}</span>
    </div>
  );
}

function FeedCell({
  id,
  label,
  variant,
  className = "",
}: {
  id: string;
  label: string;
  variant: "lot" | "street" | "hall" | "flow";
  className?: string;
}) {
  return (
    <div className={`relative min-h-[140px] overflow-hidden bg-section ${className}`}>
      <svg
        viewBox="0 0 240 140"
        className="absolute inset-0 h-full w-full transition-transform duration-500 hover:scale-[1.02]"
        preserveAspectRatio="xMidYMid slice"
        aria-hidden
      >
        {variant === "lot" ? <LotScene /> : null}
        {variant === "street" ? <StreetScene /> : null}
        {variant === "hall" ? <HallScene /> : null}
        {variant === "flow" ? <FlowScene /> : null}
      </svg>
      <div className="absolute left-2 top-2 flex items-center gap-2">
        <span className="text-[11px] tracking-[0.14em] uppercase text-ink/80">
          {id}
        </span>
        <span className="text-[11px] text-muted">{label}</span>
      </div>
    </div>
  );
}

function LotScene() {
  return (
    <g fill="none" stroke="#111111" strokeWidth="1">
      <rect x="0" y="0" width="240" height="140" fill="#F7F8F5" />
      <path d="M0 110 H240" />
      <path d="M20 110 L70 40 H170 L220 110" />
      <path d="M80 110 V70 H160 V110" />
      <path d="M40 110 L78 52" strokeDasharray="3 3" />
      <path d="M200 110 L162 52" strokeDasharray="3 3" />
      <rect x="92" y="78" width="22" height="14" />
      <rect x="126" y="82" width="22" height="14" />
      <circle cx="198" cy="28" r="3" fill="#2E7D32" stroke="none" />
    </g>
  );
}

function StreetScene() {
  return (
    <g fill="none" stroke="#111111" strokeWidth="1">
      <rect x="0" y="0" width="240" height="140" fill="#FFFFFF" />
      <path d="M0 72 H240" />
      <path d="M120 0 V140" />
      <path d="M0 72 H240" stroke="#DADADA" />
      <path d="M10 72 H50" strokeDasharray="6 6" />
      <path d="M70 72 H110" strokeDasharray="6 6" />
      <path d="M130 72 H170" strokeDasharray="6 6" />
      <path d="M190 72 H230" strokeDasharray="6 6" />
      <rect x="38" y="46" width="28" height="16" />
      <rect x="168" y="80" width="28" height="16" />
      <path d="M20 20 H70 V48 H20 Z" />
      <path d="M170 18 H220 V50 H170 Z" />
      <circle cx="24" cy="118" r="2.5" fill="#2E7D32" stroke="none" />
    </g>
  );
}

function HallScene() {
  return (
    <g fill="none" stroke="#111111" strokeWidth="1">
      <rect x="0" y="0" width="240" height="140" fill="#F7F8F5" />
      <path d="M40 140 L90 20 H150 L200 140" />
      <path d="M90 20 V0" />
      <path d="M150 20 V0" />
      <path d="M70 80 H170" />
      <path d="M58 110 H182" />
      <rect x="108" y="52" width="24" height="40" />
      <path d="M20 30 H50" />
      <path d="M190 30 H220" />
    </g>
  );
}

function FlowScene() {
  return (
    <g fill="none" stroke="#2E7D32" strokeWidth="1">
      <rect x="0" y="0" width="240" height="140" fill="#FFFFFF" />
      {Array.from({ length: 7 }).map((_, row) =>
        Array.from({ length: 10 }).map((__, col) => {
          const x = 16 + col * 22;
          const y = 18 + row * 18;
          const dx = 10 + (row % 3) * 2;
          const dy = (col % 2 === 0 ? -3 : 4) + row * 0.2;
          return (
            <path
              key={`${row}-${col}`}
              d={`M${x} ${y} l${dx} ${dy}`}
              markerEnd="url(#arrow)"
            />
          );
        }),
      )}
      <defs>
        <marker
          id="arrow"
          markerWidth="6"
          markerHeight="6"
          refX="5"
          refY="3"
          orient="auto"
        >
          <path d="M0 0 L6 3 L0 6 Z" fill="#2E7D32" />
        </marker>
      </defs>
    </g>
  );
}
