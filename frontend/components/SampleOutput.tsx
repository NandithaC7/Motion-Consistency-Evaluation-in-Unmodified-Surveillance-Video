const FRAMES = [
  {
    src: "/placeholders/original-frame.svg",
    label: "Original Frame",
  },
  {
    src: "/placeholders/motion-viz.svg",
    label: "Motion Visualization",
  },
  {
    src: "/placeholders/consistency-map.svg",
    label: "Consistency Map",
  },
];

export function SampleOutput() {
  return (
    <div className="grid-12 items-stretch">
      {FRAMES.map((frame, i) => (
        <div key={frame.label} className="col-span-12 md:col-span-4">
          <div className="border border-line bg-section overflow-hidden">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={frame.src}
              alt={frame.label}
              className="w-full h-auto transition-transform duration-500 hover:scale-[1.02]"
            />
          </div>
          <div className="mt-4 flex items-center justify-between">
            <p className="type-label">{frame.label}</p>
            {i < FRAMES.length - 1 ? (
              <span className="text-muted text-[13px] tracking-[0.12em] uppercase md:hidden">
                ↓
              </span>
            ) : null}
          </div>
          {i < FRAMES.length - 1 ? (
            <p className="md:hidden text-center text-muted py-4">↓</p>
          ) : null}
        </div>
      ))}
    </div>
  );
}
