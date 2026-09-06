import { TIMELINE } from "@/lib/site";

export function Timeline() {
  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-[15px] mb-12">
        {["Planning", "Data Collection", "Model Development", "Testing", "Final Demo"].map(
          (step, i, arr) => (
            <span key={step} className="inline-flex items-center gap-3">
              <span>{step}</span>
              {i < arr.length - 1 ? (
                <span className="text-muted" aria-hidden>
                  →
                </span>
              ) : null}
            </span>
          ),
        )}
      </div>

      <div className="relative">
        {TIMELINE.map((row, i) => (
          <div key={row.month} className="relative pl-10 pb-10 last:pb-0">
            {i < TIMELINE.length - 1 ? (
              <span className="absolute left-[5px] top-5 bottom-0 w-px bg-line" />
            ) : null}
            <span className="absolute left-0 top-1.5 h-2.5 w-2.5 bg-accent" />
            <div className="grid-12">
              <p className="col-span-12 sm:col-span-3 type-label !normal-case !tracking-[0.04em] pt-0.5">
                {row.month}
              </p>
              <p className="col-span-12 sm:col-span-9 text-[18px] tracking-[-0.015em]">
                {row.activity}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
