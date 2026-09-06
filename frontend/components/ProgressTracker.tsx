import { PROGRESS } from "@/lib/site";

const STATUS_COLOR = {
  Completed: "#2E7D32",
  "In Progress": "#2E7D32",
  Pending: "#6B7280",
};

export function ProgressTracker() {
  return (
    <ol className="relative">
      {PROGRESS.map((item, i) => (
        <li key={item.title} className="relative pl-10 pb-10 last:pb-0">
          {i < PROGRESS.length - 1 ? (
            <span className="absolute left-[5px] top-5 bottom-0 w-px bg-line" />
          ) : null}
          <span
            className="absolute left-0 top-1.5 h-2.5 w-2.5"
            style={{
              background:
                item.status === "Pending" ? "transparent" : STATUS_COLOR[item.status],
              border: `1px solid ${STATUS_COLOR[item.status]}`,
            }}
          />
          <p className="text-[22px] tracking-[-0.02em]">{item.title}</p>
          <p
            className="mt-1 text-[13px] tracking-[0.12em] uppercase"
            style={{ color: STATUS_COLOR[item.status] }}
          >
            {item.status}
          </p>
        </li>
      ))}
    </ol>
  );
}
