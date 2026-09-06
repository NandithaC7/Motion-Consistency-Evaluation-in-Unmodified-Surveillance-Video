"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Upload } from "lucide-react";
import { useRef, useState } from "react";

type UploadZoneProps = {
  onComplete?: (fileName: string) => void;
};

export function UploadZone({ onComplete }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState<"idle" | "reading" | "ready">("idle");

  const accept = (file?: File) => {
    if (!file) return;
    setFileName(file.name);
    setPhase("reading");
    setProgress(0);

    let value = 0;
    const tick = window.setInterval(() => {
      value += Math.random() * 18 + 8;
      if (value >= 100) {
        value = 100;
        window.clearInterval(tick);
        setProgress(100);
        window.setTimeout(() => {
          setPhase("ready");
          onComplete?.(file.name);
        }, 280);
      } else {
        setProgress(Math.round(value));
      }
    }, 140);
  };

  return (
    <section className="border-b border-line">
      <div className="shell py-16 md:py-24">
        <div className="grid-12">
          <div className="col-span-12 md:col-span-4 mb-8 md:mb-0">
            <p className="type-label mb-4">Interactive preview</p>
            <h2 className="type-section max-w-[10ch]">Drop a clip. Read the feed.</h2>
            <p className="mt-5 max-w-sm text-[16px] text-muted">
              No backend is attached. The drop zone simulates ingest so the
              evaluation pipeline can be walked on this page.
            </p>
          </div>

          <div className="col-span-12 md:col-span-8">
            <motion.div
              onDragOver={(e) => {
                e.preventDefault();
                setOver(true);
              }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setOver(false);
                accept(e.dataTransfer.files?.[0]);
              }}
              animate={{
                borderColor: over ? "#2E7D32" : "#DADADA",
                scale: over ? 1.01 : 1,
              }}
              transition={{ duration: 0.2 }}
              className="relative min-h-[280px] md:min-h-[340px] border bg-bg flex flex-col items-center justify-center px-6 py-12 text-center"
            >
              <input
                ref={inputRef}
                type="file"
                accept=".mp4,.avi,.mov,video/mp4,video/x-msvideo,video/quicktime"
                className="sr-only"
                onChange={(e) => accept(e.target.files?.[0] ?? undefined)}
              />

              <AnimatePresence mode="wait">
                {phase === "idle" ? (
                  <motion.div
                    key="idle"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    className="flex flex-col items-center"
                  >
                    <Upload
                      size={22}
                      strokeWidth={1.5}
                      className={over ? "text-accent" : "text-ink"}
                    />
                    <p className="mt-6 text-[22px] tracking-[-0.02em]">
                      Drop surveillance video here
                    </p>
                    <p className="mt-2 text-[16px] text-muted">or browse</p>
                    <button
                      type="button"
                      onClick={() => inputRef.current?.click()}
                      className="btn btn-secondary mt-8"
                    >
                      Browse
                    </button>
                    <p className="mt-8 type-label">MP4 · AVI · MOV</p>
                  </motion.div>
                ) : (
                  <motion.div
                    key="reading"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="w-full max-w-md"
                  >
                    <p className="type-label mb-3">
                      {phase === "ready" ? "Ready" : "Reading frames"}
                    </p>
                    <p className="text-[18px] tracking-[-0.02em] truncate">
                      {fileName}
                    </p>
                    <div className="mt-6 h-px w-full bg-line">
                      <div
                        className="h-px bg-accent transition-[width] duration-150"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <p className="mt-3 text-[13px] tabular-nums text-muted">
                      {progress}%
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}
