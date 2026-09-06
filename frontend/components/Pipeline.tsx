"use client";

import { motion } from "framer-motion";
import { PIPELINE_STEPS } from "@/lib/site";

type PipelineProps = {
  fileName?: string;
};

export function Pipeline({ fileName }: PipelineProps) {
  return (
    <section className="border-b border-line bg-section">
      <div className="shell py-16 md:py-24">
        <div className="grid-12">
          <div className="col-span-12 md:col-span-4 mb-12 md:mb-0">
            <p className="type-label mb-4">Processing preview</p>
            <h2 className="type-section">From ingest to score</h2>
            <p className="mt-5 max-w-sm text-[16px] text-muted">
              {fileName
                ? `Mock evaluation of ${fileName}. Each stage is sequential and non-destructive.`
                : "Each stage is sequential and non-destructive."}
            </p>
          </div>

          <div className="col-span-12 md:col-span-8">
            <ol className="relative">
              {PIPELINE_STEPS.map((step, i) => (
                <motion.li
                  key={step}
                  initial={{ opacity: 0, y: 16 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{
                    delay: i * 0.08,
                    duration: 0.5,
                    ease: [0.22, 1, 0.36, 1],
                  }}
                  className="relative pl-10 pb-10 last:pb-0"
                >
                  {i < PIPELINE_STEPS.length - 1 ? (
                    <span className="absolute left-[5px] top-5 bottom-0 w-px bg-line" />
                  ) : null}
                  <span
                    className={`absolute left-0 top-1.5 h-2.5 w-2.5 ${
                      i === PIPELINE_STEPS.length - 1 ? "bg-accent" : "bg-ink"
                    }`}
                  />
                  <p className="type-label mb-2">
                    Step {String(i + 1).padStart(2, "0")}
                  </p>
                  <p className="text-[22px] tracking-[-0.02em]">{step}</p>
                </motion.li>
              ))}
            </ol>

            <div className="mt-10 pt-8 border-t border-line flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
              <div>
                <p className="type-label mb-2">Output score</p>
                <p className="text-[48px] leading-none tracking-[-0.04em] font-medium">
                  0.962
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 bg-accent" />
                <span className="text-[13px] tracking-[0.12em] uppercase text-accent">
                  Healthy
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
