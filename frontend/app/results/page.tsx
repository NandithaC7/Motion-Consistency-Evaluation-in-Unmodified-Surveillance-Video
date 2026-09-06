import type { Metadata } from "next";
import {
  AccuracyCurve,
  ConfusionMatrix,
  PrecisionRecallCurve,
  RocCurve,
  TrainingLossCurve,
} from "@/components/Charts";
import { FadeIn } from "@/components/FadeIn";
import { SampleOutput } from "@/components/SampleOutput";
import { METRICS } from "@/lib/site";

export const metadata: Metadata = {
  title: "Results",
};

export default function ResultsPage() {
  return (
    <>
      <section className="border-b border-line">
        <div className="shell py-20 md:py-28">
          <FadeIn>
            <p className="type-label mb-4">Evaluation</p>
            <h1 className="type-section">Experimental Results</h1>
            <p className="mt-6 max-w-xl text-[16px] text-muted">
              Demonstration placeholders prepared for university evaluation.
              Figures share one visual language: white ground, thin grid, green
              highlight. Replace series with live logs when the experiment
              runner is attached.
            </p>
          </FadeIn>
        </div>
      </section>

      <section className="border-b border-line">
        <div className="shell py-0">
          <div className="grid grid-cols-2 lg:grid-cols-4">
            {METRICS.map((metric, i) => (
              <div
                key={metric.label}
                className={[
                  "py-12 md:py-16",
                  i % 2 === 1 ? "border-l border-line pl-6 md:pl-8" : "pr-6 md:pr-8",
                  i < 2 ? "border-b lg:border-b-0 border-line" : "",
                  i > 0 ? "lg:border-l lg:border-line lg:pl-10" : "lg:pr-10",
                  i === 1 || i === 2 ? "lg:px-10" : "",
                ].join(" ")}
              >
                <p className="type-label mb-4">{metric.label}</p>
                <p className="text-[44px] md:text-[56px] leading-none tracking-[-0.04em] font-medium">
                  {metric.value}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <ChartSection
        index="01"
        title="Confusion Matrix"
        copy="Predicted versus true class over a balanced placeholder set: Healthy, Irregular, Frozen. Intensity is green on white."
      >
        <ConfusionMatrix />
      </ChartSection>

      <ChartSection
        index="02"
        title="ROC Curve"
        copy="Receiver operating characteristic for the divergence score. Black axes, green curve, no decorative fill."
      >
        <RocCurve />
      </ChartSection>

      <ChartSection
        index="03"
        title="Precision–Recall Curve"
        copy="Precision held against recall as the operating threshold sweeps. Same stroke language as the ROC."
      >
        <PrecisionRecallCurve />
      </ChartSection>

      <ChartSection
        index="04"
        title="Training Loss Curve"
        copy="Descending reconstruction loss for Stream A (frame difference) and Stream B (optical flow)."
      >
        <TrainingLossCurve />
      </ChartSection>

      <ChartSection
        index="05"
        title="Accuracy Curve"
        copy="Ascending train and validation accuracy across placeholder epochs."
      >
        <AccuracyCurve />
      </ChartSection>

      <section className="border-b border-line bg-section">
        <div className="shell py-20 md:py-28">
          <FadeIn>
            <p className="type-label mb-4">Sample output</p>
            <h2 className="type-section">Frame, motion, map</h2>
            <p className="mt-5 max-w-xl text-[16px] text-muted">
              Horizontal reading order. Placeholders stand in for a decoded
              frame, its motion field, and the resulting consistency map.
            </p>
          </FadeIn>
          <div className="mt-14">
            <SampleOutput />
          </div>
        </div>
      </section>
    </>
  );
}

function ChartSection({
  index,
  title,
  copy,
  children,
}: {
  index: string;
  title: string;
  copy: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-b border-line">
      <div className="shell py-16 md:py-24">
        <div className="grid-12 mb-10">
          <div className="col-span-12 md:col-span-4">
            <p className="type-label mb-3">{index}</p>
            <h2 className="type-section">{title}</h2>
          </div>
          <div className="col-span-12 md:col-span-7 md:col-start-6 mt-4 md:mt-0">
            <p className="text-[16px] text-muted max-w-lg">{copy}</p>
          </div>
        </div>
        {children}
      </div>
    </section>
  );
}
