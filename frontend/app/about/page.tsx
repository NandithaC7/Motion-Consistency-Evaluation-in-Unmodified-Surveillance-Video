import type { Metadata } from "next";
import { FadeIn } from "@/components/FadeIn";
import { MethodIllustration } from "@/components/MethodIllustration";
import { METHOD_STEPS, SPECS } from "@/lib/site";

export const metadata: Metadata = {
  title: "About the Model",
};

export default function AboutPage() {
  return (
    <>
      <section className="border-b border-line">
        <div className="shell py-20 md:py-28">
          <div className="grid-12">
            <div className="col-span-12 lg:col-span-5">
              <FadeIn>
                <p className="type-label mb-4">Section 01</p>
                <h1 className="type-section">About the Model</h1>
                <div className="rule-accent mt-8" />
              </FadeIn>
            </div>
            <div className="col-span-12 lg:col-span-7 mt-10 lg:mt-0">
              <FadeIn delay={0.08}>
                <p className="text-[22px] leading-[1.5] tracking-[-0.02em] max-w-2xl">
                  Temporal motion consistency is the assumption that a healthy
                  camera records a physically plausible sequence: objects move,
                  lighting shifts, and no frame is an exact, unexplained copy of
                  the last.
                </p>
                <div className="mt-8 space-y-6 text-[16px] leading-[1.75] text-muted max-w-2xl">
                  <p>
                    Surveillance systems fail silently. Frozen frames, duplicated
                    segments, temporal jitter, and slow signal drift do not
                    introduce a new object into the scene, so conventional
                    anomaly detectors continue to report a normal environment
                    while the feed itself has become unusable.
                  </p>
                  <p>
                    This model treats the camera — not the scene — as the
                    subject of evaluation. It reads consecutive frames, builds
                    two motion representations, and asks whether those
                    representations reconstruct as a healthy temporal process.
                  </p>
                  <p>
                    Unmodified footage matters because forensic, academic, and
                    operational review cannot rely on a pipeline that rewrites
                    the source. The original video is never altered. Evaluation
                    is observational: frames are sampled, scored, and left
                    intact.
                  </p>
                </div>
              </FadeIn>
            </div>
          </div>
        </div>
      </section>

      <section className="border-b border-line">
        <div className="shell py-20 md:py-28">
          <FadeIn>
            <p className="type-label mb-4">Section 02</p>
            <h2 className="type-section">Core Specifications</h2>
          </FadeIn>

          <div className="mt-12 border-t border-line">
            {SPECS.map((row) => (
              <div
                key={row.key}
                className="grid-12 border-b border-line py-5"
              >
                <p className="col-span-5 md:col-span-4 text-[13px] tracking-[0.1em] uppercase text-muted">
                  {row.key}
                </p>
                <p className="col-span-7 md:col-span-8 text-[16px] md:text-[18px] tracking-[-0.015em]">
                  {row.value}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="border-b border-line">
        <div className="shell pt-20 md:pt-28 pb-0">
          <FadeIn>
            <p className="type-label mb-4">Section 03</p>
            <h2 className="type-section">Method Overview</h2>
            <p className="mt-5 max-w-xl text-[16px] text-muted">
              Six sequential operations. No stage writes back into the source
              video.
            </p>
          </FadeIn>

          <ol className="mt-14 grid-12">
            {METHOD_STEPS.map((step, i) => (
              <FadeIn
                key={step.n}
                delay={i * 0.04}
                className="col-span-12 md:col-span-6 py-8 border-t border-line"
              >
                <p className="text-[13px] tracking-[0.14em] uppercase text-accent mb-3">
                  {step.n}
                </p>
                <h3 className="text-[22px] tracking-[-0.02em]">{step.title}</h3>
                <p className="mt-3 max-w-md text-[16px] text-muted">{step.body}</p>
              </FadeIn>
            ))}
          </ol>
        </div>
        <MethodIllustration />
      </section>

      <section className="bg-section">
        <div className="shell py-20 md:py-28">
          <FadeIn>
            <p className="type-label mb-4">Section 04</p>
            <h2 className="type-section max-w-3xl">
              Why motion consistency?
            </h2>
            <p className="mt-8 max-w-3xl text-[22px] leading-[1.5] tracking-[-0.02em]">
              Object-centric detectors answer a different question: is there
              something unusual in the scene? Motion consistency asks whether
              the camera is still a reliable witness. A feed can be semantically
              quiet and operationally broken at the same time.
            </p>
          </FadeIn>

          <div className="mt-16 grid-12">
            {[
              {
                title: "Preserves original footage.",
                body: "The source file is treated as evidence. Nothing is overlaid, compressed in place, or watermarked by the evaluator.",
              },
              {
                title: "Captures temporal behaviour.",
                body: "Frame difference and optical flow describe how the scene moves, not what objects are named.",
              },
              {
                title: "Supports anomaly evaluation.",
                body: "A consistency score and a reliability index sit beneath downstream analytics so those models are not fed silent failures.",
              },
            ].map((item, i) => (
              <FadeIn
                key={item.title}
                delay={i * 0.08}
                className="col-span-12 md:col-span-4 pt-8 md:pt-0 border-t md:border-t-0 md:border-r border-line last:border-r-0 md:pr-8 first:border-t-0"
              >
                <h3 className="text-[22px] tracking-[-0.02em] max-w-xs">
                  {item.title}
                </h3>
                <p className="mt-4 text-[16px] text-muted max-w-xs">{item.body}</p>
              </FadeIn>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
