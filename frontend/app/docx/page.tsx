import type { Metadata } from "next";
import { Download } from "lucide-react";
import { DocNav } from "@/components/DocNav";
import { FadeIn } from "@/components/FadeIn";
import { PipelineDiagram } from "@/components/PipelineDiagram";
import { ProgressTracker } from "@/components/ProgressTracker";
import { RepoBlock } from "@/components/RepoBlock";
import { Timeline } from "@/components/Timeline";
import { OBJECTIVES, SITE, TEAM } from "@/lib/site";

export const metadata: Metadata = {
  title: "Documentation",
};

const DOWNLOADS = [
  { name: "Project Report.pdf", note: "Placeholder" },
  { name: "Research Paper.pdf", note: "Placeholder" },
  { name: "Presentation.pptx", note: "Placeholder" },
];

export default function DocxPage() {
  return (
    <div className="border-b border-line">
      <div className="shell py-12 md:py-20">
        <div className="grid-12 gap-y-12">
          <aside className="col-span-12 lg:col-span-3">
            <div className="lg:sticky lg:top-24">
              <DocNav />
            </div>
          </aside>

          <article className="col-span-12 lg:col-span-9 lg:border-l lg:border-line lg:pl-12">
            <FadeIn>
              <p className="type-label mb-4"></p>
              <h1 className="type-section">Documentation Portal</h1>
              <p className="mt-5 max-w-2xl text-[16px] text-muted">
                Formal record for {SITE.course}.
              </p>
            </FadeIn>

            <DocSection id="objectives" title="Project Objectives">
              <ul className="space-y-4">
                {OBJECTIVES.map((item) => (
                  <li
                    key={item}
                    className="pl-6 relative text-[16px] leading-[1.7]"
                  >
                    <span className="absolute left-0 top-2.5 h-1.5 w-1.5 bg-accent" />
                    {item}
                  </li>
                ))}
              </ul>
            </DocSection>

            <DocSection id="problem" title="Problem Statement">
              <p className="text-[18px] leading-[1.75] tracking-[-0.01em]">
                Surveillance camera systems generate continuous video streams;
                however, real-world feeds often suffer from hidden issues such
                as frozen frames, prolonged low motion, and temporal
                inconsistencies caused by environmental or hardware conditions.
                These issues do not trigger traditional object-based alerts but
                significantly reduce monitoring reliability. This project
                evaluates motion consistency in unmodified surveillance videos
                in order to assess camera-feed health using a Dual-Stream LSTM
                Autoencoder, without altering the original footage and without
                depending solely on conventional anomaly detection.
              </p>
            </DocSection>

            <DocSection id="methodology" title="Methodology">
              <p className="text-[16px] leading-[1.75] text-muted mb-6">
                The system is a six-stage observational pipeline. Guard layers
                G1 (pixel variance) and G2 (SSIM streak) exit early on frozen
                feeds. Remaining frames are resized to 64×64, converted to
                grayscale, and represented as two motion streams: frame
                difference and Farnebäck optical-flow magnitude. Overlapping
                16-frame clips train two independent LSTM autoencoders on
                normal footage only. Reconstruction errors are fused with a
                divergence term Score = eA + eB + λ|eA − eB|. A dual-window
                adaptive threshold classifies each clip as Healthy, Irregular,
                or Frozen and aggregates a feed reliability score R.
              </p>
              <div className="border-t border-line">
                {[
                  ["Guard layer", "Variance and SSIM early exit"],
                  ["Stream A", "Absolute frame difference"],
                  ["Stream B", "Optical-flow magnitude"],
                  ["Model", "Dual 2-layer LSTM autoencoders"],
                  ["Decision", "Divergence score + adaptive θ"],
                ].map(([k, v]) => (
                  <div
                    key={k}
                    className="grid grid-cols-12 border-b border-line py-4"
                  >
                    <p className="col-span-5 sm:col-span-4 type-label !normal-case !tracking-[0.04em]">
                      {k}
                    </p>
                    <p className="col-span-7 sm:col-span-8 text-[16px]">{v}</p>
                  </div>
                ))}
              </div>
            </DocSection>

            <DocSection id="team" title="Team Details">
              <div className="border-t border-line">
                <div className="grid grid-cols-12 border-b border-line py-3">
                  <p className="col-span-6 type-label">Name</p>
                  <p className="col-span-6 type-label">Role and Contribution</p>
                </div>
                {TEAM.map((member) => (
                  <div
                    key={member.name}
                    className="grid grid-cols-12 border-b border-line py-5"
                  >
                    <p className="col-span-6 text-[16px]">{member.name}</p>
                    <p className="col-span-6 text-[16px] text-muted">
                      {member.role}
                    </p>
                  </div>
                ))}
              </div>
              <p className="mt-6 text-[14px] text-muted">
                {SITE.department}, {SITE.university}, {SITE.campus}.
              </p>
            </DocSection>

            <DocSection id="pipeline" title="Full Pipeline Diagram">
              <p className="text-[16px] text-muted mb-8">
                Thin black strokes, green highlights on ingress, temporal model,
                and final output. Square nodes. The diagram is intended to span
                the report column.
              </p>
              <PipelineDiagram />
            </DocSection>

            <DocSection id="progress" title="Project Progress">
              <ProgressTracker />
            </DocSection>

            <DocSection id="documents" title="Documentation">
              <div className="border-t border-line">
                {DOWNLOADS.map((file) => (
                  <div
                    key={file.name}
                    className="flex items-center justify-between gap-4 border-b border-line py-5"
                  >
                    <div>
                      <p className="text-[16px]">{file.name}</p>
                      <p className="text-[13px] text-muted mt-1">{file.note}</p>
                    </div>
                    <button type="button" className="btn btn-secondary">
                      <Download size={14} />
                      Download
                    </button>
                  </div>
                ))}
              </div>
            </DocSection>

            <DocSection id="github" title="GitHub Repository">
              <RepoBlock />
            </DocSection>

            <DocSection id="timeline" title="Semester Timeline">
              <Timeline />
            </DocSection>
          </article>
        </div>
      </div>
    </div>
  );
}

function DocSection({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-28 pt-16 md:pt-20">
      <div className="rule mb-8" />
      <h2 className="text-[28px] md:text-[36px] tracking-[-0.03em] font-medium mb-8">
        {title}
      </h2>
      {children}
    </section>
  );
}
