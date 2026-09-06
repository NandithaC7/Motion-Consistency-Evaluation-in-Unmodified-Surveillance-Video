import Link from "next/link";
import { DashboardMockup } from "@/components/DashboardMockup";
import { FadeIn } from "@/components/FadeIn";
import { SITE } from "@/lib/site";

export function Hero() {
  return (
    <section className="border-b border-line">
      <div className="shell">
        <div className="grid-12 items-stretch">
          <div className="col-span-12 lg:col-span-5 flex flex-col justify-start pt-16 pb-14 md:pt-20 md:pb-20 lg:pr-6">
            <FadeIn>
              <p className="type-label mb-6">
                Final Year Project · {SITE.year}
              </p>
              <div className="rule-accent mb-8" />
              <h1 className="type-hero">
                {SITE.title}
              </h1>
              <p className="mt-8 max-w-md text-[16px] leading-[1.7] text-muted">
                {SITE.summary}
              </p>
              <div className="mt-10 flex flex-wrap gap-3">
                <Link href="/docx" className="btn btn-primary">
                  View Documentation
                </Link>
                <Link href="/results" className="btn btn-secondary">
                  View Results
                </Link>
              </div>
            </FadeIn>
          </div>

          <div className="col-span-12 lg:col-span-7 lg:border-l border-line py-0 lg:pl-0 -mx-5 md:-mx-10 xl:-mx-16 lg:mx-0 lg:h-auto">
            <div className="h-full lg:border-l-0 border-t lg:border-t-0 border-line">
              <DashboardMockup />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
