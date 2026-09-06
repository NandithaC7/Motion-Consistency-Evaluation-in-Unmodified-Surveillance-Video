import { FadeIn } from "@/components/FadeIn";
import { HIGHLIGHTS } from "@/lib/site";

export function Highlights() {
  return (
    <section className="border-b border-line">
      <div className="shell py-20 md:py-28">
        <FadeIn>
          <p className="type-label mb-4">Key highlights</p>
          <h2 className="type-section max-w-xl">
            A restrained system for a noisy operational problem.
          </h2>
        </FadeIn>

        <div className="mt-16 grid-12">
          {HIGHLIGHTS.map((item, i) => (
            <FadeIn
              key={item.title}
              delay={i * 0.08}
              className="col-span-12 md:col-span-4 pt-8 md:pt-0 md:border-0 border-t border-line first:border-t-0 md:first:border-t-0"
            >
              <div
                className={`h-full ${
                  i < HIGHLIGHTS.length - 1 ? "md:pr-8 md:border-r md:border-line" : ""
                }`}
              >
                <p className="type-label mb-5">{item.lead}</p>
                <h3 className="text-[40px] md:text-[44px] leading-none tracking-[-0.03em] font-medium">
                  {item.title}
                </h3>
                <p className="mt-6 text-[16px] leading-[1.7] text-muted max-w-xs">
                  {item.body}
                </p>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}
