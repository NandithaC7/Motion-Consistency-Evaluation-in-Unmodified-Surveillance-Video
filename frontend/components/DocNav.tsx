"use client";

import { useEffect, useState } from "react";
import { DOC_SECTIONS } from "@/lib/site";

export function DocNav() {
  const [active, setActive] = useState<string>(DOC_SECTIONS[0].id);

  useEffect(() => {
    const nodes = DOC_SECTIONS.map((s) => document.getElementById(s.id)).filter(
      (n): n is HTMLElement => Boolean(n),
    );
    if (!nodes.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible?.target.id) setActive(visible.target.id);
      },
      { rootMargin: "0px 0px -55% 0px", threshold: [0.15, 0.4, 0.7] },
    );

    nodes.forEach((n) => observer.observe(n));
    return () => observer.disconnect();
  }, []);

  return (
    <nav aria-label="Documentation">
      <p className="type-label mb-6">Documentation</p>
      <ul className="flex lg:flex-col gap-4 lg:gap-3 overflow-x-auto pb-1">
        {DOC_SECTIONS.map((section) => (
          <li key={section.id} className="shrink-0">
            <a
              href={`#${section.id}`}
              data-active={active === section.id}
              className="link-nav text-[13px] tracking-[0.08em] uppercase whitespace-nowrap"
            >
              {section.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
