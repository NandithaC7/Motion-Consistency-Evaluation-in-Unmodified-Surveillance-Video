import Link from "next/link";
import { SITE } from "@/lib/site";

export function Footer() {
  return (
    <footer className="border-t border-line mt-0">
      <div className="shell py-12 md:py-16">
        <div className="grid-12 items-start">
          <div className="col-span-12 md:col-span-5 mb-10 md:mb-0">
            <p className="type-label mb-4">Project</p>
            <p className="max-w-sm text-[15px] leading-relaxed text-ink">
              {SITE.shortTitle} — motion consistency evaluation for unmodified
              surveillance video.
            </p>
          </div>
          <div className="col-span-6 md:col-span-2 mb-8 md:mb-0">
            <p className="type-label mb-4">Index</p>
            <ul className="space-y-2 text-[14px]">
              <li>
                <a
                  href={SITE.github}
                  target="_blank"
                  rel="noreferrer"
                  className="link-nav"
                >
                  GitHub
                </a>
              </li>
              <li>
                <Link href="/docx" className="link-nav">
                  Documentation
                </Link>
              </li>
            </ul>
          </div>
          <div className="col-span-6 md:col-span-2 mb-8 md:mb-0">
            <p className="type-label mb-4">Team</p>
            <ul className="space-y-2 text-[14px]">
              <li>
                <Link href="/docx#team" className="link-nav">
                  Team
                </Link>
              </li>
              <li>
                <Link href="/about" className="link-nav">
                  Model
                </Link>
              </li>
            </ul>
          </div>
          <div className="col-span-12 md:col-span-3">
            <p className="type-label mb-4">University</p>
            <p className="text-[14px] leading-relaxed text-muted">
              {SITE.department}
              <br />
              {SITE.university}
              <br />
              {SITE.campus}
            </p>
          </div>
        </div>
        <div className="mt-12 pt-6 border-t border-line flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <p className="text-[13px] text-muted">{SITE.course}</p>
          <p className="text-[13px] text-muted">{SITE.year}</p>
        </div>
      </div>
    </footer>
  );
}
