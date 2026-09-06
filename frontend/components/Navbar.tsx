"use client";

import { Menu, X } from "lucide-react";
import { GitHubIcon } from "@/components/GitHubIcon";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { NAV_LINKS, SITE } from "@/lib/site";

export function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 6);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={`sticky top-0 z-50 bg-bg transition-[border-color] duration-300 ${
        scrolled || open ? "border-b border-line" : "border-b border-transparent"
      }`}
    >
      <div className="shell">
        <div className="grid-12 h-16 items-center">
          <div className="col-span-6 md:col-span-3 flex items-center">
            <Link
              href="/"
              className="flex items-center gap-3 group"
              onClick={() => setOpen(false)}
            >
              <span className="h-3 w-3 bg-accent" aria-hidden />
              <span className="text-[13px] tracking-[0.18em] uppercase text-ink">
                {SITE.shortTitle}
              </span>
            </Link>
          </div>

          <nav className="hidden md:flex col-span-6 items-center justify-center gap-10">
            {NAV_LINKS.map((link) => {
              const active =
                link.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  data-active={active}
                  className="link-nav text-[13px] tracking-[0.12em] uppercase text-ink"
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>

          <div className="col-span-6 md:col-span-3 flex items-center justify-end gap-4">
            <a
              href={SITE.github}
              target="_blank"
              rel="noreferrer"
              className="hidden sm:inline-flex items-center gap-2 text-[13px] tracking-[0.12em] uppercase text-ink link-nav"
            >
              <GitHubIcon size={14} />
              GitHub
            </a>
            <button
              type="button"
              className="md:hidden inline-flex h-9 w-9 items-center justify-center border border-line"
              aria-label={open ? "Close menu" : "Open menu"}
              onClick={() => setOpen((v) => !v)}
            >
              {open ? <X size={16} /> : <Menu size={16} />}
            </button>
          </div>
        </div>
      </div>

      {open ? (
        <div className="md:hidden border-t border-line bg-bg">
          <div className="shell py-6 flex flex-col gap-5">
            {NAV_LINKS.map((link) => {
              const active =
                link.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  data-active={active}
                  className="text-[15px] tracking-[0.1em] uppercase"
                  style={{ color: active ? "#2E7D32" : "#111111" }}
                  onClick={() => setOpen(false)}
                >
                  {link.label}
                </Link>
              );
            })}
            <a
              href={SITE.github}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 text-[15px] tracking-[0.1em] uppercase"
            >
              <GitHubIcon size={14} />
              GitHub
            </a>
          </div>
        </div>
      ) : null}
    </header>
  );
}
