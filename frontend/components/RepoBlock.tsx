"use client";

import { Check, Copy } from "lucide-react";
import { GitHubIcon } from "@/components/GitHubIcon";
import { useState } from "react";
import { SITE } from "@/lib/site";

export function RepoBlock() {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(SITE.clone);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="border border-line">
      <div className="grid-12 items-center px-6 py-8 md:px-8">
        <div className="col-span-12 lg:col-span-8">
          <div className="flex items-center gap-3 mb-4">
            <GitHubIcon size={18} />
            <p className="type-label">Repository</p>
          </div>
          <h3 className="text-[22px] md:text-[28px] tracking-[-0.025em] break-all">
            {SITE.githubName}
          </h3>
          <p className="mt-3 max-w-xl text-[16px] text-muted">
          </p>
        </div>
        <div className="col-span-12 lg:col-span-4 mt-6 lg:mt-0 flex lg:justify-end">
          <button type="button" onClick={copy} className="btn btn-primary">
            {copied ? <Check size={14} /> : <Copy size={14} />}
            {copied ? "Copied" : "Clone"}
          </button>
        </div>
      </div>
      <div className="border-t border-line px-6 md:px-8 py-4 text-[13px] text-muted font-mono overflow-x-auto">
        {SITE.clone}
      </div>
    </div>
  );
}
