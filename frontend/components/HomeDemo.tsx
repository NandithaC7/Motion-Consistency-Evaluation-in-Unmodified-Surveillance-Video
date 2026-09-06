"use client";

import { useState } from "react";
import { Pipeline } from "@/components/Pipeline";
import { UploadZone } from "@/components/UploadZone";

export function HomeDemo() {
  const [fileName, setFileName] = useState<string | null>(null);

  return (
    <>
      <UploadZone onComplete={setFileName} />
      {fileName ? <Pipeline fileName={fileName} /> : null}
    </>
  );
}
