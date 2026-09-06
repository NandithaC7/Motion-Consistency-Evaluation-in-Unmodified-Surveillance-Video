import { Architecture } from "@/components/Architecture";
import { Hero } from "@/components/Hero";
import { Highlights } from "@/components/Highlights";
import { HomeDemo } from "@/components/HomeDemo";

export default function HomePage() {
  return (
    <>
      <Hero />
      <HomeDemo />
      <Highlights />
      <Architecture />
    </>
  );
}
