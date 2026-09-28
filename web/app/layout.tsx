import type { Metadata } from "next";
import { Big_Shoulders, IBM_Plex_Mono, Source_Sans_3 } from "next/font/google";
import { Chrome } from "@/components/Chrome";
import "./globals.css";

const display = Big_Shoulders({ subsets: ["latin"], weight: ["700", "800", "900"], variable: "--font-display", fallback: ["Impact", "Arial Narrow", "sans-serif"], adjustFontFallback: false });
const body = Source_Sans_3({ subsets: ["latin"], variable: "--font-body" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "600"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "AI Offensive Coordinator",
  description: "Dated defensive scouting reports and broad run-versus-dropback comparisons with explicit evidence and abstention.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>
        <Chrome>{children}</Chrome>
      </body>
    </html>
  );
}
