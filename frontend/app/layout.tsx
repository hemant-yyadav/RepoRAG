import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Codebase RAG Assistant",
  description: "Phase 0 development scaffold",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
