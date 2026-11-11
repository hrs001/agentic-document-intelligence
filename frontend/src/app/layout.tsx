import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Universal Knowledge Copilot",
  description: "Advanced Agentic RAG and Knowledge Graph Explorer",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} antialiased text-slate-50 relative h-screen w-screen overflow-hidden`}
      >
        <div className="aurora-gradient-bg"></div>
        <div className="aurora-blob bg-cyan-600/20 w-96 h-96 top-[-10%] left-[-10%] rounded-full"></div>
        <div className="aurora-blob bg-indigo-600/20 w-[30rem] h-[30rem] bottom-[-20%] right-[-10%] rounded-full" style={{ animationDelay: '-5s' }}></div>
        {children}
      </body>
    </html>
  );
}
