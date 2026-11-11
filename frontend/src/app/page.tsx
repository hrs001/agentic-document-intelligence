'use client';

import React from 'react';
import { ChatInterface } from '../components/ChatInterface';
import { DataVault } from '../components/DataVault';
import { GraphVisualizer } from '../components/GraphVisualizer';

export default function Home() {
  return (
    <main className="flex h-screen w-full overflow-hidden text-slate-50 relative font-sans">

      {/* Absolute Top-Right File Upload Button */}
      <DataVault />

      {/* LEFT 35%: Chat Interface */}
      <ChatInterface />

      {/* RIGHT 65%: Interactive Knowledge Graph */}
      <GraphVisualizer />
    </main>
  );
}
