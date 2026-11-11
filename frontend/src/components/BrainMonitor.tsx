'use client';

import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface TelemetryEvent {
    id: string;
    timestamp: string;
    component: string;
    data: any;
}

interface BrainMonitorProps {
    isActive: boolean;
    onToggle: () => void;
    logs: TelemetryEvent[];
}

export function BrainMonitor({ isActive, onToggle, logs }: BrainMonitorProps) {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom of logs
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs, isActive]);

    return (
        <div className="relative w-full">
            <AnimatePresence>
                {isActive ? (
                    <motion.div
                        initial={{ opacity: 0, y: 20, height: 0 }}
                        animate={{ opacity: 1, y: 0, height: 200 }}
                        exit={{ opacity: 0, y: 20, height: 0 }}
                        className="bg-[#0b1121]/95 backdrop-blur-xl border border-cyan-500/30 rounded-xl mb-4 overflow-hidden flex flex-col shadow-[0_0_30px_-5px_rgba(6,182,212,0.3)]"
                    >
                        {/* Header */}
                        <div className="flex justify-between items-center px-4 py-2 border-b border-cyan-500/20 bg-cyan-950/30">
                            <span className="text-xs font-mono text-cyan-400 font-semibold tracking-wider flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                                ACTIVE TELEMETRY STREAM
                            </span>
                            <button onClick={onToggle} className="text-cyan-500 hover:text-cyan-300">
                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                </svg>
                            </button>
                        </div>

                        {/* Logs Body */}
                        <div ref={scrollRef} className="p-4 overflow-y-auto font-mono text-xs flex-1 space-y-2">
                            {logs.length === 0 ? (
                                <div className="text-cyan-500/50 italic">Awaiting pipeline initialization...</div>
                            ) : (
                                logs.map((log) => (
                                    <motion.div
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        key={log.id}
                                        className="flex flex-col border-l-2 border-cyan-500/30 pl-2"
                                    >
                                        <span className="text-cyan-300/60 text-[10px]">
                                            {new Date(log.timestamp).toLocaleTimeString()} - [{log.component}]
                                        </span>
                                        <span className="text-cyan-100 whitespace-pre-wrap">
                                            {typeof log.data === 'string' ? log.data : JSON.stringify(log.data, null, 2)}
                                        </span>
                                    </motion.div>
                                ))
                            )}
                        </div>
                    </motion.div>
                ) : (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="absolute -top-12 left-2 z-20"
                    >
                        <button
                            onClick={onToggle}
                            className="glass-button w-10 h-10 rounded-full flex justify-center items-center text-cyan-400 shadow-[0_0_15px_-3px_rgba(6,182,212,0.4)]"
                            title="Open Brain Monitor"
                        >
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M4 6h16M4 18h16" />
                            </svg>
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
