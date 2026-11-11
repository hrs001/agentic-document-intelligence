'use client';

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { getApiBaseUrl } from '../lib/api';
import { getOrCreateSessionId } from '../lib/session';

interface DemoAssetDownload {
    asset_id?: string;
    label?: string;
    file_name?: string;
    download_path?: string;
}

interface DemoDatasetTable {
    table_name?: string;
    download?: DemoAssetDownload;
}

interface DemoCorpus {
    corpus_id?: string;
    document?: {
        title?: string;
        issuer?: string;
        fiscal_period?: string;
        topic_tags?: string[];
        download?: DemoAssetDownload;
    };
    dataset?: {
        display_name?: string;
        tables?: DemoDatasetTable[];
    };
}

export function DataVault() {
    const [isOpen, setIsOpen] = useState(false);
    const [isHydrated, setIsHydrated] = useState(false);
    const [files, setFiles] = useState<string[]>([]);
    const [demoCorpus, setDemoCorpus] = useState<DemoCorpus | null>(null);

    const apiHost = getApiBaseUrl();

    const refreshStatus = async () => {
        try {
            const sid = getOrCreateSessionId();
            const res = await fetch(`${apiHost}/api/v1/session-status?session_id=${encodeURIComponent(sid)}`);
            if (!res.ok) {
                throw new Error(`Session status request failed (${res.status}).`);
            }
            const data = await res.json();
            setIsHydrated(Boolean(data.has_any_data));
            setFiles(Array.isArray(data.uploaded_files) ? data.uploaded_files : []);
            setDemoCorpus(data.demo_corpus ?? null);
        } catch (e) {
            console.error('Failed to refresh demo corpus status', e);
            setIsHydrated(false);
            setFiles([]);
            setDemoCorpus(null);
        }
    };

    useEffect(() => {
        void refreshStatus();
        const handleHydration = () => {
            void refreshStatus();
            setIsOpen(true);
            setTimeout(() => setIsOpen(false), 4000);
        };
        window.addEventListener('demoHydrated', handleHydration);
        return () => {
            window.removeEventListener('demoHydrated', handleHydration);
        };
    }, []);

    const buildDownloadHref = (download?: DemoAssetDownload) =>
        download?.download_path ? `${apiHost}${download.download_path}` : undefined;

    return (
        <div className="absolute top-4 right-4 z-50">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="glass-button px-4 py-2 rounded-full flex items-center gap-2 text-sm text-slate-100 shadow-[0_5px_15px_-3px_rgba(6,182,212,0.3)] relative"
            >
                <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
                Demo Corpus
                {isHydrated && (
                    <span className="absolute -top-1 -right-1 flex h-3 w-3">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
                    </span>
                )}
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        className="absolute top-12 right-0 w-96 glass-panel border border-cyan-500/20 rounded-2xl shadow-2xl p-4 flex flex-col gap-4 overflow-hidden"
                    >
                        <div className="flex justify-between items-center pb-2 border-b border-white/10">
                            <span className="text-sm font-semibold text-slate-200">Fixed Demo Corpus</span>
                        </div>

                        <div className="border border-cyan-500/20 rounded-xl p-4 bg-cyan-950/20 space-y-2">
                            <p className="text-xs uppercase tracking-widest text-cyan-300 font-semibold">Document</p>
                            <p className="text-sm text-slate-100">{demoCorpus?.document?.title ?? 'Microsoft FY2025 10-K Summary'}</p>
                            <p className="text-[11px] text-slate-400">
                                {demoCorpus?.document?.issuer ?? 'Microsoft'} / {demoCorpus?.document?.fiscal_period ?? 'FY2025'}
                            </p>
                            <div className="flex flex-wrap gap-2 pt-1">
                                {(demoCorpus?.document?.topic_tags ?? []).map((tag) => (
                                    <span key={tag} className="px-2 py-1 rounded-full text-[10px] bg-cyan-500/10 border border-cyan-400/20 text-cyan-200">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                            {buildDownloadHref(demoCorpus?.document?.download) && (
                                <a
                                    href={buildDownloadHref(demoCorpus?.document?.download)}
                                    target="_blank"
                                    rel="noreferrer"
                                    download={demoCorpus?.document?.download?.file_name}
                                    className="inline-flex items-center gap-2 text-xs text-cyan-100 px-3 py-2 rounded-lg bg-cyan-500/15 border border-cyan-400/25 hover:bg-cyan-500/20 transition-colors"
                                >
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v12m0 0 4-4m-4 4-4-4m-5 6h18" />
                                    </svg>
                                    {demoCorpus?.document?.download?.label ?? 'Download document'}
                                </a>
                            )}
                        </div>

                        <div className="border border-indigo-500/20 rounded-xl p-4 bg-indigo-950/20 space-y-2">
                            <div className="flex items-center justify-between">
                                <p className="text-xs uppercase tracking-widest text-indigo-300 font-semibold">Structured Dataset</p>
                            </div>
                            <p className="text-sm text-slate-100">{demoCorpus?.dataset?.display_name ?? 'Microsoft FY2025 Analyst Dataset'}</p>
                            <div className="space-y-2 pt-1">
                                {(demoCorpus?.dataset?.tables ?? []).map((table) => (
                                    <div key={table.table_name} className="flex items-center justify-between gap-3 p-2 rounded-lg bg-white/5 border border-indigo-500/20">
                                        <div className="flex items-center gap-3 min-w-0">
                                            <div className="p-2 bg-indigo-500/20 text-indigo-300 rounded-md">
                                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7h18M3 12h18M3 17h18" />
                                                </svg>
                                            </div>
                                            <span className="text-xs text-indigo-100 truncate">{table.table_name}</span>
                                        </div>
                                        {buildDownloadHref(table.download) && (
                                            <a
                                                href={buildDownloadHref(table.download)}
                                                target="_blank"
                                                rel="noreferrer"
                                                download={table.download?.file_name}
                                                className="shrink-0 inline-flex items-center gap-1 text-[11px] text-indigo-100 px-2 py-1 rounded-md bg-indigo-500/15 border border-indigo-400/20 hover:bg-indigo-500/20 transition-colors"
                                            >
                                                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v12m0 0 4-4m-4 4-4-4m-5 6h18" />
                                                </svg>
                                                {table.download?.label ?? 'Download'}
                                            </a>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="space-y-2">
                            <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Loaded Into Session</span>
                            {!isHydrated ? (
                                <div className="text-center py-4 text-xs font-mono text-slate-500 opacity-80 border border-white/5 rounded-lg bg-black/20">
                                    Click Load Demo Experience to activate the precomputed corpus.
                                </div>
                            ) : (
                                <div className="space-y-2 max-h-48 overflow-y-auto">
                                    {files.map((name) => (
                                        <div key={name} className="flex items-center gap-3 p-2 rounded-lg bg-white/5 border border-cyan-500/30">
                                            <div className="p-2 bg-cyan-500/20 text-cyan-300 rounded-md">
                                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 0 0 2-2V9.414a1 1 0 0 0-.293-.707l-5.414-5.414A1 1 0 0 0 12.586 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2z" />
                                                </svg>
                                            </div>
                                            <span className="text-xs text-cyan-100 truncate">{name}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
