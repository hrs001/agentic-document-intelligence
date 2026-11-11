'use client';

import React, { useState, useRef, useEffect, useLayoutEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { BrainMonitor, TelemetryEvent } from './BrainMonitor';
import { getApiBaseUrl } from '../lib/api';
import { getOrCreateSessionId } from '../lib/session';
import { createPortal } from 'react-dom';

interface Citation {
    index: number;
    source_type: 'internal_document' | 'graph_entity' | 'database' | 'web_search';
    display_label: string;
    source_file?: string;
    page_number?: number;
    section?: string;
    chunk_text?: string;
    entity_names?: string[];
    relationship_type?: string;
    table_name?: string;
    sql_query?: string;
    url?: string;
    title?: string;
}

interface GuardrailFinding {
    type?: string;
    match_preview?: string;
    action?: string;
}

interface GuardrailPayload {
    blocked?: boolean;
    reason?: string;
    risk_level?: string;
    warnings?: string[];
    attack_types?: string[];
    pii_findings?: GuardrailFinding[];
    sanitized_query?: string;
    policy_version?: string;
    confidence?: string;
}

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    citations?: Citation[];
    evaluation?: any;
    evaluationPending?: boolean;
    truncatedSubqueries?: boolean;
    subqueryCap?: number;
    resolvedQuery?: string;
    sourcesUsed?: string[];
    guardrails?: GuardrailPayload;
}

const DEFAULT_DEMO_QUERY = "Summarize the management commentary about AI demand, cloud momentum, and capital expenditure needs, and explain how those themes relate to Microsoft's strategy.";

const CitationHoverCard = ({ citation }: { citation: Citation }) => {
    const anchorRef = useRef<HTMLButtonElement>(null);
    const cardRef = useRef<HTMLDivElement>(null);
    const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const [open, setOpen] = useState(false);
    const [mounted, setMounted] = useState(false);
    const [coords, setCoords] = useState({ top: 0, left: 0, width: 340 });

    useEffect(() => setMounted(true), []);

    const clearCloseTimer = () => {
        if (closeTimerRef.current) {
            clearTimeout(closeTimerRef.current);
            closeTimerRef.current = null;
        }
    };

    const openCard = () => {
        clearCloseTimer();
        setOpen(true);
    };

    const closeCard = () => {
        clearCloseTimer();
        closeTimerRef.current = setTimeout(() => setOpen(false), 120);
    };

    const updatePosition = useCallback(() => {
        if (!anchorRef.current || !cardRef.current) return;
        const anchor = anchorRef.current.getBoundingClientRect();
        const card = cardRef.current.getBoundingClientRect();
        const margin = 12;
        const gap = 10;
        const width = Math.min(380, window.innerWidth - margin * 2);

        let left = anchor.left + anchor.width / 2 - width / 2;
        left = Math.max(margin, Math.min(left, window.innerWidth - width - margin));

        let top = anchor.top - card.height - gap;
        if (top < margin) {
            top = anchor.bottom + gap;
        }

        setCoords({ top, left, width });
    }, []);

    useLayoutEffect(() => {
        if (!open) return;
        updatePosition();
        const onViewportChange = () => updatePosition();
        window.addEventListener('resize', onViewportChange);
        window.addEventListener('scroll', onViewportChange, true);
        return () => {
            window.removeEventListener('resize', onViewportChange);
            window.removeEventListener('scroll', onViewportChange, true);
        };
    }, [open, updatePosition]);

    return (
        <>
            <button
                ref={anchorRef}
                type="button"
                onMouseEnter={openCard}
                onMouseLeave={closeCard}
                onFocus={openCard}
                onBlur={closeCard}
                className="inline-flex items-center px-1.5 py-0.5 rounded-md text-[10px] font-mono bg-cyan-500/15 border border-cyan-400/45 text-cyan-200 cursor-help hover:bg-cyan-500/25 transition-all mx-0.5"
                aria-label={`View citation ${citation.display_label}`}
            >
                {citation.display_label}
            </button>

            {mounted && open && createPortal(
                <div
                    ref={cardRef}
                    className="fixed z-[1100] rounded-xl border border-cyan-300/60 shadow-[0_20px_45px_rgba(2,6,23,0.85)] bg-[#020617] p-4"
                    style={{ top: coords.top, left: coords.left, width: coords.width }}
                    onMouseEnter={openCard}
                    onMouseLeave={closeCard}
                >
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-[10px] uppercase tracking-widest font-bold text-cyan-200">
                            {citation.source_type.replace('_', ' ')}
                        </span>
                        <span className="text-[10px] text-slate-200">#{citation.index}</span>
                    </div>

                    {citation.title && <h4 className="text-sm font-medium text-white mb-1 leading-tight">{citation.title}</h4>}

                    <div className="space-y-2 mt-1">
                        {citation.source_file && (
                            <p className="text-[11px] text-cyan-200 font-mono">
                                File: {citation.source_file} {citation.page_number ? `| pg. ${citation.page_number}` : ''}
                            </p>
                        )}

                        {citation.chunk_text && (
                            <p className="text-[12px] text-slate-100 leading-relaxed italic line-clamp-6">
                                "{citation.chunk_text}"
                            </p>
                        )}

                        {citation.url && (
                            <a href={citation.url} target="_blank" rel="noreferrer" className="text-[11px] text-cyan-300 hover:underline block truncate mt-1">
                                {citation.url}
                            </a>
                        )}

                        {citation.sql_query && (
                            <div className="mt-2 p-2 rounded-lg bg-slate-900 border border-cyan-500/20">
                                <code className="text-[10px] text-indigo-100 block font-mono">
                                    {citation.sql_query}
                                </code>
                            </div>
                        )}
                    </div>
                </div>,
                document.body
            )}
        </>
    );
};

const EvaluationDashboard = ({ evalData, pending }: { evalData?: any, pending?: boolean }) => {
    if (pending) {
        return (
            <div className="mt-6 pt-4 border-t border-white/5">
                <div className="px-2.5 py-2 rounded-md bg-slate-900 border border-cyan-300/25 text-[11px] text-slate-200">
                    Evaluation pending...
                </div>
            </div>
        );
    }
    if (!evalData) return null;
    if (typeof evalData.faithfulness === 'number' && evalData.faithfulness < 0) {
        return (
            <div className="mt-6 pt-4 border-t border-white/5">
                <div className="px-2.5 py-2 rounded-md bg-slate-900 border border-cyan-300/25 text-[11px] text-slate-200">
                    Evaluation unavailable.
                </div>
            </div>
        );
    }

    const metrics = [
        { label: 'Faithfulness', value: Math.max(0, Math.min(1, evalData.faithfulness)), color: 'bg-cyan-500' },
        { label: 'Answer Relevancy', value: Math.max(0, Math.min(1, evalData.answer_relevancy)), color: 'bg-emerald-500' },
        { label: 'Context Precision', value: Math.max(0, Math.min(1, evalData.context_precision)), color: 'bg-indigo-500' },
        { label: 'Citation Grounding', value: Math.max(0, Math.min(1, evalData.citation_grounding)), color: 'bg-amber-500' }
    ];

    return (
        <div className="mt-6 pt-4 border-t border-white/5 animate-in fade-in slide-in-from-bottom-2 duration-700">
            <div className="flex items-center gap-2 mb-3">
                <div className="px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20 text-[10px] text-cyan-400 uppercase tracking-widest font-bold">
                    RAGAS Evaluation: {evalData.overall_badge}
                </div>
            </div>
            <div className="grid grid-cols-1 gap-3">
                {metrics.map(m => (
                    <div key={m.label} className="space-y-1">
                        <div className="flex justify-between text-[10px] text-slate-400 uppercase tracking-tighter">
                            <span>{m.label}</span>
                            <span>{Math.round(m.value * 100)}%</span>
                        </div>
                        <div className="h-1 w-full bg-slate-900 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${m.value * 100}%` }}
                                transition={{ duration: 1, ease: 'easeOut' }}
                                className={`h-full ${m.color} shadow-[0_0_10px_rgba(6,182,212,0.5)]`}
                            />
                        </div>
                    </div>
                ))}
            </div>
            {evalData.reasoning && (
                <p className="mt-3 text-[11px] italic text-slate-100 leading-relaxed bg-slate-900 border border-cyan-300/35 rounded-md px-2.5 py-2">
                    "Judge Reasoning: {evalData.reasoning}"
                </p>
            )}
        </div>
    );
};

const GuardrailNotice = ({ guardrails }: { guardrails?: GuardrailPayload }) => {
    if (!guardrails) return null;
    const warnings = Array.isArray(guardrails.warnings) ? guardrails.warnings : [];
    const piiFindings = Array.isArray(guardrails.pii_findings) ? guardrails.pii_findings : [];
    const attackTypes = Array.isArray(guardrails.attack_types) ? guardrails.attack_types : [];
    if (!guardrails.blocked && warnings.length === 0 && piiFindings.length === 0 && attackTypes.length === 0) {
        return null;
    }

    const toneClass = guardrails.blocked
        ? 'bg-rose-500/8 border-rose-400/30 text-rose-100'
        : 'bg-amber-500/8 border-amber-400/30 text-amber-100';

    return (
        <div className={`mt-4 rounded-md border px-3 py-2 text-[11px] ${toneClass}`}>
            {guardrails.blocked ? (
                <p className="font-medium">Input blocked by safety guardrails.</p>
            ) : (
                <p className="font-medium">Input safety processing applied.</p>
            )}
            {warnings.map((warning) => (
                <p key={warning} className="mt-1 opacity-90">{warning}</p>
            ))}
            {piiFindings.length > 0 && (
                <p className="mt-1 opacity-90">
                    Redacted: {piiFindings.map((finding) => finding.type || 'sensitive_data').join(', ')}.
                </p>
            )}
            {attackTypes.length > 0 && (
                <p className="mt-1 opacity-90">
                    Signals: {attackTypes.join(', ')}.
                </p>
            )}
        </div>
    );
};

const ParsedResponse = ({ content, citations, evaluation, evaluationPending, truncatedSubqueries, subqueryCap, guardrails }: { content: string, citations?: Citation[], evaluation?: any, evaluationPending?: boolean, truncatedSubqueries?: boolean, subqueryCap?: number, guardrails?: GuardrailPayload }) => {
    if (!citations || citations.length === 0) {
        return (
            <div>
                <div className="prose prose-invert prose-sm max-w-none prose-a:text-cyan-400 prose-code:text-indigo-300">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
                </div>
                <GuardrailNotice guardrails={guardrails} />
            </div>
        );
    }

    // Robust Multi-Citation Parser
    // Matches patterns like [Evidence 1], [FLARE Doc 1], etc.
    const citationRegex = /(\[(?:Evidence|FLARE Doc) \d+\])/g;

    const parts = content.split(citationRegex);
    const elements: React.ReactNode[] = [];

    parts.forEach((part, i) => {
        if (i % 2 === 0) {
            // Text part: wrap in span to keep inline within the citation root
            if (part) elements.push(
                <ReactMarkdown
                    key={i}
                    remarkPlugins={[remarkGfm]}
                    components={{
                        p: ({ children }) => <span className="inline">{children}</span>,
                        div: ({ children }) => <span className="inline">{children}</span>
                    }}
                >
                    {part}
                </ReactMarkdown>
            );
        } else {
            // Citation part
            const citeLabel = part;
            const found = citations.find(c => c.display_label === citeLabel);
            if (found) {
                elements.push(<CitationHoverCard key={i} citation={found} />);
            } else {
                elements.push(<span key={i} className="text-cyan-500/50 font-mono text-[10px] bg-cyan-500/5 px-1 rounded">{citeLabel}</span>);
            }
        }
    });

    return (
        <div className="space-y-2">
            <div className="prose prose-invert prose-sm max-w-none prose-a:text-cyan-400 prose-code:text-indigo-300 citation-root">
                {elements}
            </div>
            {truncatedSubqueries && (
                <p className="text-[11px] text-amber-300/80 bg-amber-500/5 border border-amber-400/20 rounded-md px-2.5 py-2">
                    Complex question truncated to {subqueryCap ?? 3} sub-queries for latency control.
                </p>
            )}
            <GuardrailNotice guardrails={guardrails} />
            <EvaluationDashboard evalData={evaluation} pending={evaluationPending} />
        </div>
    );
};

export function ChatInterface() {
    const [sessionId, setSessionId] = useState('session-123');
    const [hasData, setHasData] = useState(false);
    const [questionsRemaining, setQuestionsRemaining] = useState(5);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [isDemoHydrating, setIsDemoHydrating] = useState(false);
    const [isHydratedState, setIsHydratedState] = useState(false);
    const [graphragMaxPages, setGraphragMaxPages] = useState(8);

    const [telemetryLogs, setTelemetryLogs] = useState<TelemetryEvent[]>([]);
    const [isTelemetryActive, setIsTelemetryActive] = useState(false);

    const endOfMessagesRef = useRef<HTMLDivElement>(null);
    const POLL_TIMEOUT_CHAT_MS = Number(process.env.NEXT_PUBLIC_POLL_TIMEOUT_CHAT_SEC ?? 300) * 1000;
    const POLL_TIMEOUT_HYDRATE_MS = Number(process.env.NEXT_PUBLIC_POLL_TIMEOUT_HYDRATE_SEC ?? 150) * 1000;
    const POLL_IDLE_TIMEOUT_MS = Number(process.env.NEXT_PUBLIC_POLL_IDLE_TIMEOUT_SEC ?? 20) * 1000;
    const POLL_TIMEOUT_EVALUATION_MS = Number(process.env.NEXT_PUBLIC_POLL_TIMEOUT_EVALUATION_SEC ?? 25) * 1000;

    const isValidJobId = (v: unknown): v is string =>
        typeof v === 'string' && /^[a-z0-9._-]{16,80}$/i.test(v);

    const buildConversationHistory = (historyMessages: Message[]) => {
        const turns: Array<{
            turn_id: string;
            user_query: string;
            resolved_query: string;
            answer_summary: string;
            sources_used: string[];
        }> = [];

        for (let index = 0; index < historyMessages.length - 1; index += 1) {
            const userMessage = historyMessages[index];
            const assistantMessage = historyMessages[index + 1];
            if (!userMessage || !assistantMessage) continue;
            if (userMessage.role !== 'user' || assistantMessage.role !== 'assistant') continue;

            turns.push({
                turn_id: assistantMessage.id,
                user_query: userMessage.content,
                resolved_query: assistantMessage.resolvedQuery || userMessage.content,
                answer_summary: assistantMessage.content.slice(0, 500),
                sources_used: assistantMessage.sourcesUsed || []
            });
        }

        return turns.slice(-10);
    };

    const pollJobUntilDone = async (
        apiHost: string,
        jobId: string,
        timeoutMs: number,
        onEvents: (events: any[]) => void
    ) => {
        if (!isValidJobId(jobId)) {
            throw new Error(`Invalid job id returned by server: ${String(jobId)}`);
        }
        let after = 0;
        const startedAt = Date.now();
        let lastProgressAt = Date.now();

        while (true) {
            if (Date.now() - startedAt > timeoutMs) {
                throw new Error(`Polling timeout reached (${Math.round(timeoutMs / 1000)}s).`);
            }
            if (Date.now() - lastProgressAt > POLL_IDLE_TIMEOUT_MS) {
                throw new Error(`No backend progress for ${Math.round(POLL_IDLE_TIMEOUT_MS / 1000)}s.`);
            }

            const res = await fetch(`${apiHost}/api/v1/jobs/${encodeURIComponent(jobId)}?after=${after}`);
            if (!res.ok) {
                throw new Error(`Polling request failed (${res.status}).`);
            }
            const data = await res.json();
            if (!data || typeof data !== 'object') {
                throw new Error('Malformed polling response.');
            }
            if (data.success === false || data.status === 'not_found') {
                throw new Error(data.error || data.hint || 'Job not found or invalid.');
            }

            const events = Array.isArray(data.events) ? data.events : [];
            const nextAfter = Number(data.next_after ?? after);
            if (!Number.isFinite(nextAfter) || nextAfter < after) {
                throw new Error('Malformed polling cursor (next_after).');
            }
            if (events.length > 0 || data.status === 'running') {
                lastProgressAt = Date.now();
            }
            after = nextAfter;
            onEvents(events);

            if (data.status === 'completed' || data.status === 'failed') {
                return data;
            }
            await new Promise((r) => setTimeout(r, 1000));
        }
    };

    const pollForEvaluationUpdate = async (
        apiHost: string,
        jobId: string,
        timeoutMs: number,
        onEvents: (events: any[]) => void
    ) => {
        let after = 0;
        const startedAt = Date.now();
        while (Date.now() - startedAt <= timeoutMs) {
            const res = await fetch(`${apiHost}/api/v1/jobs/${encodeURIComponent(jobId)}?after=${after}`);
            if (!res.ok) {
                throw new Error(`Evaluation polling request failed (${res.status}).`);
            }
            const data = await res.json();
            if (!data || typeof data !== 'object') {
                throw new Error('Malformed evaluation polling response.');
            }
            const events = Array.isArray(data.events) ? data.events : [];
            const nextAfter = Number(data.next_after ?? after);
            if (!Number.isFinite(nextAfter) || nextAfter < after) {
                throw new Error('Malformed evaluation polling cursor.');
            }
            after = nextAfter;
            onEvents(events);
            if (data.status === 'completed' && data.result?.evaluation_pending === false) {
                return data;
            }
            await new Promise((r) => setTimeout(r, 1000));
        }
        return null;
    };

    const refreshSessionStatus = async (sid: string) => {
        try {
            const apiHost = getApiBaseUrl();
            const res = await fetch(`${apiHost}/api/v1/session-status?session_id=${encodeURIComponent(sid)}`);
            if (!res.ok) {
                throw new Error(`Session status request failed (${res.status}).`);
            }
            const data = await res.json();
            setHasData(Boolean(data.has_any_data));
            setQuestionsRemaining(Number(data.questions_remaining ?? 0));
            setIsHydratedState(Boolean(data.has_any_data));
            setGraphragMaxPages(Number(data.graphrag_max_pages ?? 8));
        } catch (e) {
            console.error("Failed to fetch session status", e);
            setHasData(false);
            setIsHydratedState(false);
        }
    };

    // Global Telemetry Listener
    useEffect(() => {
        const sid = getOrCreateSessionId();
        setSessionId(sid);
        void refreshSessionStatus(sid);

        const handleTelemetry = (e: any) => {
            if (!e.detail) return;
            const { component, data } = e.detail;
            setTelemetryLogs(prev => [...prev, {
                id: Date.now().toString() + Math.random().toString(),
                timestamp: new Date().toISOString(),
                component: component || "System",
                data: data || ""
            }]);
            setIsTelemetryActive(true);
        };
        const handleDataLoaded = () => { void refreshSessionStatus(sid); };
        window.addEventListener('demoHydrated', handleDataLoaded);
        window.addEventListener('dataUploaded', handleDataLoaded);
        window.addEventListener('telemetryEvent', handleTelemetry);
        return () => {
            window.removeEventListener('telemetryEvent', handleTelemetry);
            window.removeEventListener('demoHydrated', handleDataLoaded);
            window.removeEventListener('dataUploaded', handleDataLoaded);
        };
    }, []);

    useEffect(() => {
        if (isHydratedState && !input.trim() && messages.length === 0) {
            setInput(DEFAULT_DEMO_QUERY);
        }
    }, [isHydratedState, input, messages.length]);

    const loadDemoExperience = async () => {
        setIsDemoHydrating(true);
        setIsTelemetryActive(true);

        try {
            const apiHost = getApiBaseUrl();
            const start = await fetch(`${apiHost}/api/v1/demo-hydrate/jobs?session_id=${encodeURIComponent(sessionId)}`, { method: 'POST' });
            const started = await start.json();
            if (!started.success || !started.job_id) {
                throw new Error(started.status || 'Failed to start hydration job');
            }

            const data = await pollJobUntilDone(apiHost, started.job_id, POLL_TIMEOUT_HYDRATE_MS, (events) => {
                for (const eventData of events) {
                    window.dispatchEvent(new CustomEvent('telemetryEvent', {
                        detail: {
                            component: eventData.component,
                            data: eventData.data
                        }
                    }));
                    if (eventData.component === "Orchestrator" && eventData.data === "Demo Hydration Complete.") {
                        window.dispatchEvent(new Event('demoHydrated'));
                        setIsHydratedState(true);
                    }
                }
            });
            if (data.status === 'failed') {
                throw new Error(data.error || 'Hydration job failed');
            }
            window.dispatchEvent(new Event('demoHydrated'));
            setInput(DEFAULT_DEMO_QUERY);
        } catch (e) {
            window.dispatchEvent(new CustomEvent('telemetryEvent', {
                detail: { component: "Error", data: `Connection failed: ${e instanceof Error ? e.message : String(e)}` }
            }));
            console.error(e);
        } finally {
            setIsDemoHydrating(false);
            await refreshSessionStatus(sessionId);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isProcessing) return;
        if (!hasData) {
            setMessages(prev => [...prev, {
                id: Date.now().toString(),
                role: 'assistant',
                content: 'Load Demo Experience before asking questions.'
            }]);
            return;
        }
        if (questionsRemaining <= 0) {
            setMessages(prev => [...prev, {
                id: Date.now().toString(),
                role: 'assistant',
                content: 'Session question limit reached. Start a new session to continue.'
            }]);
            return;
        }

        const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
        const requestHistory = buildConversationHistory(messages);
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsProcessing(true);

        setIsTelemetryActive(true);

        const assistantMsgId = (Date.now() + 1).toString();
        setMessages(prev => [...prev, { id: assistantMsgId, role: 'assistant', content: '' }]);

        try {
            const apiHost = getApiBaseUrl();

            const startRes = await fetch(`${apiHost}/api/v1/chat/jobs?session_id=${encodeURIComponent(sessionId)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: userMsg.content,
                    conversation_history: requestHistory
                }),
            });
            const started = await startRes.json();

            if (!startRes.ok || !started.success || !started.job_id) {
                const payload = started || {};
                setMessages(prev => prev.map(msg =>
                    msg.id === assistantMsgId ? {
                        ...msg,
                        content: payload.detail || payload.status || payload.answer || payload.error || 'Request blocked.',
                        citations: payload.citations,
                        evaluation: payload.evaluation,
                        evaluationPending: payload.evaluation_pending,
                        truncatedSubqueries: payload.truncated_subqueries,
                        subqueryCap: payload.subquery_cap,
                        guardrails: payload.guardrails
                    } : msg
                ));
                return;
            }

            const status = await pollJobUntilDone(apiHost, started.job_id, POLL_TIMEOUT_CHAT_MS, (events) => {
                for (const eventData of events) {
                    setTelemetryLogs(prev => [...prev, {
                        id: Date.now().toString() + Math.random().toString(),
                        timestamp: eventData.timestamp || new Date().toISOString(),
                        component: eventData.component || "System",
                        data: eventData.data || ""
                    }]);
                }
            });

            if (status.status === 'completed') {
                const payload = status.result || {};
                setMessages(prev => prev.map(msg =>
                    msg.id === assistantMsgId ? {
                        ...msg,
                        content: payload.answer || 'No answer generated.',
                        citations: payload.citations,
                        evaluation: payload.evaluation,
                        evaluationPending: Boolean(payload.evaluation_pending),
                        truncatedSubqueries: Boolean(payload.truncated_subqueries),
                        subqueryCap: Number(payload.subquery_cap ?? 3),
                        resolvedQuery: payload.resolved_query,
                        sourcesUsed: payload.sources_used,
                        guardrails: payload.guardrails
                    } : msg
                ));
                if (payload.evaluation_pending) {
                    const updated = await pollForEvaluationUpdate(apiHost, started.job_id, POLL_TIMEOUT_EVALUATION_MS, (events) => {
                        for (const eventData of events) {
                            setTelemetryLogs(prev => [...prev, {
                                id: Date.now().toString() + Math.random().toString(),
                                timestamp: eventData.timestamp || new Date().toISOString(),
                                component: eventData.component || "System",
                                data: eventData.data || ""
                            }]);
                        }
                    });
                    if (updated?.status === 'completed') {
                        const latest = updated.result || {};
                        setMessages(prev => prev.map(msg =>
                            msg.id === assistantMsgId ? {
                                ...msg,
                                evaluation: latest.evaluation,
                                evaluationPending: Boolean(latest.evaluation_pending),
                                citations: latest.citations || msg.citations
                            } : msg
                        ));
                    } else {
                        setMessages(prev => prev.map(msg =>
                            msg.id === assistantMsgId ? { ...msg, evaluationPending: false } : msg
                        ));
                    }
                }
            } else {
                setMessages(prev => prev.map(msg =>
                    msg.id === assistantMsgId ? { ...msg, content: `Error: ${status.error || 'Job failed.'}` } : msg
                ));
            }
        } catch (error) {
            console.error(error);
            const msg = error instanceof Error ? error.message : 'Connection failed.';
            setMessages(prev => prev.map(m =>
                m.id === assistantMsgId ? { ...m, content: `Error: ${msg}` } : m
            ));
            window.dispatchEvent(new CustomEvent('telemetryEvent', {
                detail: { component: "Error", data: `Chat polling aborted: ${msg}` }
            }));
        } finally {
            setIsProcessing(false);
            await refreshSessionStatus(sessionId);
        }
    };

    return (
        <div className="w-[35%] h-full flex flex-col glass-panel border-y-0 border-l-0 border-r border-r-cyan-500/20 shadow-[10px_0_30px_-5px_rgba(6,182,212,0.1)] z-10 relative overflow-hidden">

            {/* Header */}
            <div className="p-6 border-b border-white/5 bg-slate-950/20 flex-shrink-0">
                <h1 className="text-3xl font-light tracking-tight text-glow text-cyan-400">Knowledge Copilot</h1>
                <p className="text-sm font-mono text-cyan-600">Advanced Agentic RAG System</p>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide flex flex-col">
                {messages.length === 0 ? (
                    <div className="flex-1 flex flex-col items-center justify-center opacity-90 mt-10 transition-all duration-700">
                        {/* Empty Space - Jony Ive Style Demo Button OR Ready State */}
                        {!isHydratedState ? (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ duration: 0.8, ease: "easeOut" }}
                                className="backdrop-blur-3xl ring-1 ring-inset ring-white/10 rounded-[2rem] p-8 flex flex-col items-center justify-center shadow-2xl space-y-4 max-w-sm w-full relative overflow-hidden group min-h-[380px]"
                            >
                                {/* Liquid animated dynamic background - Absolute corner-to-corner coverage */}
                                <div className="absolute inset-0 bg-slate-950 z-[-2] pointer-events-none m-0 mb-0"></div>

                                <div className="absolute inset-0 opacity-80 group-hover:opacity-100 transition-opacity duration-1000 z-[-1] pointer-events-none overflow-hidden m-0 mb-0" style={{ marginBottom: '0px' }}>
                                    {/* 4 Corners Area-Filling Gradients - Ensures no dark gaps at bottom */}
                                    <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_0%_0%,rgba(6,182,212,0.15),transparent_50%)]"></div>
                                    <div className="absolute top-0 right-0 w-full h-full bg-[radial-gradient(circle_at_100%_0%,rgba(79,70,229,0.15),transparent_50%)]"></div>
                                    <div className="absolute bottom-0 left-0 w-full h-full bg-[radial-gradient(circle_at_0%_100%,rgba(6,182,212,0.2),transparent_50%)]"></div>
                                    <div className="absolute bottom-0 right-0 w-full h-full bg-[radial-gradient(circle_at_100%_100%,rgba(79,70,229,0.2),transparent_50%)]"></div>

                                    {/* Central wash for overall luminosity */}
                                    <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 via-transparent to-cyan-500/10 pointer-events-none"></div>

                                    {/* Animated liquid highlights */}
                                    <div className="absolute top-[-20%] left-[-10%] w-[140%] h-[140%] bg-[conic-gradient(from_0deg_at_50%_50%,rgba(6,182,212,0.05)_0deg,transparent_90deg,rgba(79,70,229,0.05)_180deg,transparent_270deg,rgba(6,182,212,0.05)_360deg)] animate-[spin_20s_linear_infinite]"></div>
                                </div>

                                <div className="z-10 flex flex-col items-center justify-center w-full h-full relative">
                                    <svg className="w-12 h-12 text-cyan-300/80 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                                    </svg>

                                    <h2 className="text-xl font-medium text-slate-100 tracking-wide text-center">Interactive Quickstart</h2>
                                    <p className="text-[13px] text-slate-400 text-center leading-relaxed mt-2 mb-6 px-2">
                                        Experience the power of the Agentic RAG Brain instantly.
                                        The fixed Microsoft FY2025 corpus is precomputed and ready for grounded retrieval, GraphRAG, and SQL analysis.
                                    </p>

                                    <div className="w-full flex-grow flex flex-col justify-end items-center">
                                        <button
                                            onClick={loadDemoExperience}
                                            disabled={isDemoHydrating}
                                            className="relative overflow-hidden w-full bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400/30 text-cyan-100 rounded-2xl py-3.5 px-6 text-[15px] font-medium transition-all duration-500 shadow-[0_0_20px_-5px_rgba(6,182,212,0.3)] hover:shadow-[0_0_30px_-5px_rgba(6,182,212,0.5)] disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-[1.02] active:scale-[0.98]"
                                        >
                                            {isDemoHydrating ? (
                                                <span className="flex items-center justify-center gap-2">
                                                    <svg className="w-5 h-5 animate-spin text-cyan-300" fill="none" viewBox="0 0 24 24">
                                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"></circle>
                                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                    </svg>
                                                    Preparing Demo Corpus...
                                                </span>
                                            ) : (
                                                "Load Demo Experience"
                                            )}
                                        </button>

                                        <div className="mt-4 text-center">
                                            <p className="text-[10px] text-slate-500 uppercase tracking-widest font-medium max-w-[200px] leading-relaxed">
                                                Precomputed graph spans through page {graphragMaxPages}.<br />
                                                Full fixed corpus available for Q&A.
                                            </p>
                                        </div>

                                        <div className={`transition-all duration-700 h-[30px] flex items-center justify-center overflow-hidden mt-2 ${isDemoHydrating ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 pointer-events-none'}`}>
                                            <p className="text-[11px] text-cyan-300/70 uppercase tracking-widest font-mono text-center">
                                                Note: First query may take longer because the full runtime pipeline is real.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        ) : (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="text-center space-y-3"
                            >
                                <div className="inline-flex items-center justify-center p-3 rounded-full bg-emerald-500/10 border border-emerald-500/20 shadow-[0_0_20px_-5px_rgba(16,185,129,0.3)] mb-2">
                                    <svg className="w-6 h-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                    </svg>
                                </div>
                                <h3 className="text-lg font-medium text-slate-200 tracking-wide">Brain Hydrated & Ready</h3>
                                <p className="text-sm font-light text-slate-400 max-w-sm leading-relaxed">
                                    The Agentic pipeline is fully armed with your data context. You may now query the system.
                                </p>
                            </motion.div>
                        )}
                    </div>
                ) : (
                    messages.map((m) => (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            key={m.id}
                            className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}
                        >
                            <div className={`max-w-[95%] rounded-2xl p-4 ${m.role === 'user' ? 'bg-cyan-600/20 border border-cyan-500/30 text-cyan-100 rounded-tr-sm shadow-[0_0_15px_-3px_rgba(6,182,212,0.2)]' : 'glass-button rounded-tl-sm w-full'}`}>
                                {m.role === 'assistant' ? (
                                    <ParsedResponse
                                        content={m.content || '...'}
                                        citations={m.citations}
                                        evaluation={m.evaluation}
                                        evaluationPending={m.evaluationPending}
                                        truncatedSubqueries={m.truncatedSubqueries}
                                        subqueryCap={m.subqueryCap}
                                        guardrails={m.guardrails}
                                    />
                                ) : (
                                    <span className="text-sm">{m.content}</span>
                                )}
                            </div>
                        </motion.div>
                    ))
                )}
                <div ref={endOfMessagesRef} />
            </div>

            {/* Footer Area with Telemetry & Input */}
            <div className="p-4 flex-shrink-0 flex flex-col justify-end">
                <BrainMonitor
                    isActive={isTelemetryActive}
                    onToggle={() => setIsTelemetryActive(!isTelemetryActive)}
                    logs={telemetryLogs}
                />

                {/* Input Box */}
                <form
                    onSubmit={handleSubmit}
                    className="w-full relative z-30"
                >
                    {!hasData && (
                        <p className="text-[11px] text-amber-300/80 mb-2 px-2">
                            Load Demo Experience before asking questions.
                        </p>
                    )}
                    {questionsRemaining <= 0 && (
                        <p className="text-[11px] text-rose-300/80 mb-2 px-2">
                            Question limit reached for this session.
                        </p>
                    )}
                    {hasData && (
                        <p className="text-[10px] text-slate-500 mb-2 px-2">
                            Questions remaining: {questionsRemaining}. Complex questions may be capped at 3 sub-queries for latency control.
                        </p>
                    )}
                    <div className="glass-panel flex items-end rounded-[24px] p-2 border border-white/10 shadow-[0_8px_32px_0_rgba(0,0,0,0.4)] bg-gradient-to-b from-[#020617]/90 to-[#020617]/95 hover:bg-[#020617]/100 transition-all duration-500 focus-within:border-cyan-500/50 focus-within:shadow-[0_8px_32px_0_rgba(6,182,212,0.25)]">
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSubmit(e as unknown as React.FormEvent);
                                }
                            }}
                            disabled={isProcessing || isDemoHydrating || !hasData || questionsRemaining <= 0}
                            placeholder={
                                isDemoHydrating
                                    ? "Hydrating Brain (Please Wait)..."
                                    : !hasData
                                        ? "Load demo experience first..."
                                        : questionsRemaining <= 0
                                            ? "Session question limit reached..."
                                            : isProcessing
                                                ? "Brain is processing..."
                                                : "Enter your query here..."
                            }
                            className="w-full bg-transparent outline-none px-5 py-4 text-[15px] font-light leading-relaxed text-slate-100 placeholder:text-slate-500 resize-none min-h-[70px] max-h-[250px] scrollbar-hide rounded-2xl transition-all duration-300 focus:bg-white/5"
                            rows={Math.min(8, Math.max(2, input.split('\n').length))}
                        />
                        <button
                            type="submit"
                            disabled={isProcessing || isDemoHydrating || !input.trim() || !hasData || questionsRemaining <= 0}
                            className="glass-button p-4 mx-2 mb-2 rounded-[18px] text-cyan-400 disabled:opacity-30 transform hover:scale-105 active:scale-95 transition-all self-end shadow-[0_0_15px_-3px_rgba(6,182,212,0.3)] hover:shadow-[0_0_20px_-3px_rgba(6,182,212,0.5)] bg-slate-900/50"
                        >
                            {isProcessing ? (
                                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                            ) : (
                                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                                </svg>
                            )}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
