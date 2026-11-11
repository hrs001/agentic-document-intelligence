'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
    ReactFlow,
    Controls,
    Background,
    applyNodeChanges,
    applyEdgeChanges,
    Node,
    Edge,
    NodeChange,
    EdgeChange,
    ConnectionLineType,
    useReactFlow,
    ReactFlowProvider
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { motion } from 'framer-motion';
import { getApiBaseUrl } from '../lib/api';
import { getOrCreateSessionId } from '../lib/session';

const API_URL = getApiBaseUrl();

function GraphVisualizerContent() {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);

    // We store the full pristine graph in refs to allow unhiding logic
    const fullNodesRef = useRef<Node[]>([]);
    const fullEdgesRef = useRef<Edge[]>([]);
    const [search, setSearch] = useState('');
    const [isHydrated, setIsHydrated] = useState(false);
    const [dismissHud, setDismissHud] = useState(false);
    const [graphragMaxPages, setGraphragMaxPages] = useState(8);

    const { setCenter } = useReactFlow();

    useEffect(() => {
        const sid = getOrCreateSessionId();
        const fetchSessionStatus = async () => {
            try {
                const res = await fetch(`${API_URL}/api/v1/session-status?session_id=${encodeURIComponent(sid)}`);
                if (!res.ok) {
                    throw new Error(`Session status request failed (${res.status}).`);
                }
                const data = await res.json();
                setGraphragMaxPages(Number(data.graphrag_max_pages ?? 8));
            } catch (e) {
                // keep defaults on transient failures
            }
        };

        const fetchGraph = async () => {
            try {
                // Fetch real RAG graph data
                const res = await fetch(`${API_URL}/api/v1/graph?session_id=${encodeURIComponent(sid)}`);
                if (!res.ok) {
                    throw new Error(`Graph request failed (${res.status}).`);
                }
                const data = await res.json();
                if (!data || !data.nodes || data.nodes.length === 0) return;

                // Layout engine processing (Assign positions)
                const layoutNodes: Node[] = [];
                const layoutEdges: Edge[] = [];

                // Identify the most connected node as the "core" central hub
                const degreeMap: Record<string, number> = {};
                data.nodes.forEach((n: any) => { degreeMap[n.id] = 0; });
                data.edges.forEach((e: any) => {
                    if (degreeMap[e.source] !== undefined) degreeMap[e.source]++;
                    if (degreeMap[e.target] !== undefined) degreeMap[e.target]++;
                });

                let maxDegreeId = data.nodes[0].id;
                let maxDegree = -1;
                Object.keys(degreeMap).forEach(k => {
                    if (degreeMap[k] > maxDegree) { maxDegree = degreeMap[k]; maxDegreeId = k; }
                });

                // Breadth-First traversal to assign distances from core (hop count)
                const distances: Record<string, number> = { [maxDegreeId]: 0 };
                const queue: string[] = [maxDegreeId];
                while (queue.length > 0) {
                    const current = queue.shift()!;
                    data.edges.forEach((e: any) => {
                        const neighbor = e.source === current ? e.target : (e.target === current ? e.source : null);
                        if (neighbor && distances[neighbor] === undefined) {
                            distances[neighbor] = distances[current] + 1;
                            queue.push(neighbor);
                        }
                    });
                }

                const d1Nodes = data.nodes.filter((n: any) => distances[n.id] === 1);
                const d2Nodes = data.nodes.filter((n: any) => distances[n.id] === 2);

                const centerX = 800;
                const centerY = 500;

                data.nodes.forEach((n: any) => {
                    let rx = centerX, ry = centerY;
                    const d = distances[n.id];
                    let isHidden = true;

                    if (d === 0) {
                        isHidden = false;
                    } else if (d === 1) {
                        const idx = d1Nodes.findIndex((x: any) => x.id === n.id);
                        const angle = (idx / d1Nodes.length) * Math.PI * 2;
                        rx = centerX + Math.cos(angle) * 300;
                        ry = centerY + Math.sin(angle) * 300;
                        isHidden = false;
                    } else if (d === 2) {
                        const idx = d2Nodes.findIndex((x: any) => x.id === n.id);
                        const angle = (idx / d2Nodes.length) * Math.PI * 2;
                        rx = centerX + Math.cos(angle) * 600;
                        ry = centerY + Math.sin(angle) * 600;
                        isHidden = true;
                    } else {
                        rx = centerX + (Math.random() - 0.5) * 1500;
                        ry = centerY + (Math.random() - 0.5) * 1500;
                        isHidden = true;
                    }

                    layoutNodes.push({
                        id: n.id,
                        position: { x: rx, y: ry },
                        data: {
                            label: n.name || n.id,
                            type: n.type
                        },
                        className: d === 0 ? 'glass-node core-node' : 'glass-node',
                        hidden: isHidden
                    });
                });

                data.edges.forEach((e: any, idx: number) => {
                    const srcHidden = layoutNodes.find(n => n.id === e.source)?.hidden ?? true;
                    const tgtHidden = layoutNodes.find(n => n.id === e.target)?.hidden ?? true;

                    layoutEdges.push({
                        id: `e${idx}`,
                        source: e.source,
                        target: e.target,
                        type: 'bezier',
                        animated: true,
                        style: { stroke: '#06b6d4', opacity: 0.5, strokeWidth: 1.5 },
                        label: e.relationship || 'RELATED',
                        labelBgStyle: { fill: '#0f172a' },
                        labelStyle: { fill: '#fff', fontSize: 10 },
                        hidden: srcHidden || tgtHidden
                    });
                });

                fullNodesRef.current = layoutNodes;
                fullEdgesRef.current = layoutEdges;
                setNodes(layoutNodes);
                setEdges(layoutEdges);
            } catch (err) {
                console.error("Graph fetch failed:", err);
            }
        };

        const handleHydration = () => {
            setIsHydrated(true);
            fetchSessionStatus();
            setTimeout(fetchGraph, 1000);
        };
        fetchSessionStatus();
        window.addEventListener('demoHydrated', handleHydration);
        return () => {
            window.removeEventListener('demoHydrated', handleHydration);
        };
    }, []);

    // Search Effect
    useEffect(() => {
        if (!isHydrated || fullNodesRef.current.length === 0) return;

        if (!search.trim()) {
            setNodes(nds => nds.map(n => ({
                ...n,
                hidden: fullNodesRef.current.find(fn => fn.id === n.id)?.hidden ?? true,
                className: n.className?.replace(' highlight-node', '')
            })));
            setEdges(eds => eds.map(e => ({
                ...e,
                hidden: fullEdgesRef.current.find(fe => fe.id === e.id)?.hidden ?? true
            })));
            return;
        }

        const term = search.toLowerCase();
        const matchedNodeIds = new Set<string>();
        const contextNodeIds = new Set<string>();
        let bestMatch: Node | null = null;

        fullNodesRef.current.forEach(n => {
            const name = (n.data?.label as string || "").toLowerCase();
            if (name.includes(term)) {
                matchedNodeIds.add(n.id);
                contextNodeIds.add(n.id);
                if (!bestMatch || name.length < (bestMatch.data?.label as string).length) {
                    bestMatch = n;
                }
                fullEdgesRef.current.forEach(e => {
                    if (e.source === n.id) contextNodeIds.add(e.target);
                    if (e.target === n.id) contextNodeIds.add(e.source);
                });
            }
        });

        if (bestMatch) {
            setCenter((bestMatch as Node).position.x, (bestMatch as Node).position.y, { zoom: 1.2, duration: 800 });
        }

        setNodes(nds => nds.map(n => ({
            ...n,
            hidden: !contextNodeIds.has(n.id),
            className: matchedNodeIds.has(n.id)
                ? (n.className?.includes('highlight-node') ? n.className : `${n.className} highlight-node`)
                : n.className?.replace(' highlight-node', '')
        })));

        setEdges(eds => eds.map(e => {
            const showEdge = contextNodeIds.has(e.source) && contextNodeIds.has(e.target) && (matchedNodeIds.has(e.source) || matchedNodeIds.has(e.target));
            return {
                ...e,
                hidden: !showEdge,
                animated: showEdge,
                style: showEdge ? { stroke: '#10b981', opacity: 0.8, strokeWidth: 2 } : e.style
            };
        }));
    }, [search, isHydrated, setCenter]);

    const onNodesChange = useCallback(
        (changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
        []
    );
    const onEdgesChange = useCallback(
        (changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
        []
    );

    const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
        const connectedEdges = fullEdgesRef.current.filter(e => e.source === node.id || e.target === node.id);
        const nodeIdsToUnhide = new Set<string>();
        connectedEdges.forEach(e => {
            nodeIdsToUnhide.add(e.source);
            nodeIdsToUnhide.add(e.target);
        });

        setNodes(nds => nds.map(n => {
            if (nodeIdsToUnhide.has(n.id)) return { ...n, hidden: false };
            return n;
        }));

        setEdges(eds => eds.map(e => {
            if (e.source === node.id || e.target === node.id) {
                return { ...e, hidden: false, animated: true, style: { stroke: '#10b981', opacity: 0.8, strokeWidth: 2 } };
            }
            return e;
        }));
    }, []);

    const tidyNodes = useCallback(() => {
        setNodes(nds => {
            const visibleNodes = nds.filter(n => !n.hidden);
            if (visibleNodes.length === 0) return nds;

            const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
            const visibleEdges = edges.filter(e => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target));

            const degreeMap: Record<string, number> = {};
            visibleNodeIds.forEach(id => degreeMap[id] = 0);
            visibleEdges.forEach(e => {
                degreeMap[e.source]++;
                degreeMap[e.target]++;
            });

            let coreId = visibleNodes[0].id;
            let maxDeg = -1;
            Object.entries(degreeMap).forEach(([id, deg]) => {
                if (deg > maxDeg) { maxDeg = deg; coreId = id; }
            });

            const distances: Record<string, number> = { [coreId]: 0 };
            const q = [coreId];
            while (q.length > 0) {
                const curr = q.shift()!;
                visibleEdges.forEach(e => {
                    const neighbor = e.source === curr ? e.target : (e.target === curr ? e.source : null);
                    if (neighbor && distances[neighbor] === undefined) {
                        distances[neighbor] = distances[curr] + 1;
                        q.push(neighbor);
                    }
                });
            }

            const centerX = 800;
            const centerY = 500;

            return nds.map(n => {
                if (n.hidden) return n;
                const d = distances[n.id] ?? 3;
                let rx = centerX, ry = centerY;

                if (d === 0) {
                    rx = centerX; ry = centerY;
                } else {
                    const layerNodes = visibleNodes.filter(vn => (distances[vn.id] ?? 3) === d);
                    const idx = layerNodes.findIndex(vn => vn.id === n.id);
                    const angle = (idx / (layerNodes.length || 1)) * Math.PI * 2;
                    const radius = d * 250;
                    rx = centerX + Math.cos(angle) * radius;
                    ry = centerY + Math.sin(angle) * radius;
                }

                const pos = { x: rx, y: ry };

                // Refocus if there's a search match
                if (search.trim()) {
                    const term = search.toLowerCase();
                    const name = (n.data?.label as string || "").toLowerCase();
                    if (name.includes(term)) {
                        // Refocus on this match
                        setCenter(rx, ry, { zoom: 1.2, duration: 800 });
                    }
                }

                return { ...n, position: pos };
            });
        });

        window.dispatchEvent(new CustomEvent('telemetryEvent', {
            detail: { component: "GraphExplorer", data: "Tidying visible nodes into orbital formation..." }
        }));
    }, [edges, search, setCenter]);

    return (
        <div className="w-[65%] h-full relative">
            {/* Legend & HUD */}
            <div className="absolute top-6 left-6 z-20 flex flex-col gap-4 pointer-events-none">

                <div className="glass-panel px-4 py-3 rounded-2xl flex items-center gap-4 pointer-events-auto">
                    <div className="flex items-center gap-2 border-r border-white/10 pr-4">
                        <div className={`w-2 h-2 rounded-full ${isHydrated ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`}></div>
                        <span className={`text-xs font-mono ${isHydrated ? 'text-emerald-400' : 'text-slate-500'}`}>
                            {isHydrated ? 'GRAPH ACTIVE' : 'GRAPH STANDBY'}
                        </span>
                    </div>
                    <div className="relative">
                        <input
                            type="text"
                            placeholder="Search entities..."
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            disabled={!isHydrated}
                            className="bg-white/5 border border-cyan-500/30 rounded-lg px-3 py-1 text-xs text-slate-200 outline-none w-48 placeholder:text-slate-500 focus:border-cyan-400 transition-colors disabled:opacity-50"
                        />
                    </div>
                </div>

                {isHydrated && !dismissHud && (
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.5 }}
                        className="flex flex-col gap-2 bg-slate-900/60 backdrop-blur-xl border border-cyan-500/20 px-5 py-3 rounded-2xl shadow-[0_5px_25px_-5px_rgba(0,0,0,0.5)] self-start max-w-sm pointer-events-auto"
                    >
                        <div className="flex items-center justify-between">
                            <span className="text-[10px] text-cyan-300 uppercase tracking-widest font-mono font-bold">GraphRAG Engine Ready</span>
                            <button onClick={() => setDismissHud(true)} className="text-slate-500 hover:text-white transition-colors focus:outline-none ml-4">
                                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                            </button>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400"></div>
                                <span className="text-[11px] text-slate-200 font-medium">Mode: Precomputed Kuzu Graph</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-cyan-400"></div>
                                <span className="text-[11px] text-slate-300">Coverage: Graph evidence through page {graphragMaxPages}</span>
                            </div>
                            <div className="text-[10px] text-slate-500 font-light mt-1 pl-3.5 italic">
                                {`Click nodes to explore relationships grounded in the precomputed Microsoft FY2025 graph.`}
                            </div>
                            <div className="text-[10px] text-emerald-400/80 font-medium mt-0.5 pl-3.5 flex items-center gap-1.5 animate-pulse">
                                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                                Tip: Click on nodes to browse more connections!
                            </div>
                        </div>
                    </motion.div>
                )}
            </div>

            {/* Floating Action Buttons (Bottom Right) */}
            <div className="absolute bottom-6 right-6 z-20 flex flex-col items-end gap-3">
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={tidyNodes}
                    className="glass-panel flex items-center gap-2 border-cyan-500/30 px-4 py-2.5 text-cyan-100 hover:bg-cyan-500/10 transition-colors shadow-[0_5px_20px_-5px_rgba(6,182,212,0.3)] group"
                >
                    <svg className="w-4 h-4 text-cyan-400 group-hover:rotate-180 transition-transform duration-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    <span className="text-[11px] font-medium tracking-wide uppercase">Tidy Layout</span>
                </motion.button>
            </div>

            {!isHydrated && nodes.length === 0 && (
                <div className="absolute inset-0 flex flex-col items-center justify-center z-10 text-slate-500 font-mono text-sm opacity-50 gap-4 pointer-events-none transition-opacity duration-500">
                    <svg className="w-12 h-12 text-slate-600 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                    </svg>
                    <p>Interactive Graph Environment Standing By</p>
                </div>
            )}

            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onNodeClick={onNodeClick}
                connectionLineType={ConnectionLineType.SmoothStep}
                fitView
                fitViewOptions={{ padding: 0.2 }}
                minZoom={0.1}
                className="bg-transparent"
                proOptions={{ hideAttribution: true }}
            >
                <Background color="#06b6d4" gap={30} size={1} className={`transition-opacity duration-1000 ${isHydrated ? 'opacity-20' : 'opacity-5'}`} />
                <Controls className="glass-panel fill-cyan-400 border-none rounded-lg overflow-hidden [&>button]:border-b-white/5 [&>button]:bg-transparent hover:[&>button]:bg-white/10" />
            </ReactFlow>

            <style jsx global>{`
        .glass-node {
          background: rgba(255, 255, 255, 0.05) !important;
          backdrop-filter: blur(10px) !important;
          border: 1px solid rgba(6, 182, 212, 0.3) !important;
          border-radius: 8px !important;
          color: #f8fafc !important;
          box-shadow: 0 4px 20px -5px rgba(6, 182, 212, 0.1) !important;
          padding: 8px 16px !important;
          font-family: 'Inter', sans-serif !important;
          font-size: 11px !important;
          letter-spacing: 0.5px;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          animation: nodeAppearOp 0.7s ease-out forwards;
        }
        .glass-node:hover {
          border-color: rgba(6, 182, 212, 0.8) !important;
          box-shadow: 0 4px 25px -5px rgba(6, 182, 212, 0.4) !important;
          z-index: 1000;
        }
        .core-node {
          border-color: rgba(6, 182, 212, 0.8) !important;
          box-shadow: 0 0 30px 0 rgba(6, 182, 212, 0.3) !important;
          font-size: 14px !important;
          font-weight: 600 !important;
          padding: 12px 24px !important;
        }
        @keyframes nodeAppearOp {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .react-flow__edge-textbg {
          rx: 4px;
          ry: 4px;
          fill-opacity: 0.9;
        }
        .highlight-node {
          border-color: #10b981 !important;
          box-shadow: 0 0 25px 0 rgba(16, 185, 129, 0.6) !important;
          transform: scale(1.1);
          z-index: 2000 !important;
        }
      `}</style>
        </div>
    );
}

export function GraphVisualizer() {
    return (
        <ReactFlowProvider>
            <GraphVisualizerContent />
        </ReactFlowProvider>
    )
}
