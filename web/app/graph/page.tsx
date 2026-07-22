"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  Share2,
  Loader2,
  AlertCircle,
  RefreshCw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Shuffle,
  X,
  Hash,
  TrendingUp,
  Zap,
  GitBranch,
  Layers,
  Tag,
  ShieldAlert,
  FileText,
  Link as LinkIcon,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { apiUrl, getAuthHeaders } from "@/lib/api";
import { getTranslation, type Language } from "@/lib/i18n";
import { useGlobal } from "@/context/GlobalContext";
import {
  ClusterGraph,
  LIFECYCLE_COLORS,
  lifecycleColor,
  type GraphNode,
  type GraphEdge,
  type ClusterGraphHandle,
  type SelectedEdge,
} from "@/components/ClusterGraph";

// === Types ===

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface RelatedClusterRef {
  target_cluster_id: string;
  weight: number;
  source?: string;
}

interface Constraint {
  condition?: string;
  severity?: string;
  description?: string;
}

interface EvidenceSnippet {
  snippet?: string;
  score?: number;
  reasoning?: string;
}

interface EvidenceUnit {
  doc_id?: string;
  file_or_url?: string;
  summary?: string;
  is_found?: boolean;
  snippets?: EvidenceSnippet[];
}

interface ClusterDetail {
  id: string;
  name: string;
  description?: string | string[];
  content?: string | string[];
  scripts?: string[] | null;
  resources?: Array<{ type: string; value: string }> | null;
  evidences?: EvidenceUnit[];
  patterns?: string[];
  constraints?: Constraint[];
  confidence?: number | null;
  abstraction_level?: string | null;
  landmark_potential?: number | null;
  hotness?: number | null;
  lifecycle?: string | null;
  create_time?: string;
  last_modified?: string;
  version?: number;
  related_clusters?: RelatedClusterRef[];
  search_results?: string[];
  queries?: string[];
  merge_count?: number;
}

type LifecycleKey =
  | "STABLE"
  | "EMERGING"
  | "CONTESTED"
  | "DEPRECATED"
  | "META";

const ALL_LIFECYCLES: LifecycleKey[] = [
  "STABLE",
  "EMERGING",
  "CONTESTED",
  "DEPRECATED",
  "META",
];

function pct(v?: number | null): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(0)}%`;
}

function joinField(v?: string | string[] | null): string {
  if (v == null) return "";
  return Array.isArray(v) ? v.join("\n\n") : v;
}

export default function GraphPage() {
  const { uiSettings, theme } = useGlobal();
  const t = (key: string) =>
    getTranslation((uiSettings?.language || "en") as Language, key);

  const [graph, setGraph] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ClusterDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<SelectedEdge | null>(null);

  const [weightThreshold, setWeightThreshold] = useState(0);
  const [charge, setCharge] = useState(200);
  const [centeringSlider, setCenteringSlider] = useState(100); // centering = slider / 1000
  const [enabledLifecycles, setEnabledLifecycles] = useState<Set<LifecycleKey>>(
    () => new Set(ALL_LIFECYCLES),
  );

  const graphRef = useRef<ClusterGraphHandle>(null);

  // === Data fetching ===

  const fetchGraph = useCallback(async (showLoading: boolean = true) => {
    try {
      if (showLoading) setLoading(true);
      else setRefreshing(true);
      setError("");

      const res = await fetch(apiUrl("/api/v1/knowledge/graph"), {
        headers: { ...getAuthHeaders() },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (json.success && json.data) {
        setGraph(json.data as GraphData);
      } else {
        throw new Error(json.error || "Failed to load graph");
      }
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchGraph(true);
  }, [fetchGraph]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await fetch(apiUrl("/api/v1/knowledge/refresh"), {
        method: "POST",
        headers: { ...getAuthHeaders() },
      });
    } catch {
      // Non-critical: backend auto-reload still works
    }
    await fetchGraph(false);
  };

  // Fetch cluster detail on selection
  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    setDetail(null);
    fetch(apiUrl(`/api/v1/knowledge/clusters/${encodeURIComponent(selectedId)}`), {
      headers: { ...getAuthHeaders() },
    })
      .then((r) => r.json())
      .then((json) => {
        if (cancelled) return;
        if (json.success && json.data) setDetail(json.data as ClusterDetail);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const handleSelect = useCallback((id: string | null) => {
    setSelectedId(id);
  }, []);

  const handleSelectEdge = useCallback((edge: SelectedEdge | null) => {
    setSelectedEdge(edge);
    if (edge) setSelectedId(null);
  }, []);

  const handleFocusRelated = useCallback(
    (id: string) => {
      setSelectedId(id);
      graphRef.current?.focusNode(id);
    },
    [],
  );

  // === Derived filtered data (memoized so hover/selection re-renders
  // don't change array identity and re-trigger the graph layout) ===

  const { filteredNodes, filteredEdges } = useMemo(() => {
    if (!graph) return { filteredNodes: [], filteredEdges: [] };
    const fnodes = graph.nodes.filter((n) =>
      enabledLifecycles.has((n.lifecycle as LifecycleKey) || "EMERGING"),
    );
    const fids = new Set(fnodes.map((n) => n.id));
    const fedges = graph.edges.filter(
      (e) =>
        fids.has(e.source) &&
        fids.has(e.target) &&
        (typeof e.weight !== "number" || e.weight >= weightThreshold),
    );
    return { filteredNodes: fnodes, filteredEdges: fedges };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph, enabledLifecycles, weightThreshold]);

  // lifecycle counts (from full dataset, for legend)
  const lifecycleCounts: Record<string, number> = {};
  for (const n of graph?.nodes ?? []) {
    const lc = (n.lifecycle as string) || "EMERGING";
    lifecycleCounts[lc] = (lifecycleCounts[lc] || 0) + 1;
  }

  const toggleLifecycle = (lc: LifecycleKey) => {
    setEnabledLifecycles((prev) => {
      const next = new Set(prev);
      if (next.has(lc)) {
        if (next.size > 1) next.delete(lc); // keep at least one
      } else {
        next.add(lc);
      }
      return next;
    });
  };

  // === Render ===

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-blue-950/20">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-slate-600 dark:text-slate-400">
            {t("Loading clusters...")}
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-blue-950/20">
        <div className="text-center max-w-md">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
            {t("Error")}
          </h2>
          <p className="text-slate-600 dark:text-slate-400 mb-4">{error}</p>
          <button
            onClick={() => fetchGraph(true)}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm font-medium"
          >
            {t("Retry")}
          </button>
        </div>
      </div>
    );
  }

  const hasData = (graph?.nodes.length ?? 0) > 0;

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-blue-950/20">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
        <div className="px-6 py-3">
          {/* Row 1: title + counts + refresh */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center flex-shrink-0">
                <Share2 className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="min-w-0">
                <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100 truncate">
                  {t("Cluster Graph")}
                </h1>
                <p className="text-xs text-slate-500 dark:text-slate-400 truncate">
                  {t("Visualize knowledge clusters and their relationships")}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 flex-shrink-0">
              <div className="hidden sm:flex items-center gap-3 text-xs">
                <span className="flex items-center gap-1 text-slate-500 dark:text-slate-400">
                  <Hash className="w-3.5 h-3.5" />
                  <span className="font-semibold text-slate-700 dark:text-slate-200">
                    {filteredNodes.length}
                  </span>
                  {t("nodes")}
                </span>
                <span className="flex items-center gap-1 text-slate-500 dark:text-slate-400">
                  <GitBranch className="w-3.5 h-3.5" />
                  <span className="font-semibold text-slate-700 dark:text-slate-200">
                    {filteredEdges.length}
                  </span>
                  {t("edges")}
                </span>
              </div>
              <button
                onClick={handleRefresh}
                disabled={refreshing}
                className="flex items-center gap-2 px-3 py-1.5 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-300 dark:disabled:bg-slate-700 text-white rounded-lg transition-colors text-sm"
              >
                <RefreshCw
                  className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`}
                />
                <span className="hidden sm:inline font-medium">
                  {t("Refresh")}
                </span>
              </button>
            </div>
          </div>

          {/* Row 2: controls */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-700/60 rounded-lg p-0.5">
              <button
                onClick={() => graphRef.current?.zoomIn()}
                title={t("Zoom In")}
                className="p-1.5 rounded-md hover:bg-white dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => graphRef.current?.zoomOut()}
                title={t("Zoom Out")}
                className="p-1.5 rounded-md hover:bg-white dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                onClick={() => graphRef.current?.resetView()}
                title={t("Fit View")}
                className="p-1.5 rounded-md hover:bg-white dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
              <button
                onClick={() => graphRef.current?.relayout()}
                title={t("Reset Layout")}
                className="p-1.5 rounded-md hover:bg-white dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                <Shuffle className="w-4 h-4" />
              </button>
            </div>

            {/* Weight threshold */}
            <div className="flex items-center gap-2 px-2 py-1 bg-slate-100 dark:bg-slate-700/60 rounded-lg">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400 whitespace-nowrap">
                {t("Weight Threshold")}
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={weightThreshold}
                onChange={(e) => setWeightThreshold(parseFloat(e.target.value))}
                className="w-24 accent-blue-500"
              />
              <span className="text-xs font-mono text-slate-600 dark:text-slate-300 w-8 text-right">
                {weightThreshold.toFixed(2)}
              </span>
            </div>

            {/* Repulsion (charge) */}
            <div className="flex items-center gap-2 px-2 py-1 bg-slate-100 dark:bg-slate-700/60 rounded-lg">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400 whitespace-nowrap">
                {t("Repulsion")}
              </span>
              <input
                type="range"
                min={0}
                max={1000}
                step={10}
                value={charge}
                onChange={(e) => setCharge(parseInt(e.target.value, 10))}
                className="w-24 accent-blue-500"
              />
              <span className="text-xs font-mono text-slate-600 dark:text-slate-300 w-10 text-right">
                {charge}
              </span>
            </div>

            {/* Centering force */}
            <div className="flex items-center gap-2 px-2 py-1 bg-slate-100 dark:bg-slate-700/60 rounded-lg">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400 whitespace-nowrap">
                {t("Centering")}
              </span>
              <input
                type="range"
                min={0}
                max={200}
                step={10}
                value={centeringSlider}
                onChange={(e) =>
                  setCenteringSlider(parseInt(e.target.value, 10))
                }
                className="w-24 accent-blue-500"
              />
              <span className="text-xs font-mono text-slate-600 dark:text-slate-300 w-12 text-right">
                {(centeringSlider / 1000).toFixed(3)}
              </span>
            </div>

            {/* Lifecycle filter */}
            <div className="flex items-center gap-1 px-2 py-1 bg-slate-100 dark:bg-slate-700/60 rounded-lg">
              {ALL_LIFECYCLES.map((lc) => {
                const on = enabledLifecycles.has(lc);
                const count = lifecycleCounts[lc] || 0;
                if (count === 0) return null;
                return (
                  <button
                    key={lc}
                    onClick={() => toggleLifecycle(lc)}
                    title={`${lc} (${count})`}
                    className={`flex items-center gap-1.5 px-2 py-0.5 rounded-md text-xs font-medium transition-all ${
                      on
                        ? "bg-white dark:bg-slate-900 shadow-sm"
                        : "opacity-40 hover:opacity-70"
                    }`}
                    style={
                      on
                        ? {
                            color: lifecycleColor(lc),
                            border: `1px solid ${lifecycleColor(lc)}40`,
                          }
                        : undefined
                    }
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ background: lifecycleColor(lc) }}
                    />
                    <span className="text-slate-600 dark:text-slate-300">
                      {count}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Main: graph + detail panel */}
      <div className="flex-1 flex min-h-0">
        {/* Graph */}
        <div className="flex-1 relative min-w-0">
          {hasData ? (
            <>
              <ClusterGraph
                ref={graphRef}
                nodes={filteredNodes}
                edges={filteredEdges}
                theme={theme}
                weightThreshold={0}
                charge={charge}
                centering={centeringSlider / 1000}
                selectedId={selectedId}
                selectedEdge={selectedEdge}
                onSelect={handleSelect}
                onHover={setHoveredId}
                onSelectEdge={handleSelectEdge}
              />
              {/* Legend */}
              <Legend
                counts={lifecycleCounts}
                total={graph?.nodes.length ?? 0}
                t={t}
              />
              {/* Hover tooltip */}
              {hoveredId && !selectedEdge && (
                <HoverBadge
                  id={hoveredId}
                  nodes={filteredNodes}
                  edges={filteredEdges}
                  t={t}
                />
              )}
              {/* Edge detail card */}
              {selectedEdge && (
                <EdgeDetailCard
                  edge={selectedEdge}
                  nodes={filteredNodes}
                  onFocusNode={(id) => {
                    setSelectedEdge(null);
                    handleFocusRelated(id);
                  }}
                  onClose={() => setSelectedEdge(null)}
                  t={t}
                />
              )}
            </>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <Share2 className="w-12 h-12 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
                <p className="text-slate-500 dark:text-slate-400">
                  {t("No clusters to display")}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selectedId && (
          <aside className="w-full sm:w-[380px] flex-shrink-0 border-l border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 overflow-y-auto animate-slide-in-from-bottom">
            <DetailPanel
              id={selectedId}
              detail={detail}
              loading={detailLoading}
              onClose={() => setSelectedId(null)}
              onFocusRelated={handleFocusRelated}
              t={t}
            />
          </aside>
        )}
      </div>
    </div>
  );
}

// === Hover badge ===

function HoverBadge({
  id,
  nodes,
  edges,
  t,
}: {
  id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  t: (key: string) => string;
}) {
  const node = nodes.find((n) => n.id === id);
  if (!node) return null;
  const degree = edges.filter(
    (e) => e.source === id || e.target === id,
  ).length;
  return (
    <div className="absolute top-3 left-3 max-w-xs bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg p-3 pointer-events-none z-10">
      <div className="flex items-center gap-2 mb-1">
        <span
          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
          style={{ background: lifecycleColor(node.lifecycle) }}
        />
        <p className="text-sm font-semibold text-slate-900 dark:text-slate-100 truncate">
          {node.name || node.id}
        </p>
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-slate-500 dark:text-slate-400">
        <span>
          {t("ID")}: <span className="font-mono">{node.id}</span>
        </span>
        <span>
          {t("Degree")}: <span className="font-semibold">{degree}</span>
        </span>
        {node.confidence != null && (
          <span>
            {t("Confidence")}: {pct(node.confidence)}
          </span>
        )}
        {node.hotness != null && (
          <span>
            {t("Hotness")}: {pct(node.hotness)}
          </span>
        )}
      </div>
    </div>
  );
}

// === Legend overlay ===

function Legend({
  counts,
  total,
  t,
}: {
  counts: Record<string, number>;
  total: number;
  t: (key: string) => string;
}) {
  const entries = ALL_LIFECYCLES.filter((lc) => (counts[lc] || 0) > 0);
  return (
    <div className="absolute top-3 right-3 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg p-2.5 z-10">
      <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-1.5">
        {t("Lifecycle")}
      </p>
      <div className="space-y-1">
        {entries.map((lc) => (
          <div key={lc} className="flex items-center gap-2">
            <span
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ background: lifecycleColor(lc) }}
            />
            <span className="text-xs font-medium text-slate-600 dark:text-slate-300 lowercase">
              {lc.toLowerCase()}
            </span>
            <span className="text-xs font-semibold text-slate-700 dark:text-slate-200 ml-auto">
              {counts[lc] || 0}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-1.5 pt-1.5 border-t border-slate-100 dark:border-slate-700 flex items-center justify-between">
        <span className="text-[10px] text-slate-400 dark:text-slate-500 uppercase tracking-wider">
          {t("Total")}
        </span>
        <span className="text-xs font-bold text-slate-700 dark:text-slate-200">
          {total}
        </span>
      </div>
    </div>
  );
}

// === Edge detail card ===

function EdgeDetailCard({
  edge,
  nodes,
  onFocusNode,
  onClose,
  t,
}: {
  edge: SelectedEdge;
  nodes: GraphNode[];
  onFocusNode: (id: string) => void;
  onClose: () => void;
  t: (key: string) => string;
}) {
  const src = nodes.find((n) => n.id === edge.source);
  const tgt = nodes.find((n) => n.id === edge.target);
  const weightPct = Math.round((edge.weight || 0) * 100);
  return (
    <div className="absolute top-3 left-1/2 -translate-x-1/2 max-w-md w-full max-w-[420px] bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg p-3 z-10">
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
          <GitBranch className="w-3.5 h-3.5" />
          {t("Edge")}
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-md text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="flex items-center gap-2 mb-3">
        <button
          onClick={() => onFocusNode(edge.source)}
          className="flex-1 min-w-0 text-left p-2 rounded-md bg-slate-50 dark:bg-slate-700/40 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
        >
          <div className="flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: lifecycleColor(src?.lifecycle) }}
            />
            <span className="text-xs font-mono text-blue-600 dark:text-blue-400">
              {edge.source}
            </span>
          </div>
          <p className="text-xs text-slate-700 dark:text-slate-200 truncate mt-0.5">
            {src?.name || edge.source}
          </p>
        </button>
        <span className="text-slate-300 dark:text-slate-600 text-lg flex-shrink-0">
          →
        </span>
        <button
          onClick={() => onFocusNode(edge.target)}
          className="flex-1 min-w-0 text-left p-2 rounded-md bg-slate-50 dark:bg-slate-700/40 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
        >
          <div className="flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: lifecycleColor(tgt?.lifecycle) }}
            />
            <span className="text-xs font-mono text-blue-600 dark:text-blue-400">
              {edge.target}
            </span>
          </div>
          <p className="text-xs text-slate-700 dark:text-slate-200 truncate mt-0.5">
            {tgt?.name || edge.target}
          </p>
        </button>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="flex items-center justify-between p-1.5 bg-slate-50 dark:bg-slate-700/40 rounded-md">
          <span className="text-slate-500 dark:text-slate-400">{t("Weight")}</span>
          <span className="font-semibold text-slate-700 dark:text-slate-200">
            {weightPct}%
          </span>
        </div>
        <div className="flex items-center justify-between p-1.5 bg-slate-50 dark:bg-slate-700/40 rounded-md">
          <span className="text-slate-500 dark:text-slate-400">{t("Type")}</span>
          <span className="font-mono text-slate-700 dark:text-slate-200">
            {edge.type || "—"}
          </span>
        </div>
      </div>
      <div className="mt-2 h-1.5 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${weightPct}%`,
            background: lifecycleColor("EMERGING"),
          }}
        />
      </div>
    </div>
  );
}

// === Detail panel ===

function DetailPanel({
  id,
  detail,
  loading,
  onClose,
  onFocusRelated,
  t,
}: {
  id: string;
  detail: ClusterDetail | null;
  loading: boolean;
  onClose: () => void;
  onFocusRelated: (id: string) => void;
  t: (key: string) => string;
}) {
  return (
    <div className="p-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-4">
        <div className="min-w-0 flex-1">
          {detail ? (
            <div className="flex items-center gap-2 mb-1">
              <span
                className="px-2 py-0.5 rounded-md text-xs font-semibold"
                style={{
                  background: `${lifecycleColor(detail.lifecycle)}22`,
                  color: lifecycleColor(detail.lifecycle),
                  border: `1px solid ${lifecycleColor(detail.lifecycle)}55`,
                }}
              >
                {detail.lifecycle || "—"}
              </span>
              {detail.abstraction_level && (
                <span className="px-2 py-0.5 rounded-md text-xs font-medium bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300">
                  {detail.abstraction_level}
                </span>
              )}
            </div>
          ) : null}
          <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 break-words leading-tight">
            {detail?.name || id}
          </h2>
          <p className="text-xs font-mono text-slate-400 dark:text-slate-500 mt-0.5">
            {id}
          </p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-md text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors flex-shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400 py-6">
          <Loader2 className="w-4 h-4 animate-spin" />
          {t("Loading...")}
        </div>
      )}

      {!loading && detail && (
        <div className="space-y-5">
          {/* Stat pills */}
          <div className="grid grid-cols-3 gap-2">
            <StatPill
              icon={<TrendingUp className="w-3.5 h-3.5" />}
              label={t("Confidence")}
              value={pct(detail.confidence)}
              color="text-green-600 dark:text-green-400"
            />
            <StatPill
              icon={<Zap className="w-3.5 h-3.5" />}
              label={t("Hotness")}
              value={pct(detail.hotness)}
              color="text-orange-600 dark:text-orange-400"
            />
            <StatPill
              icon={<GitBranch className="w-3.5 h-3.5" />}
              label={t("Version")}
              value={String(detail.version ?? 0)}
            />
            <StatPill
              icon={<Layers className="w-3.5 h-3.5" />}
              label={t("Merge Count")}
              value={String(detail.merge_count ?? 0)}
            />
            <StatPill
              icon={<Hash className="w-3.5 h-3.5" />}
              label={t("Related")}
              value={String(detail.related_clusters?.length ?? 0)}
            />
            <StatPill
              icon={<TrendingUp className="w-3.5 h-3.5" />}
              label={t("Landmark")}
              value={pct(detail.landmark_potential)}
            />
          </div>

          {/* Description */}
          {joinField(detail.description) && (
            <Section icon={<FileText className="w-4 h-4" />} title={t("Description")}>
              <p className="text-sm text-slate-600 dark:text-slate-300 whitespace-pre-wrap leading-relaxed">
                {joinField(detail.description)}
              </p>
            </Section>
          )}

          {/* Content (markdown) */}
          {joinField(detail.content) && (
            <Section icon={<FileText className="w-4 h-4" />} title={t("Content")}>
              <div className="prose prose-sm dark:prose-invert max-w-none prose-slate prose-pre:bg-slate-100 dark:prose-pre:bg-slate-900 prose-pre:text-slate-700 dark:prose-pre:text-slate-200 prose-code:before:content-none prose-code:after:content-none prose-code:bg-slate-100 dark:prose-code:bg-slate-700 prose-code:px-1 prose-code:py-0.5 prose-code:rounded">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {joinField(detail.content)}
                </ReactMarkdown>
              </div>
            </Section>
          )}

          {/* Patterns */}
          {detail.patterns && detail.patterns.length > 0 && (
            <Section icon={<Tag className="w-4 h-4" />} title={t("Patterns")}>
              <div className="flex flex-wrap gap-1.5">
                {detail.patterns.map((p, i) => (
                  <span
                    key={i}
                    className="px-2 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-xs"
                  >
                    {p}
                  </span>
                ))}
              </div>
            </Section>
          )}

          {/* Constraints */}
          {detail.constraints && detail.constraints.length > 0 && (
            <Section
              icon={<ShieldAlert className="w-4 h-4" />}
              title={t("Constraints")}
            >
              <div className="space-y-1.5">
                {detail.constraints.map((c, i) => (
                  <div
                    key={i}
                    className="p-2 bg-slate-50 dark:bg-slate-700/40 rounded-md border border-slate-100 dark:border-slate-700"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="text-xs font-mono text-slate-600 dark:text-slate-300 break-all">
                        {c.condition || c.description || "—"}
                      </span>
                      {c.severity && (
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-semibold flex-shrink-0 ${
                            c.severity === "high"
                              ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
                              : c.severity === "medium"
                                ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                                : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                          }`}
                        >
                          {c.severity}
                        </span>
                      )}
                    </div>
                    {c.description && c.condition && (
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        {c.description}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* Queries */}
          {detail.queries && detail.queries.length > 0 && (
            <Section icon={<Tag className="w-4 h-4" />} title={t("Queries")}>
              <div className="space-y-1">
                {detail.queries.map((q, i) => (
                  <p
                    key={i}
                    className="text-xs text-slate-600 dark:text-slate-300 px-2 py-1 bg-slate-50 dark:bg-slate-700/40 rounded"
                  >
                    {q}
                  </p>
                ))}
              </div>
            </Section>
          )}

          {/* Related clusters */}
          {detail.related_clusters && detail.related_clusters.length > 0 && (
            <Section
              icon={<LinkIcon className="w-4 h-4" />}
              title={`${t("Related Clusters")} (${detail.related_clusters.length})`}
            >
              <div className="space-y-1">
                {detail.related_clusters
                  .slice()
                  .sort((a, b) => (b.weight || 0) - (a.weight || 0))
                  .slice(0, 30)
                  .map((rc, i) => (
                    <button
                      key={i}
                      onClick={() => onFocusRelated(rc.target_cluster_id)}
                      className="w-full flex items-center justify-between gap-2 px-2 py-1.5 rounded-md hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors text-left group"
                    >
                      <span className="font-mono text-xs text-blue-600 dark:text-blue-400 group-hover:underline truncate">
                        {rc.target_cluster_id}
                      </span>
                      <span className="flex items-center gap-2 flex-shrink-0">
                        {rc.source && (
                          <span className="text-[10px] text-slate-400 dark:text-slate-500">
                            {rc.source}
                          </span>
                        )}
                        <span className="text-xs font-semibold text-slate-600 dark:text-slate-300">
                          {(rc.weight * 100).toFixed(0)}%
                        </span>
                        <div className="w-12 h-1.5 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.min(100, rc.weight * 100)}%`,
                              background: lifecycleColor("EMERGING"),
                            }}
                          />
                        </div>
                      </span>
                    </button>
                  ))}
              </div>
            </Section>
          )}

          {/* Scripts */}
          {detail.scripts && detail.scripts.length > 0 && (
            <Section icon={<FileText className="w-4 h-4" />} title={t("Scripts")}>
              <div className="space-y-2">
                {detail.scripts.map((s, i) => (
                  <pre
                    key={i}
                    className="text-xs bg-slate-100 dark:bg-slate-900 rounded-md p-2 overflow-x-auto text-slate-700 dark:text-slate-200"
                  >
                    <code>{s}</code>
                  </pre>
                ))}
              </div>
            </Section>
          )}

          {/* Timestamps */}
          {(detail.create_time || detail.last_modified) && (
            <div suppressHydrationWarning className="text-[11px] text-slate-400 dark:text-slate-500 pt-2 border-t border-slate-100 dark:border-slate-700 space-y-0.5">
              {detail.create_time && (
                <p>
                  {t("Created")}:{" "}
                  {new Date(detail.create_time).toLocaleString()}
                </p>
              )}
              {detail.last_modified && (
                <p>
                  {t("Modified")}:{" "}
                  {new Date(detail.last_modified).toLocaleString()}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StatPill({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-slate-50 dark:bg-slate-700/40 rounded-lg p-2 text-center">
      <div
        className={`flex items-center justify-center gap-1 mb-0.5 text-slate-400 dark:text-slate-500 ${
          color || ""
        }`}
      >
        {icon}
      </div>
      <p
        className={`text-sm font-bold ${
          color || "text-slate-700 dark:text-slate-200"
        }`}
      >
        {value}
      </p>
      <p className="text-[10px] text-slate-400 dark:text-slate-500 truncate">
        {label}
      </p>
    </div>
  );
}

function Section({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2">
        <span className="text-blue-500 dark:text-blue-400">{icon}</span>
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          {title}
        </h3>
      </div>
      {children}
    </div>
  );
}
