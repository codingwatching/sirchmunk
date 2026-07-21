"use client";

import React, {
  useEffect,
  useRef,
  useImperativeHandle,
  forwardRef,
} from "react";
import * as d3 from "d3";

export type LifecycleKey =
  | "STABLE"
  | "EMERGING"
  | "CONTESTED"
  | "DEPRECATED"
  | "META";

export interface GraphNode {
  id: string;
  name: string;
  confidence?: number | null;
  hotness?: number | null;
  lifecycle?: string | null;
  abstraction_level?: string | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight?: number | null;
  type?: string | null;
}

export interface ClusterGraphHandle {
  zoomIn: () => void;
  zoomOut: () => void;
  resetView: () => void;
  focusNode: (id: string) => void;
  relayout: () => void;
}

export interface SelectedEdge {
  source: string;
  target: string;
  weight: number;
  type: string;
}

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  theme: "light" | "dark";
  weightThreshold: number;
  charge: number;
  centering: number;
  selectedId: string | null;
  selectedEdge: SelectedEdge | null;
  onSelect: (id: string | null) => void;
  onHover: (id: string | null) => void;
  onSelectEdge: (edge: SelectedEdge | null) => void;
}

export const LIFECYCLE_COLORS: Record<string, string> = {
  STABLE: "#22c55e",
  EMERGING: "#3b82f6",
  CONTESTED: "#f59e0b",
  DEPRECATED: "#94a3b8",
  META: "#a855f7",
};

export function lifecycleColor(lc?: string | null): string {
  if (!lc) return "#94a3b8";
  return LIFECYCLE_COLORS[lc] || "#94a3b8";
}

interface SimNode extends d3.SimulationNodeDatum, GraphNode {
  degree: number;
  isMeta: boolean;
  neighbors: Set<string>;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  weight: number;
  type: string;
}

function nodeRadius(d: SimNode): number {
  return (d.isMeta ? 13 : 8) + Math.sqrt(Math.max(0, d.degree)) * 2.5;
}

function truncate(s: string, n = 26): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + "…";
}

const WARMUP_TICKS = 300;
const PAD = 60;

export const ClusterGraph = forwardRef<ClusterGraphHandle, Props>(
  function ClusterGraph(
    {
      nodes,
      edges,
      theme,
      weightThreshold,
      charge,
      centering,
      selectedId,
      selectedEdge,
      onSelect,
      onHover,
      onSelectEdge,
    },
    ref,
  ) {
    const svgRef = useRef<SVGSVGElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const stateRef = useRef({
      simulation: null as d3.Simulation<SimNode, SimLink> | null,
      zoom: null as d3.ZoomBehavior<SVGSVGElement, unknown> | null,
      svg: null as d3.Selection<SVGSVGElement, unknown, null, undefined> | null,
      g: null as d3.Selection<SVGGElement, unknown, null, undefined> | null,
      nodeSel: null as d3.Selection<SVGGElement, SimNode, SVGGElement, unknown> | null,
      linkSel: null as d3.Selection<SVGGElement, SimLink, SVGGElement, unknown> | null,
      width: 0,
      height: 0,
      simNodes: [] as SimNode[],
      simLinks: [] as SimLink[],
      nodeByKey: new Map<string, SimNode>(),
      posCache: new Map<string, { x: number; y: number }>(),
      hoveredId: null as string | null,
    });

    // keep latest selection/hover accessible to d3 handlers
    const selectedRef = useRef<string | null>(selectedId);
    // Only auto-focus the initial view on the very first data load;
    // subsequent data changes (weight threshold / lifecycle filter) must
    // preserve the user's current pan/zoom, otherwise the screen jumps.
    const firstLoadRef = useRef(true);
    useEffect(() => {
      selectedRef.current = selectedId;
      applyStyles();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedId]);

    const selectedEdgeRef = useRef<SelectedEdge | null>(selectedEdge);
    useEffect(() => {
      selectedEdgeRef.current = selectedEdge;
      applyStyles();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedEdge]);

    const palette =
      theme === "dark"
        ? {
            edge: "#64748b",
            edgeLit: "#60a5fa",
            edgeMeta: "#c489ff",
            label: "#e2e8f0",
            labelStroke: "#0f172a",
            nodeStroke: "#0f172a",
            ringSelected: "#60a5fa",
            gridHint: "rgba(148,163,184,0.10)",
          }
        : {
            edge: "#94a3b8",
            edgeLit: "#2563eb",
            edgeMeta: "#a855f7",
            label: "#1e293b",
            labelStroke: "#ffffff",
            nodeStroke: "#ffffff",
            ringSelected: "#2563eb",
            gridHint: "rgba(148,163,184,0.14)",
          };

    // Keep latest palette accessible to D3 handlers / imperative methods
    // (which capture closures once and would otherwise hold a stale theme).
    const paletteRef = useRef(palette);
    paletteRef.current = palette;

    function applyStyles() {
      const s = stateRef.current;
      if (!s.nodeSel || !s.linkSel) return;
      const selId = selectedRef.current;
      const hovId = s.hoveredId;
      const p = paletteRef.current;

      // links
      const selEdge = selectedEdgeRef.current;
      const isSelEdge = (l: SimLink): boolean =>
        !!selEdge &&
        (l.source as SimNode).id === selEdge.source &&
        (l.target as SimNode).id === selEdge.target;
      const isMetaEdge = (l: SimLink): boolean =>
        (l.source as SimNode).isMeta || (l.target as SimNode).isMeta;
      s.linkSel
        .select("line.link")
        .attr("stroke", (l) => {
          if (isMetaEdge(l)) return p.edgeMeta;
          if (isSelEdge(l)) return p.edgeLit;
          if (hovId && isIncident(l, hovId)) return p.edgeLit;
          return p.edge;
        })
        .attr("stroke-opacity", (l) => {
          if (isSelEdge(l)) return 1;
          const meta = isMetaEdge(l);
          const baseOpacity = meta
            ? 0.45 + (l.weight || 0) * 0.3
            : 0.22 + (l.weight || 0) * 0.4;
          if (!hovId) return baseOpacity;
          return isIncident(l, hovId) ? 0.95 : 0.03;
        })
        .attr("stroke-width", (l) => {
          const w = 1 + (l.weight || 0) * 3;
          return isSelEdge(l) ? w + 1.5 : isMetaEdge(l) ? w + 0.5 : w;
        });

      s.linkSel
        .select("line.link-hit")
        .attr("stroke", "transparent")
        .attr("stroke-width", 14);

      // nodes
      s.nodeSel.attr("opacity", (d) => {
        if (selEdge && (d.id === selEdge.source || d.id === selEdge.target))
          return 1;
        if (!hovId) return 1;
        if (d.id === hovId || d.neighbors.has(hovId)) return 1;
        return 0.12;
      });
      s.nodeSel
        .select("circle.main")
        .attr("r", (d) => nodeRadius(d))
        .attr("fill", (d) => lifecycleColor(d.lifecycle))
        .attr("stroke", (d) =>
          d.id === selId
            ? d.isMeta
              ? p.edgeMeta
              : p.ringSelected
            : p.nodeStroke,
        )
        .attr("stroke-width", (d) => (d.id === selId ? 3.2 : 1.5))
        .attr("stroke-dasharray", (d) =>
          d.lifecycle === "DEPRECATED" ? "3 3" : null,
        );

      s.nodeSel
        .select("circle.ring")
        .attr("r", (d) => nodeRadius(d) + 5)
        .attr("fill", "none")
        .attr("stroke", (d) =>
          d.id === selId
            ? d.isMeta
              ? p.edgeMeta
              : p.ringSelected
            : "none",
        )
        .attr("stroke-width", 2)
        .attr("opacity", (d) => (d.id === selId ? 0.9 : 0));

      // labels
      s.nodeSel
        .select("text")
        .attr("fill", p.label)
        .attr("stroke", p.labelStroke)
        .attr("paint-order", "stroke")
        .attr("stroke-width", 3);
      updateLabelVisibility();
    }

    function updateLabelVisibility() {
      const s = stateRef.current;
      if (!s.nodeSel) return;
      const selId = selectedRef.current;
      const hovId = s.hoveredId;
      s.nodeSel.select("text").style("display", (d) => {
        // Labels appear only for the active interaction target and META nodes.
        // Showing all labels on zoom-in caused unreadable text-on-text piles
        // on large graphs; hover any node to reveal its name.
        if (d.id === selId || d.id === hovId) return null;
        if (d.isMeta) return null;
        return "none";
      });
    }

    function isIncident(l: SimLink, id: string): boolean {
      const s = l.source as SimNode | string;
      const t = l.target as SimNode | string;
      const sid = typeof s === "string" ? s : s.id;
      const tid = typeof t === "string" ? t : t.id;
      return sid === id || tid === id;
    }

    function currentZoomScale(): number {
      const s = stateRef.current;
      if (!s.svg) return 1;
      const node = s.svg.node();
      if (!node) return 1;
      try {
        const t = d3.zoomTransform(node);
        return t.k;
      } catch {
        return 1;
      }
    }

    function ticked() {
      const s = stateRef.current;
      if (!s.nodeSel || !s.linkSel) return;
      s.linkSel
        .select("line.link")
        .attr("x1", (l) => (l.source as SimNode).x!)
        .attr("y1", (l) => (l.source as SimNode).y!)
        .attr("x2", (l) => (l.target as SimNode).x!)
        .attr("y2", (l) => (l.target as SimNode).y!);
      s.linkSel
        .select("line.link-hit")
        .attr("x1", (l) => (l.source as SimNode).x!)
        .attr("y1", (l) => (l.source as SimNode).y!)
        .attr("x2", (l) => (l.target as SimNode).x!)
        .attr("y2", (l) => (l.target as SimNode).y!);
      s.nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
    }

    function fitView(animate = true) {
      const s = stateRef.current;
      if (!s.svg || !s.g || !s.zoom || s.simNodes.length === 0) return;
      const xs = s.simNodes.map((n) => n.x!);
      const ys = s.simNodes.map((n) => n.y!);
      if (xs.length === 0) return;
      const minX = Math.min(...xs),
        maxX = Math.max(...xs);
      const minY = Math.min(...ys),
        maxY = Math.max(...ys);
      const w = maxX - minX || 1;
      const h = maxY - minY || 1;
      const scale = Math.min(
        4,
        Math.max(
          0.2,
          Math.min(
            (s.width - PAD * 2) / w,
            (s.height - PAD * 2) / h,
          ),
        ),
      );
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      const tx = s.width / 2 - scale * cx;
      const ty = s.height / 2 - scale * cy;
      const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);
      if (animate) {
        s.svg.transition().duration(600).call(s.zoom!.transform, transform);
      } else {
        s.svg.call(s.zoom!.transform, transform);
      }
    }

    function focusNode(id: string) {
      const s = stateRef.current;
      const n = s.nodeByKey.get(id);
      if (!n || !s.svg || !s.zoom) return;
      const scale = Math.max(1.3, currentZoomScale());
      const tx = s.width / 2 - scale * (n.x || 0);
      const ty = s.height / 2 - scale * (n.y || 0);
      const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);
      s.svg.transition().duration(600).call(s.zoom.transform, transform);
      applyStyles();
    }

    function relayout() {
      const s = stateRef.current;
      s.posCache.clear();
      s.simNodes.forEach((n) => {
        n.x = s.width / 2 + (Math.random() - 0.5) * 200;
        n.y = s.height / 2 + (Math.random() - 0.5) * 200;
        n.vx = 0;
        n.vy = 0;
        n.fx = null;
        n.fy = null;
      });
      if (s.simulation) {
        s.simulation.alpha(1).restart();
        for (let i = 0; i < WARMUP_TICKS; i++) s.simulation.tick();
        s.simulation.alpha(0).stop();
      }
      ticked();
      setTimeout(() => focusInitial(), 50);
    }

    // Initial view: for large graphs, focus on one hub region at a readable
    // zoom instead of shrinking all nodes to fit. The user pans/zooms out to
    // explore other regions. For small graphs, fit everything.
    function focusInitial() {
      const s = stateRef.current;
      if (!s.svg || !s.zoom || s.simNodes.length === 0) return;
      if (s.simNodes.length < 40) {
        fitView(false);
        return;
      }
      // Pick focal node: prefer META, else highest degree.
      let focal = s.simNodes[0];
      for (const n of s.simNodes) {
        if (n.isMeta) {
          focal = n;
          break;
        }
        if ((n.degree || 0) > (focal.degree || 0)) focal = n;
      }
      const scale = 1.0;
      const tx = s.width / 2 - scale * (focal.x || 0);
      const ty = s.height / 2 - scale * (focal.y || 0);
      const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);
      s.svg.call(s.zoom.transform, transform);
    }

    useImperativeHandle(
      ref,
      () => ({
        zoomIn: () => {
          const s = stateRef.current;
          if (s.svg && s.zoom)
            s.svg.transition().duration(200).call(s.zoom.scaleBy, 1.3);
        },
        zoomOut: () => {
          const s = stateRef.current;
          if (s.svg && s.zoom)
            s.svg.transition().duration(200).call(s.zoom.scaleBy, 1 / 1.3);
        },
        resetView: () => fitView(true),
        focusNode,
        relayout,
      }),
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [],
    );

    // One-time SVG scaffold + zoom + resize observer
    useEffect(() => {
      const svgEl = svgRef.current!;
      const container = containerRef.current!;
      const svg = d3.select(svgEl);
      const g = svg.append("g").attr("class", "zoom-layer");
      // defs for arrow marker (not used by default but kept minimal)
      // background rect for deselect
      const bg = g
        .append("rect")
        .attr("class", "graph-bg")
        .attr("x", -100000)
        .attr("y", -100000)
        .attr("width", 200000)
        .attr("height", 200000)
        .attr("fill", "transparent")
        .attr("pointer-events", "all");
      bg.on("click", () => {
        onSelect(null);
        onSelectEdge(null);
      });
      const linkLayer = g.append("g").attr("class", "links");
      const nodeLayer = g.append("g").attr("class", "nodes");

      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.2, 4])
        .on("zoom", (event) => {
          g.attr("transform", event.transform.toString());
          updateLabelVisibility();
        });
      svg.call(zoom);

      const ro = new ResizeObserver(() => {
        const rect = container.getBoundingClientRect();
        stateRef.current.width = rect.width;
        stateRef.current.height = rect.height;
        svg.attr("width", rect.width).attr("height", rect.height);
      });
      ro.observe(container);
      const rect = container.getBoundingClientRect();
      stateRef.current.width = rect.width;
      stateRef.current.height = rect.height;
      svg.attr("width", rect.width).attr("height", rect.height);

      stateRef.current.svg = svg;
      stateRef.current.g = g;
      stateRef.current.zoom = zoom;
      stateRef.current.nodeSel = nodeLayer
        .selectAll<SVGGElement, SimNode>("g.node")
        .data([] as SimNode[]);
      stateRef.current.linkSel = linkLayer
        .selectAll<SVGGElement, SimLink>("g.edge")
        .data([] as SimLink[]);

      return () => {
        ro.disconnect();
        if (stateRef.current.simulation) stateRef.current.simulation.stop();
        svg.selectAll("*").remove();
        svg.on(".zoom", null);
        stateRef.current.simulation = null;
        stateRef.current.zoom = null;
        stateRef.current.svg = null;
        stateRef.current.g = null;
        stateRef.current.nodeSel = null;
        stateRef.current.linkSel = null;
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Data effect: rebuild sim nodes/links, render, layout
    useEffect(() => {
      const s = stateRef.current;
      if (!s.svg || !s.g) return;

      // Filter + dedup edges by weight threshold and undirected key
      const seen = new Set<string>();
      const filtered: GraphEdge[] = [];
      for (const e of edges) {
        const w = typeof e.weight === "number" ? e.weight : 0.5;
        if (w < weightThreshold) continue;
        const key = [e.source, e.target].sort().join("||");
        if (seen.has(key)) continue;
        seen.add(key);
        filtered.push(e);
      }

      // degree map
      const degree = new Map<string, number>();
      const neighborMap = new Map<string, Set<string>>();
      const ensure = (id: string) => {
        if (!neighborMap.has(id)) neighborMap.set(id, new Set());
        return neighborMap.get(id)!;
      };
      for (const e of filtered) {
        ensure(e.source).add(e.target);
        ensure(e.target).add(e.source);
        degree.set(e.source, (degree.get(e.source) || 0) + 1);
        degree.set(e.target, (degree.get(e.target) || 0) + 1);
      }

      // Build SimNodes; preserve cached positions
      const simNodes: SimNode[] = nodes.map((n) => {
        const cached = s.posCache.get(n.id);
        const prev = s.nodeByKey.get(n.id);
        const isMeta = n.lifecycle === "META";
        const sim: SimNode = {
          ...n,
          id: n.id,
          name: n.name,
          confidence: n.confidence ?? null,
          hotness: n.hotness ?? null,
          lifecycle: n.lifecycle ?? null,
          abstraction_level: n.abstraction_level ?? null,
          degree: degree.get(n.id) || 0,
          isMeta,
          neighbors: neighborMap.get(n.id) || new Set<string>(),
          x: prev?.x ?? cached?.x ?? s.width / 2 + (Math.random() - 0.5) * 300,
          y: prev?.y ?? cached?.y ?? s.height / 2 + (Math.random() - 0.5) * 300,
          vx: 0,
          vy: 0,
        };
        return sim;
      });
      const nodeByKey = new Map(simNodes.map((n) => [n.id, n] as const));

      const simLinks: SimLink[] = filtered
        .map((e) => {
          const src = nodeByKey.get(e.source);
          const tgt = nodeByKey.get(e.target);
          if (!src || !tgt) return null;
          return {
            source: src.id,
            target: tgt.id,
            weight: typeof e.weight === "number" ? e.weight : 0.5,
            type: e.type || "related",
          } as SimLink;
        })
        .filter((x): x is SimLink => x !== null);

      s.simNodes = simNodes;
      s.simLinks = simLinks;
      s.nodeByKey = nodeByKey;

      // Build / reuse simulation
      if (s.simulation) s.simulation.stop();
      const sim = d3
        .forceSimulation<SimNode>(simNodes)
        .force(
          "link",
          d3
            .forceLink<SimNode, SimLink>(simLinks)
            .id((d) => d.id)
            .distance((l) => 70 - (l.weight || 0.5) * 25)
            .strength((l) => 0.12 + (l.weight || 0.5) * 0.35),
        )
        .force("charge", d3.forceManyBody<SimNode>().strength(-charge))
        .force(
          "collide",
          d3.forceCollide<SimNode>().radius((d) => nodeRadius(d) + 6),
        )
        .force("x", d3.forceX<SimNode>(s.width / 2).strength(centering))
        .force("y", d3.forceY<SimNode>(s.height / 2).strength(centering))
        .alphaDecay(0.03)
        .on("tick", ticked);
      s.simulation = sim;

      // Pre-warm then FULLY STOP so the graph is static until the user
      // drags a node or requests a relayout. A lingering low-alpha sim
      // causes the "everything jitters when I move the mouse" effect.
      for (let i = 0; i < WARMUP_TICKS; i++) sim.tick();
      sim.alpha(0).stop();

      // Render links (two-line: visible + invisible hit)
      const linkSel = s
        .g!.select("g.links")
        .selectAll<SVGGElement, SimLink>("g.edge")
        .data(simLinks, (l) => `${(l.source as SimNode).id}-${(l.target as SimNode).id}`);
      linkSel.exit().remove();
      const linkEnter = linkSel
        .enter()
        .append("g")
        .attr("class", "edge")
        .style("pointer-events", "none");
      linkEnter
        .append("line")
        .attr("class", "link")
        .attr("stroke-linecap", "round");
      linkEnter
        .append("line")
        .attr("class", "link-hit")
        .attr("stroke", "transparent")
        .attr("stroke-width", 14)
        .style("cursor", "pointer")
        .style("pointer-events", "stroke")
        .on("mouseover", function (event, l) {
          const sid = (l.source as SimNode).id;
          const tid = (l.target as SimNode).id;
          s.hoveredId = sid;
          // highlight both endpoints briefly via neighbors trick
          s.nodeSel?.attr("opacity", (d) =>
            d.id === sid || d.id === tid ? 1 : 0.12,
          );
          s.linkSel?.select("line.link").attr("stroke-opacity", (x) =>
            x === l ? 0.95 : 0.04,
          );
          onHover(sid);
        })
        .on("mouseout", function () {
          s.hoveredId = null;
          applyStyles();
          onHover(null);
        })
        .on("click", function (event, l) {
          event.stopPropagation();
          onSelectEdge({
            source: (l.source as SimNode).id,
            target: (l.target as SimNode).id,
            weight: l.weight,
            type: l.type,
          });
        });
      const linkMerge = linkEnter.merge(linkSel as any);
      // hit line should receive pointer events
      linkMerge.style("pointer-events", "none");
      linkMerge.select("line.link-hit").style("pointer-events", "stroke");

      // Render nodes
      const nodeSel = s
        .g!.select("g.nodes")
        .selectAll<SVGGElement, SimNode>("g.node")
        .data(simNodes, (d) => d.id);
      nodeSel.exit().remove();
      const nodeEnter = nodeSel
        .enter()
        .append("g")
        .attr("class", "node")
        .style("cursor", "pointer")
        .style("pointer-events", "all");
      nodeEnter.append("circle").attr("class", "ring");
      nodeEnter
        .append("circle")
        .attr("class", "main")
        .attr("stroke-width", 1.5);
      nodeEnter
        .append("text")
        .attr("class", "label")
        .attr("dy", "-0.35em")
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("font-weight", 600)
        .style("pointer-events", "none")
        .style("user-select", "none")
        .text((d) => truncate(d.name || d.id));

      const nodeMerge = nodeEnter.merge(nodeSel as any);
      // Refresh label text for persisted nodes (name may have changed)
      nodeMerge
        .select("text")
        .text((d) => truncate((d as SimNode).name || (d as SimNode).id));

      // drag
      const drag = d3
        .drag<SVGGElement, SimNode>()
        .on("start", (event, d) => {
          if (!event.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) sim.alphaTarget(0);
          d.fx = null;
          d.fy = null;
          // persist to cache
          if (d.x != null && d.y != null)
            s.posCache.set(d.id, { x: d.x, y: d.y });
        });
      nodeMerge.call(drag as any);

      // click + hover on nodes
      nodeMerge
        .on("click", (event, d) => {
          event.stopPropagation();
          onSelectEdge(null);
          onSelect(d.id);
        })
        .on("mouseover", (event, d) => {
          s.hoveredId = d.id;
          applyStyles();
          onHover(d.id);
        })
        .on("mouseout", () => {
          s.hoveredId = null;
          applyStyles();
          onHover(null);
        });

      s.nodeSel = nodeMerge as any;
      s.linkSel = linkMerge as any;

      // persist positions after warmup
      simNodes.forEach((n) => {
        if (n.x != null && n.y != null) s.posCache.set(n.id, { x: n.x, y: n.y });
      });

      ticked();
      applyStyles();
      if (firstLoadRef.current) {
        firstLoadRef.current = false;
        setTimeout(() => focusInitial(), 60);
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [nodes, edges, weightThreshold]);

    // Theme effect: re-apply styles without re-running layout
    useEffect(() => {
      applyStyles();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [theme]);

    // Force-parameter tuning: update the existing simulation's forces in
    // place (no rebuild / no 300-tick warmup) so slider dragging is smooth
    // and the current pan/zoom is preserved.
    useEffect(() => {
      const s = stateRef.current;
      const sim = s.simulation;
      if (!sim) return;
      const chargeF = sim.force("charge") as d3.ForceManyBody<SimNode> | undefined;
      if (chargeF) chargeF.strength(-charge);
      const xF = sim.force("x") as d3.ForceX<SimNode> | undefined;
      if (xF) xF.strength(centering);
      const yF = sim.force("y") as d3.ForceY<SimNode> | undefined;
      if (yF) yF.strength(centering);
      sim.alpha(0.5).restart();
      for (let i = 0; i < 80; i++) sim.tick();
      sim.alpha(0).stop();
      ticked();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [charge, centering]);

    return (
      <div
        ref={containerRef}
        className="relative w-full h-full overflow-hidden"
        style={{
          background:
            theme === "dark"
              ? "radial-gradient(circle at 30% 20%, rgba(59,130,246,0.06), transparent 60%), radial-gradient(circle at 80% 80%, rgba(168,85,247,0.05), transparent 55%)"
              : "radial-gradient(circle at 30% 20%, rgba(59,130,246,0.05), transparent 60%), radial-gradient(circle at 80% 80%, rgba(168,85,247,0.04), transparent 55%)",
        }}
      >
        <svg
          ref={svgRef}
          className="absolute inset-0 w-full h-full"
          style={{ display: "block", cursor: "grab" }}
        />
      </div>
    );
  },
);

export default ClusterGraph;
