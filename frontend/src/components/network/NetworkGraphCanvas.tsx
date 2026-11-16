import React, { useState, useMemo, useRef, useCallback } from 'react';
import type { NetworkNodeItem, NetworkEdgeItem } from '../../utils/analytics';

interface NetworkGraphCanvasProps {
  nodes: NetworkNodeItem[];
  edges?: NetworkEdgeItem[];
  platformFilter?: string;
}

interface NodePosition {
  x: number;
  y: number;
  node: NetworkNodeItem;
  radius: number;
  isHub: boolean;
  totalDegree: number;
  inDegree: number;
  outDegree: number;
}

export const NetworkGraphCanvas: React.FC<NetworkGraphCanvasProps> = ({
  nodes,
  edges = [],
  platformFilter = 'all',
}) => {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'all' | 'connected'>('connected');
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const svgRef = useRef<SVGSVGElement | null>(null);

  const CANVAS_WIDTH = 960;
  const CANVAS_HEIGHT = 560;
  const CX = CANVAS_WIDTH / 2;
  const CY = CANVAS_HEIGHT / 2;

  // Calculate Degree Centrality & Connection Maps (O(V + E))
  const { nodeDegrees, nodeConnections, topInfluencers } = useMemo(() => {
    const degrees = new Map<string, { inDegree: number; outDegree: number; totalDegree: number }>();
    const connections = new Map<string, { in: { id: string; weight: number }[]; out: { id: string; weight: number }[] }>();

    for (const n of nodes) {
      degrees.set(n.nodeId, {
        inDegree: n.inDegree ?? 0,
        outDegree: n.outDegree ?? 0,
        totalDegree: (n.inDegree ?? 0) + (n.outDegree ?? 0),
      });
      connections.set(n.nodeId, { in: [], out: [] });
    }

    for (const e of edges) {
      const srcDeg = degrees.get(e.source);
      if (srcDeg) {
        srcDeg.outDegree += 1;
        srcDeg.totalDegree += 1;
      }
      const tgtDeg = degrees.get(e.target);
      if (tgtDeg) {
        tgtDeg.inDegree += 1;
        tgtDeg.totalDegree += 1;
      }
      const srcConn = connections.get(e.source);
      if (srcConn) {
        srcConn.out.push({ id: e.target, weight: e.weight });
      }
      const tgtConn = connections.get(e.target);
      if (tgtConn) {
        tgtConn.in.push({ id: e.source, weight: e.weight });
      }
    }

    const influencers = Array.from(degrees.entries())
      .filter(([_, d]) => d.inDegree > 0)
      .sort((a, b) => b[1].inDegree - a[1].inDegree)
      .slice(0, 5)
      .map(([id, d]) => {
        const n = nodes.find((node) => node.nodeId === id);
        return {
          nodeId: id,
          username: n?.username ?? id.replace(/^(reddit:user:|youtube:channel:)/, ''),
          platform: n?.platform ?? 'unknown',
          inDegree: d.inDegree,
          outDegree: d.outDegree,
        };
      });

    return { nodeDegrees: degrees, nodeConnections: connections, topInfluencers: influencers };
  }, [nodes, edges]);

  // Determine active visible nodes based on viewMode
  const visibleNodes = useMemo(() => {
    if (edges.length === 0 || viewMode === 'all') {
      return nodes;
    }
    // 'connected' mode: only nodes that participate in at least one interaction edge
    return nodes.filter((n) => {
      const deg = nodeDegrees.get(n.nodeId);
      return deg && deg.totalDegree > 0;
    });
  }, [nodes, edges, viewMode, nodeDegrees]);

  // Calculate multi-tiered orbital layout
  const nodePositions = useMemo(() => {
    const map = new Map<string, NodePosition>();
    const count = visibleNodes.length;
    if (count === 0) return map;

    if (count === 1) {
      const n = visibleNodes[0];
      const deg = nodeDegrees.get(n.nodeId) || { inDegree: 0, outDegree: 0, totalDegree: 0 };
      map.set(n.nodeId, {
        x: CX,
        y: CY,
        node: n,
        radius: 20,
        isHub: false,
        totalDegree: deg.totalDegree,
        inDegree: deg.inDegree,
        outDegree: deg.outDegree,
      });
      return map;
    }

    // Partition visible nodes into Hubs, Connected Peers, and Singletons
    const hubs: NetworkNodeItem[] = [];
    const peers: NetworkNodeItem[] = [];
    const singletons: NetworkNodeItem[] = [];

    for (const n of visibleNodes) {
      const deg = nodeDegrees.get(n.nodeId) || { inDegree: 0, outDegree: 0, totalDegree: 0 };
      if (deg.inDegree >= 4 || deg.totalDegree >= 8) {
        hubs.push(n);
      } else if (deg.totalDegree > 0) {
        peers.push(n);
      } else {
        singletons.push(n);
      }
    }

    // Inner Constellation: Key Influencers / Hubs (Radius 130)
    hubs.forEach((n, idx) => {
      const deg = nodeDegrees.get(n.nodeId) || { inDegree: 0, outDegree: 0, totalDegree: 0 };
      const angle = (2 * Math.PI * idx) / (hubs.length || 1) - Math.PI / 2;
      const x = Math.round(CX + 130 * Math.cos(angle));
      const y = Math.round(CY + 130 * Math.sin(angle));
      map.set(n.nodeId, {
        x,
        y,
        node: n,
        radius: 22,
        isHub: true,
        totalDegree: deg.totalDegree,
        inDegree: deg.inDegree,
        outDegree: deg.outDegree,
      });
    });

    // Middle Orbit: Connected Discussant Peers (Radius 240)
    peers.forEach((n, idx) => {
      const deg = nodeDegrees.get(n.nodeId) || { inDegree: 0, outDegree: 0, totalDegree: 0 };
      const angle = (2 * Math.PI * idx) / (peers.length || 1) - Math.PI / 2;
      const x = Math.round(CX + 240 * Math.cos(angle));
      const y = Math.round(CY + 240 * Math.sin(angle));
      map.set(n.nodeId, {
        x,
        y,
        node: n,
        radius: 16,
        isHub: false,
        totalDegree: deg.totalDegree,
        inDegree: deg.inDegree,
        outDegree: deg.outDegree,
      });
    });

    // Outer Orbit Rings: Singletons / Peripheral Accounts (Radius 340, 420)
    const singletonOrbits = [
      { radius: 340, capacity: 32 },
      { radius: 420, capacity: 48 },
    ];
    let sIdx = 0;
    for (const orbit of singletonOrbits) {
      const slice = singletons.slice(sIdx, sIdx + orbit.capacity);
      slice.forEach((n, idx) => {
        const deg = nodeDegrees.get(n.nodeId) || { inDegree: 0, outDegree: 0, totalDegree: 0 };
        const angle = (2 * Math.PI * idx) / (slice.length || 1) - Math.PI / 2;
        const x = Math.round(CX + orbit.radius * Math.cos(angle));
        const y = Math.round(CY + orbit.radius * Math.sin(angle));
        map.set(n.nodeId, {
          x,
          y,
          node: n,
          radius: 13,
          isHub: false,
          totalDegree: deg.totalDegree,
          inDegree: deg.inDegree,
          outDegree: deg.outDegree,
        });
      });
      sIdx += slice.length;
      if (sIdx >= singletons.length) break;
    }

    return map;
  }, [visibleNodes, nodeDegrees, CX, CY]);

  // Handle Zoom
  const handleZoomIn = () => setZoom((z) => Math.min(2.5, Math.round((z + 0.2) * 10) / 10));
  const handleZoomOut = () => setZoom((z) => Math.max(0.4, Math.round((z - 0.2) * 10) / 10));
  const handleReset = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
    setSelectedNodeId(null);
  };

  // Drag to pan
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging) return;
      setPan({
        x: e.clientX - dragStartRef.current.x,
        y: e.clientY - dragStartRef.current.y,
      });
    },
    [isDragging]
  );

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Find active node and connected peers
  const activeFocusId = selectedNodeId || hoveredNodeId;
  const activeNodeConnections = activeFocusId ? nodeConnections.get(activeFocusId) : null;
  const selectedNode = selectedNodeId ? nodePositions.get(selectedNodeId)?.node : null;
  const isZeroEdge = edges.length === 0;

  return (
    <div className="card-panel" style={{ padding: '0', overflow: 'hidden', backgroundColor: '#FFFFFF', marginBottom: '24px' }}>
      {/* Top Toolbar: Legend, View Toggle, Influencer Chips & Zoom Controls */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '12px 20px',
          borderBottom: '1px solid #E2E8F0',
          backgroundColor: '#F8FAFC',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        {/* Left: Legend & View Filter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '12px', color: '#475569', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#FF4500', display: 'inline-block' }} />
            <span>Reddit Author</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#FF0000', display: 'inline-block' }} />
            <span>YouTube Creator</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '18px', height: '2px', backgroundColor: '#2563EB', display: 'inline-block' }} />
            <span>Directed Mention Edge</span>
          </div>

          {/* Subnetwork View Toggle */}
          {!isZeroEdge && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginLeft: '8px', borderLeft: '1px solid #CBD5E1', paddingLeft: '12px' }}>
              <span style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>Topology:</span>
              <button
                type="button"
                onClick={() => setViewMode('connected')}
                style={{
                  padding: '3px 8px',
                  fontSize: '11px',
                  fontWeight: 600,
                  borderRadius: '4px',
                  border: viewMode === 'connected' ? '1px solid #2563EB' : '1px solid #CBD5E1',
                  backgroundColor: viewMode === 'connected' ? '#EFF6FF' : '#FFFFFF',
                  color: viewMode === 'connected' ? '#1E40AF' : '#475569',
                  cursor: 'pointer',
                }}
              >
                Connected Network ({nodes.filter(n => (nodeDegrees.get(n.nodeId)?.totalDegree ?? 0) > 0).length})
              </button>
              <button
                type="button"
                onClick={() => setViewMode('all')}
                style={{
                  padding: '3px 8px',
                  fontSize: '11px',
                  fontWeight: 600,
                  borderRadius: '4px',
                  border: viewMode === 'all' ? '1px solid #2563EB' : '1px solid #CBD5E1',
                  backgroundColor: viewMode === 'all' ? '#EFF6FF' : '#FFFFFF',
                  color: viewMode === 'all' ? '#1E40AF' : '#475569',
                  cursor: 'pointer',
                }}
              >
                All Accounts ({nodes.length})
              </button>
            </div>
          )}
        </div>

        {/* Right: Zoom & Reset Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '11px', color: '#64748B', fontFamily: 'monospace', marginRight: '4px' }}>
            Zoom: {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={handleZoomIn}
            className="refresh-button"
            style={{ padding: '4px 10px', fontSize: '13px', fontWeight: 600 }}
            title="Zoom In"
          >
            +
          </button>
          <button
            type="button"
            onClick={handleZoomOut}
            className="refresh-button"
            style={{ padding: '4px 10px', fontSize: '13px', fontWeight: 600 }}
            title="Zoom Out"
          >
            −
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="refresh-button"
            style={{ padding: '4px 10px', fontSize: '12px' }}
            title="Reset Zoom & Pan"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Top Influencers Quick Focus Strip (when influencers exist) */}
      {topInfluencers.length > 0 && !isZeroEdge && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 20px',
            backgroundColor: '#F1F5F9',
            borderBottom: '1px solid #E2E8F0',
            fontSize: '11px',
            overflowX: 'auto',
          }}
        >
          <span style={{ fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Top Hubs:
          </span>
          {topInfluencers.map((inf) => {
            const isTargetSelected = selectedNodeId === inf.nodeId;
            return (
              <button
                key={inf.nodeId}
                type="button"
                onClick={() => setSelectedNodeId((prev) => (prev === inf.nodeId ? null : inf.nodeId))}
                style={{
                  padding: '3px 10px',
                  borderRadius: '12px',
                  border: isTargetSelected ? '1px solid #2563EB' : '1px solid #CBD5E1',
                  backgroundColor: isTargetSelected ? '#2563EB' : '#FFFFFF',
                  color: isTargetSelected ? '#FFFFFF' : '#1E293B',
                  fontWeight: 600,
                  fontSize: '11px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  whiteSpace: 'nowrap',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                  transition: 'all 0.15s ease',
                }}
              >
                <span>{inf.username}</span>
                <span
                  style={{
                    backgroundColor: isTargetSelected ? 'rgba(255,255,255,0.25)' : '#EFF6FF',
                    color: isTargetSelected ? '#FFFFFF' : '#1E40AF',
                    padding: '1px 5px',
                    borderRadius: '8px',
                    fontSize: '10px',
                  }}
                >
                  {inf.inDegree} in
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* SVG Canvas Container */}
      <div
        style={{
          position: 'relative',
          width: '100%',
          height: '560px',
          backgroundColor: '#FAFAFA',
          cursor: isDragging ? 'grabbing' : 'grab',
          userSelect: 'none',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
          style={{ display: 'block' }}
        >
          {/* Arrowhead Defs */}
          <defs>
            <marker
              id="graph-arrow"
              viewBox="0 0 10 10"
              refX="18"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#3B82F6" />
            </marker>
            <marker
              id="graph-arrow-faded"
              viewBox="0 0 10 10"
              refX="18"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#94A3B8" />
            </marker>
          </defs>

          {/* Transform Group for Pan & Zoom */}
          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`} style={{ transformOrigin: `${CX}px ${CY}px` }}>
            {/* Background grid dots for spatial orientation */}
            <pattern id="dot-grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <circle cx="20" cy="20" r="1" fill="#E2E8F0" />
            </pattern>
            <rect x="-1500" y="-1500" width="4000" height="4000" fill="url(#dot-grid)" pointerEvents="none" />

            {/* Orbital Guide Rings */}
            <circle cx={CX} cy={CY} r="130" fill="none" stroke="#E2E8F0" strokeWidth="1.2" strokeDasharray="4 4" />
            <circle cx={CX} cy={CY} r="240" fill="none" stroke="#E2E8F0" strokeWidth="1.2" strokeDasharray="4 4" />
            {viewMode === 'all' && (
              <>
                <circle cx={CX} cy={CY} r="340" fill="none" stroke="#F1F5F9" strokeWidth="1" strokeDasharray="3 3" />
                <circle cx={CX} cy={CY} r="420" fill="none" stroke="#F1F5F9" strokeWidth="1" strokeDasharray="3 3" />
              </>
            )}

            {/* Directed Curved Edges */}
            {edges.map((edge, idx) => {
              const src = nodePositions.get(edge.source);
              const tgt = nodePositions.get(edge.target);
              if (!src || !tgt) return null;

              const isDirectlyConnected =
                edge.source === activeFocusId || edge.target === activeFocusId;
              const hasActiveFocus = Boolean(activeFocusId);
              const isDimmed = hasActiveFocus && !isDirectlyConnected;

              // Curved Bézier path to separate bidirectional mentions cleanly
              const dx = tgt.x - src.x;
              const dy = tgt.y - src.y;
              const dist = Math.sqrt(dx * dx + dy * dy) || 1;
              const curvature = 16;
              const cx = (src.x + tgt.x) / 2 - (dy / dist) * curvature;
              const cy = (src.y + tgt.y) / 2 + (dx / dist) * curvature;
              const pathD = `M ${src.x} ${src.y} Q ${cx} ${cy} ${tgt.x} ${tgt.y}`;

              return (
                <path
                  key={`edge-${idx}-${edge.source}-${edge.target}`}
                  d={pathD}
                  fill="none"
                  stroke={isDirectlyConnected ? '#2563EB' : isDimmed ? '#CBD5E1' : '#64748B'}
                  strokeWidth={isDirectlyConnected ? 2.5 : 1.2}
                  strokeOpacity={isDirectlyConnected ? 0.95 : isDimmed ? 0.15 : 0.45}
                  markerEnd={isDirectlyConnected ? 'url(#graph-arrow)' : 'url(#graph-arrow-faded)'}
                />
              );
            })}

            {/* Nodes */}
            {Array.from(nodePositions.values()).map((pos) => {
              const { node, x, y, radius, isHub, inDegree } = pos;
              const isSelected = selectedNodeId === node.nodeId;
              const isHovered = hoveredNodeId === node.nodeId;
              const isDirectPeer =
                activeNodeConnections &&
                (activeNodeConnections.in.some((c) => c.id === node.nodeId) ||
                  activeNodeConnections.out.some((c) => c.id === node.nodeId));
              const hasActiveFocus = Boolean(activeFocusId);
              const isFocused = isSelected || isHovered;
              const isDimmed = hasActiveFocus && !isFocused && !isDirectPeer;

              const isReddit = node.platform.toLowerCase() === 'reddit';
              const nodeColor = isReddit ? '#FF4500' : '#FF0000';
              const displayName = node.username.length > 15 ? `${node.username.slice(0, 13)}…` : node.username;

              return (
                <g
                  key={node.nodeId}
                  transform={`translate(${x}, ${y})`}
                  style={{
                    cursor: 'pointer',
                    opacity: isDimmed ? 0.2 : 1.0,
                    transition: 'opacity 0.15s ease',
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedNodeId((prev) => (prev === node.nodeId ? null : node.nodeId));
                  }}
                  onMouseEnter={() => setHoveredNodeId(node.nodeId)}
                  onMouseLeave={() => setHoveredNodeId(null)}
                >
                  {/* Hub Ring Glow */}
                  {isHub && (
                    <circle
                      r={radius + 5}
                      fill="none"
                      stroke="#F59E0B"
                      strokeWidth="2"
                      strokeDasharray="3 3"
                      strokeOpacity={0.8}
                    />
                  )}

                  {/* Selected / Focused Halo */}
                  {isFocused && (
                    <circle
                      r={radius + 7}
                      fill="none"
                      stroke="#2563EB"
                      strokeWidth={3}
                      strokeOpacity={0.9}
                    />
                  )}

                  {/* Connected Peer Halo */}
                  {isDirectPeer && !isFocused && (
                    <circle
                      r={radius + 4}
                      fill="none"
                      stroke="#3B82F6"
                      strokeWidth={2}
                      strokeDasharray="2 2"
                      strokeOpacity={0.7}
                    />
                  )}

                  {/* Main Node Circle */}
                  <circle
                    r={radius}
                    fill={nodeColor}
                    stroke="#FFFFFF"
                    strokeWidth="2.5"
                    style={{
                      filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))',
                      transition: 'transform 0.15s ease',
                    }}
                  />

                  {/* Inner Platform Initial */}
                  <text
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="#FFFFFF"
                    fontSize={radius > 16 ? '11' : '9'}
                    fontWeight="700"
                    pointerEvents="none"
                  >
                    {isReddit ? 'r/' : 'yt'}
                  </text>

                  {/* Username Label */}
                  <text
                    y={radius + 14}
                    textAnchor="middle"
                    fill={isFocused ? '#1D4ED8' : '#334155'}
                    fontSize={isHub ? '11' : '10'}
                    fontWeight={isFocused || isHub ? '700' : '500'}
                    pointerEvents="none"
                    style={{
                      textShadow: '0 1px 2px rgba(255,255,255,0.95), 0 -1px 2px rgba(255,255,255,0.95)',
                    }}
                  >
                    {displayName}
                  </text>

                  {/* In-Degree Badge for Hubs */}
                  {isHub && inDegree > 0 && (
                    <g transform={`translate(${radius - 2}, ${-radius + 2})`}>
                      <circle r="7" fill="#F59E0B" stroke="#FFFFFF" strokeWidth="1.5" />
                      <text
                        textAnchor="middle"
                        dominantBaseline="central"
                        fill="#FFFFFF"
                        fontSize="8"
                        fontWeight="800"
                        pointerEvents="none"
                      >
                        {inDegree}
                      </text>
                    </g>
                  )}
                </g>
              );
            })}
          </g>
        </svg>

        {/* Zero-Edge Diagnostic Pill inside canvas */}
        {isZeroEdge && (
          <div
            style={{
              position: 'absolute',
              bottom: '12px',
              left: '50%',
              transform: 'translateX(-50%)',
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #CBD5E1',
              borderRadius: '20px',
              padding: '6px 16px',
              fontSize: '11px',
              color: '#475569',
              boxShadow: '0 2px 6px rgba(0, 0, 0, 0.05)',
              pointerEvents: 'none',
              textAlign: 'center',
            }}
          >
            Disconnected Topology: <strong>{nodes.length} isolated singletons</strong> with <strong>0 interaction edges</strong> (|E| = 0)
          </div>
        )}

        {/* Floating Node Inspector Drawer (shown when node clicked) */}
        {selectedNode && (
          <div
            style={{
              position: 'absolute',
              top: '16px',
              right: '16px',
              width: '300px',
              maxHeight: '520px',
              overflowY: 'auto',
              backgroundColor: '#FFFFFF',
              borderRadius: '8px',
              border: '1px solid #CBD5E1',
              boxShadow: '0 4px 16px rgba(0, 0, 0, 0.12)',
              padding: '16px',
              zIndex: 10,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748B', textTransform: 'uppercase' }}>
                Account Inspector
              </span>
              <button
                type="button"
                onClick={() => setSelectedNodeId(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '14px',
                  color: '#94A3B8',
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                ✕
              </button>
            </div>

            <div style={{ fontSize: '15px', fontWeight: 700, color: '#0F172A', marginBottom: '4px' }}>
              {selectedNode.username}
            </div>
            <div style={{ fontSize: '11px', color: '#94A3B8', fontFamily: 'monospace', marginBottom: '12px' }}>
              {selectedNode.nodeId}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569' }}>
                <span>Platform:</span>
                <strong style={{ color: selectedNode.platform.toLowerCase() === 'reddit' ? '#FF4500' : '#FF0000', textTransform: 'capitalize' }}>
                  {selectedNode.platform}
                </strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569' }}>
                <span>Identity Type:</span>
                <strong style={{ color: '#0F172A' }}>{selectedNode.identityType}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569' }}>
                <span>Activity Count:</span>
                <strong style={{ color: '#0F172A' }}>{selectedNode.activityCount} submission(s)</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569' }}>
                <span>In-Degree (Mentions received):</span>
                <strong style={{ color: '#0F172A' }}>{nodeDegrees.get(selectedNode.nodeId)?.inDegree ?? selectedNode.inDegree ?? 0}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569' }}>
                <span>Out-Degree (Mentions sent):</span>
                <strong style={{ color: '#0F172A' }}>{nodeDegrees.get(selectedNode.nodeId)?.outDegree ?? selectedNode.outDegree ?? 0}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569' }}>
                <span>Topological Role:</span>
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 600,
                    padding: '2px 8px',
                    borderRadius: '4px',
                    backgroundColor:
                      (nodeDegrees.get(selectedNode.nodeId)?.inDegree ?? 0) >= 4
                        ? '#FEF3C7'
                        : (nodeDegrees.get(selectedNode.nodeId)?.totalDegree ?? 0) > 0
                        ? '#EFF6FF'
                        : '#F1F5F9',
                    color:
                      (nodeDegrees.get(selectedNode.nodeId)?.inDegree ?? 0) >= 4
                        ? '#B45309'
                        : (nodeDegrees.get(selectedNode.nodeId)?.totalDegree ?? 0) > 0
                        ? '#1E40AF'
                        : '#64748B',
                  }}
                >
                  {(nodeDegrees.get(selectedNode.nodeId)?.inDegree ?? 0) >= 4
                    ? 'Key Hub Influencer'
                    : (nodeDegrees.get(selectedNode.nodeId)?.totalDegree ?? 0) > 0
                    ? 'Connected Peer'
                    : 'Isolated Singleton'}
                </span>
              </div>
            </div>

            {/* Direct Interaction Details */}
            {activeNodeConnections && (activeNodeConnections.in.length > 0 || activeNodeConnections.out.length > 0) && (
              <div style={{ marginTop: '14px', borderTop: '1px solid #E2E8F0', paddingTop: '10px' }}>
                <div style={{ fontSize: '11px', fontWeight: 700, color: '#475569', marginBottom: '6px', textTransform: 'uppercase' }}>
                  Direct Interactions ({activeNodeConnections.in.length + activeNodeConnections.out.length})
                </div>

                {activeNodeConnections.in.length > 0 && (
                  <div style={{ marginBottom: '8px' }}>
                    <span style={{ fontSize: '11px', color: '#16A34A', fontWeight: 600 }}>Mentioned by:</span>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '4px' }}>
                      {activeNodeConnections.in.slice(0, 8).map((c) => {
                        const targetUser = c.id.replace(/^(reddit:user:|youtube:channel:)/, '');
                        return (
                          <span
                            key={c.id}
                            onClick={() => setSelectedNodeId(c.id)}
                            style={{
                              fontSize: '11px',
                              color: '#16A34A',
                              cursor: 'pointer',
                              fontWeight: 600,
                              padding: '1px 2px',
                            }}
                            title={`Click to inspect ${targetUser}`}
                          >
                            @{targetUser}
                          </span>
                        );
                      })}
                      {activeNodeConnections.in.length > 8 && (
                        <span style={{ fontSize: '10px', color: '#64748B', padding: '1px 2px' }}>
                          +{activeNodeConnections.in.length - 8} more
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {activeNodeConnections.out.length > 0 && (
                  <div>
                    <span style={{ fontSize: '11px', color: '#2563EB', fontWeight: 600 }}>Mentioned others:</span>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '4px' }}>
                      {activeNodeConnections.out.slice(0, 8).map((c) => {
                        const targetUser = c.id.replace(/^(reddit:user:|youtube:channel:)/, '');
                        return (
                          <span
                            key={c.id}
                            onClick={() => setSelectedNodeId(c.id)}
                            style={{
                              fontSize: '11px',
                              color: '#2563EB',
                              cursor: 'pointer',
                              fontWeight: 600,
                              padding: '1px 2px',
                            }}
                            title={`Click to inspect ${targetUser}`}
                          >
                            @{targetUser}
                          </span>
                        );
                      })}
                      {activeNodeConnections.out.length > 8 && (
                        <span style={{ fontSize: '10px', color: '#64748B', padding: '1px 2px' }}>
                          +{activeNodeConnections.out.length - 8} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div style={{ padding: '10px 20px', backgroundColor: '#F8FAFC', borderTop: '1px solid #E2E8F0', fontSize: '11px', color: '#64748B', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>
          Click any account node or top influencer chip to inspect directed interactions. Drag canvas to pan, use + / − to zoom.
        </span>
        <span style={{ fontFamily: 'monospace' }}>
          Platform filter: {platformFilter} | Showing: {visibleNodes.length} / {nodes.length} nodes
        </span>
      </div>
    </div>
  );
};
