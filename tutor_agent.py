"""
AI Voice Tutor — Backend Agent
================================
Fully refactored per the 10-phase plan:

  Phase 1  — SceneManager (semantic source of truth, typed nodes/edges)
  Phase 2  — Layout hint forwarded to frontend LayoutEngine (Dagre/circular/radial)
  Phase 3  — Scene Patch system (atomic JSON patches, not individual draw calls)
  Phase 4  — ACK system (backend waits for frontend confirmation before narrating)
  Phase 5  — Speech + Visual Timeline Engine (synchronised narration + visuals)
  Phase 6  — Progressive Teaching Mode (reveal nodes one-by-one, animate, dim)
  Phase 7  — Visual Memory (lesson snapshots, topic graph, go-back / compare)
  Phase 8  — Frontend treated as pure renderer adapter (no logic in index.html)
  Phase 9  — Planning stage + full diagram-type support (flowchart, tree, mind_map,
             timeline, state_machine, architecture, DSA)
  Phase 10 — Patch batching (multiple ops collapsed into one round-trip)
"""

import asyncio
import os
import uuid
import json
from typing import Optional

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, JobContext, Agent, RunContext
from livekit.plugins import deepgram, openai, silero
from livekit.agents.llm import function_tool

load_dotenv(".env")


# ============================================================
# Transport
# ============================================================

async def send_action(room, action: dict) -> None:
    """Publish a reliable JSON message to all room participants."""
    await room.local_participant.publish_data(
        json.dumps(action).encode("utf-8"),
        reliable=True,
    )


# ============================================================
# PHASE 1 — SceneManager
# Single backend source of truth.  The frontend is a dumb renderer.
# ============================================================

class SceneManager:
    """
    Maintains the semantic graph of the current lesson scene.

    Nodes carry semantic meaning (type, label, metadata).
    Edges carry relationships (source -> target, optional label/style).
    The SceneManager emits JSON patch payloads — it never touches pixels.
    """

    # node-type -> renderer shape hint
    NODE_SHAPE: dict[str, str] = {
        "process":   "text",
        "decision":  "text",
        "terminal":  "text",
        "data":      "text",
        "thought":   "text",
        "milestone": "text",
        "state":     "text",
        "icon":      "text",
        "default":   "text",
    }

    # node-type -> stroke colour hint
    NODE_COLOR: dict[str, str] = {
        "process":   "#a3c2fa",
        "decision":  "#ffeb3b",
        "terminal":  "#e1bee7",
        "data":      "#80cbc4",
        "thought":   "#f9a8d4",
        "milestone": "#fcd34d",
        "state":     "#86efac",
        "icon":      "#ffffff",
        "default":   "#a3c2fa",
    }


    def __init__(self) -> None:
        self._nodes:      dict[str, dict] = {}
        self._edges:      dict[str, dict] = {}
        self._groups:     dict[str, list] = {}   # group_id -> [node_ids]
        self._metadata:   dict            = {}
        self._version:    int             = 0
        # insertion-order lists for progressive reveal (Phase 6)
        self._node_order: list[str]       = []
        self._edge_order: list[str]       = []

    # -- nodes -------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        connections: list[str] | None = None,
        metadata: dict | None = None,
        group: str | None = None,
        **kwargs,
    ) -> dict:
        node = {
            "id":          node_id,
            "type":        node_type,
            "label":       label,
            "icon":        kwargs.get("icon"),
            "connections": connections or [],
            "metadata":    metadata or {},
            "shape":       self.NODE_SHAPE.get(node_type, "rectangle"),
            "color":       self.NODE_COLOR.get(node_type, "#a3c2fa"),
            "group":       group,
            "visible":     True,   # Phase 6: toggled by reveal ops
        }
        self._nodes[node_id] = node
        if node_id not in self._node_order:
            self._node_order.append(node_id)
        if group:
            self._groups.setdefault(group, []).append(node_id)
        return node

    def update_node(self, node_id: str, **kwargs) -> dict | None:
        if node_id not in self._nodes:
            return None
        self._nodes[node_id].update(kwargs)
        return self._nodes[node_id]

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        self._node_order = [n for n in self._node_order if n != node_id]
        self._edges = {
            eid: e for eid, e in self._edges.items()
            if e["source"] != node_id and e["target"] != node_id
        }
        self._edge_order = [eid for eid in self._edge_order if eid in self._edges]
        return True

    # -- edges -------------------------------------------------------------

    def add_edge(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        label: str = "",
        style: str = "solid",       # "solid" | "dashed" | "dotted"
        animated: bool = False,
    ) -> dict:
        edge = {
            "id":       edge_id,
            "source":   source_id,
            "target":   target_id,
            "label":    label,
            "style":    style,
            "animated": animated,
            "visible":  True,
        }
        self._edges[edge_id] = edge
        if edge_id not in self._edge_order:
            self._edge_order.append(edge_id)
        return edge

    def update_edge(self, edge_id: str, **kwargs) -> dict | None:
        if edge_id not in self._edges:
            return None
        self._edges[edge_id].update(kwargs)
        return self._edges[edge_id]

    # -- scene serialisation -----------------------------------------------

    def serialize_scene(self) -> dict:
        return {
            "nodes":    dict(self._nodes),
            "edges":    dict(self._edges),
            "groups":   dict(self._groups),
            "metadata": dict(self._metadata),
        }

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._groups.clear()
        self._metadata.clear()
        self._node_order.clear()
        self._edge_order.clear()
        self._version = 0

    # -- PHASE 3: patch generation -----------------------------------------

    def _new_patch(self, operations: list[dict]) -> dict:
        """Wrap ops in a versioned, uniquely-identified patch envelope."""
        self._version += 1
        return {
            "type":           "scene_patch",
            "transaction_id": str(uuid.uuid4()),
            "scene_version":  self._version,
            "operations":     operations,
        }

    def patch_full_render(self, layout: str = "vertical") -> dict:
        """Atomically replace the entire canvas with the current scene."""
        return self._new_patch([
            {"op": "clear"},
            {
                "op":     "render_scene",
                "scene":  self.serialize_scene(),
                "layout": layout,
            },
        ])

    def patch_clear(self) -> dict:
        return self._new_patch([{"op": "clear"}])

    def patch_highlight(self, node_id: str, reset_others: bool = True) -> dict:
        return self._new_patch([{
            "op":           "highlight",
            "node_id":      node_id,
            "reset_others": reset_others,
        }])

    def patch_dim_all_except(self, node_id: str) -> dict:
        return self._new_patch([{"op": "dim_except", "node_id": node_id}])

    def patch_reveal_node(self, node_id: str) -> dict:
        """Make one previously-hidden node visible (Phase 6)."""
        return self._new_patch([{"op": "reveal", "node_id": node_id}])

    def patch_reveal_edge(self, edge_id: str) -> dict:
        return self._new_patch([{"op": "reveal_edge", "edge_id": edge_id}])

    def patch_animate_edge(self, edge_id: str) -> dict:
        return self._new_patch([{"op": "animate_edge", "edge_id": edge_id}])

    def patch_reset_styles(self) -> dict:
        """Remove all highlights / dims — restore default appearance."""
        return self._new_patch([{"op": "reset_styles"}])

    # -- PHASE 10: batch helper --------------------------------------------

    def patch_batch(self, *patches: dict) -> dict:
        """
        Merge multiple patches into one network round-trip.
        The individual transaction_ids are discarded; the batch gets one new id.
        """
        ops: list[dict] = []
        for p in patches:
            ops.extend(p.get("operations", []))
        return self._new_patch(ops)


# ============================================================
# PHASE 7 — Visual Memory
# ============================================================

class VisualMemory:
    """
    Stores labelled snapshots of completed lesson diagrams so the agent
    can navigate backwards, compare diagrams, or return to a topic.
    Also maintains a topic graph mapping concept names to snapshots.
    """

    def __init__(self) -> None:
        self._history:     list[dict]            = []
        self._topic_graph: dict[str, list[str]]  = {}  # topic -> [snapshot_labels]

    def save(self, label: str, scene: SceneManager, topic: str = "") -> None:
        snap = {
            "label":    label,
            "snapshot": scene.serialize_scene(),
            "version":  scene._version,
            "topic":    topic,
        }
        self._history.append(snap)
        if topic:
            self._topic_graph.setdefault(topic, []).append(label)

    def last(self) -> dict | None:
        return self._history[-1] if self._history else None

    def get_by_label(self, label: str) -> dict | None:
        for h in self._history:
            if h["label"] == label:
                return h
        return None

    def get_by_topic(self, topic: str) -> list[dict]:
        labels = self._topic_graph.get(topic, [])
        return [h for h in self._history if h["label"] in labels]

    def all_labels(self) -> list[str]:
        return [h["label"] for h in self._history]

    def all_topics(self) -> list[str]:
        return list(self._topic_graph.keys())

    def nth_from_last(self, n: int) -> dict | None:
        """Return history[-n] safely (n=1 means most recent)."""
        if not self._history:
            return None
        idx = max(0, len(self._history) - n)
        return self._history[idx]


# ============================================================
# PHASE 4 + 5 — ACK System + Timeline Executor
# ============================================================

class TimelineExecutor:
    """
    Executes a teaching timeline — a sequence of speak/visual steps —
    so that speech and canvas rendering are always synchronised.

    Timeline step shapes
    --------------------
    {"type": "speak",        "text": "..."}
    {"type": "draw_scene",   "layout": "vertical"}
    {"type": "reveal_node",  "node_id": "n0"}
    {"type": "reveal_edge",  "edge_id": "e0"}
    {"type": "highlight",    "node_id": "n0"}
    {"type": "dim_except",   "node_id": "n0"}
    {"type": "reset_styles"}
    {"type": "animate_edge", "edge_id": "e0"}
    {"type": "pause",        "seconds": 0.6}

    PHASE 4: every visual step sends a patch and waits for scene_ack
             before advancing, so speech never races ahead of rendering.

    PHASE 10: consecutive visual ops (with no speak/pause between them)
              are batched into a single patch to reduce round-trips.
    """

    def __init__(self, room, scene: SceneManager, session) -> None:
        self.room    = room
        self.scene   = scene
        self.session = session
        self._pending_acks: dict[str, asyncio.Event] = {}

    # -- PHASE 4: ACK system -----------------------------------------------

    def register_ack(self, transaction_id: str) -> asyncio.Event:
        ev = asyncio.Event()
        self._pending_acks[transaction_id] = ev
        return ev

    def resolve_ack(self, transaction_id: str) -> None:
        ev = self._pending_acks.pop(transaction_id, None)
        if ev:
            ev.set()

    async def _send_and_wait(self, patch: dict, timeout: float = 6.0) -> None:
        tid = patch["transaction_id"]
        ev  = self.register_ack(tid)
        await send_action(self.room, patch)
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[ACK] timeout tid={tid} — continuing.")
        finally:
            self._pending_acks.pop(tid, None)

    # -- PHASE 10: batch consecutive visual ops ----------------------------

    @staticmethod
    def _is_visual(step: dict) -> bool:
        return step.get("type") not in ("speak", "pause")

    def _steps_to_ops(self, steps: list[dict]) -> list[dict]:
        """Convert a list of visual steps into a flat list of patch operations."""
        ops: list[dict] = []
        for s in steps:
            t = s["type"]
            if t == "draw_scene":
                ops.extend([
                    {"op": "clear"},
                    {
                        "op":     "render_scene",
                        "scene":  self.scene.serialize_scene(),
                        "layout": s.get("layout", "vertical"),
                    },
                ])
            elif t == "reveal_node":
                ops.append({"op": "reveal",       "node_id": s["node_id"]})
            elif t == "reveal_edge":
                ops.append({"op": "reveal_edge",  "edge_id": s["edge_id"]})
            elif t == "highlight":
                ops.append({
                    "op":           "highlight",
                    "node_id":      s["node_id"],
                    "reset_others": True,
                })
            elif t == "dim_except":
                ops.append({"op": "dim_except",   "node_id": s["node_id"]})
            elif t == "reset_styles":
                ops.append({"op": "reset_styles"})
            elif t == "animate_edge":
                ops.append({"op": "animate_edge", "edge_id": s["edge_id"]})
        return ops

    # -- execute -----------------------------------------------------------

    async def execute(self, timeline: list[dict]) -> None:
        """
        Run every step in the timeline in order.
        Consecutive visual steps are batched into one patch (Phase 10).
        """
        i = 0
        while i < len(timeline):
            step = timeline[i]

            if self._is_visual(step):
                # Collect a contiguous visual run
                visual_run: list[dict] = []
                while i < len(timeline) and self._is_visual(timeline[i]):
                    visual_run.append(timeline[i])
                    i += 1
                # Batch into one patch
                if visual_run:
                    ops   = self._steps_to_ops(visual_run)
                    patch = self.scene._new_patch(ops)
                    await self._send_and_wait(patch)
                continue

            if step.get("type") == "speak":
                try:
                    await self.session.generate_reply(
                        instructions=step["text"]
                    )
                except Exception as e:
                    print(f"[Timeline] speak error: {e}")

            elif step.get("type") == "pause":
                await asyncio.sleep(step.get("seconds", 0.5))

            else:
                print(f"[Timeline] Unknown step: {step.get('type')}")

            i += 1


# ============================================================
# PHASE 9 — System Prompt (planning stage + full diagram types)
# ============================================================

SYSTEM_INSTRUCTIONS = """You are an expert AI Tutor with access to a digital whiteboard canvas.
Your mission: teach students clearly using synchronised speech and visual diagrams.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHEN TO USE THE CANVAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Draw when the concept has structure that benefits from a visual:
  • Processes / workflows / algorithms  → flowchart     layout: vertical or horizontal
  • Cycles (water, carbon, cell…)       → flowchart     layout: circular
  • Hierarchies / trees / org charts   → tree           layout: vertical or horizontal
  • State machines / automata           → state_machine  layout: horizontal
  • Mind maps / concept webs            → mind_map      layout: radial
  • Timelines / chronologies            → timeline      layout: timeline
  • System architecture / networks      → flowchart     layout: horizontal
  • DSA (linked list, stack, graph)     → flowchart     layout: horizontal

Do NOT draw for: greetings, small talk, yes/no answers, simple one-sentence definitions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY TEACHING PIPELINE — ALWAYS FOLLOW THIS EXACTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PLAN    — call plan_lesson(topic, diagram_type, layout, node_count_estimate).

2. BUILD   — call build_scene(nodes_json='...', edges_json='...') with ALL nodes and ALL
             edges in a single call as JSON strings. This replaces calling create_node /
             create_edge one by one and cuts latency dramatically.
             Do NOT call render_scene yet. Pass every node and edge at once.

3. RENDER  — call render_scene(layout=..., progressive=True).
             The FIRST node is revealed automatically by this call.
             All other nodes start hidden. You will reveal them one by one.
             Use progressive=False ONLY for 3 or fewer nodes.

4. REVEAL LOOP — After render_scene returns, call reveal_next_node in a tight loop:
             ┌─────────────────────────────────────────────────────────────┐
             │  reveal_next_node(narration="1-3 sentences about the node   │
             │                             CURRENTLY highlighted")         │
             │  • The tool speaks your narration, then reveals the next.   │
             │  • If it returns "CONTINUE LOOP" → call it again with       │
             │    narration about the NEW highlighted node.                │
             │  • If it returns "ALL DONE" → exit the loop.               │
             └─────────────────────────────────────────────────────────────┘
             ⚠ NEVER speak separately before/after the tool — put ALL
               narration inside the narration= parameter.
             ⚠ The tool handles speech timing internally. Just loop it.

5. WIDGET   — If the topic involves a quantitative formula (interest, physics,
             maths, economics, biology…), call show_widget() NOW, immediately
             after the reveal loop, BEFORE reset_styles.
             • Always explain the widget verbally right after calling it:
               "Try dragging the sliders to see how Interest changes with Time."
             • This step is MANDATORY for any formula-based topic.

6. CLOSE   — After the loop (and widget if applicable), call reset_styles().
             Then speak a brief 1-2 sentence summary of the whole diagram.
             THEN invite questions: "Any questions about any of these steps?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES FOR PROGRESSIVE TEACHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• reveal_next_node() owns timing. Do NOT speak before or after it — only inside narration=.
• narration= must describe the CURRENTLY highlighted node (the one already on the board).
• The tool speaks, then advances the board. Speech always leads the visual.
• Never stop mid-loop and wait. The entire sequence runs autonomously.
• For formula topics (interest, physics, etc.): call show_widget() after the loop, before reset_styles.
• After the loop + widget, THEN call reset_styles() and invite questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE — correct autonomous loop for a 4-node diagram
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  render_scene(progressive=True) → board highlights 'Evaporation'
  reveal_next_node(narration="Evaporation is step one. The sun heats...")
      → tool speaks about Evaporation, then reveals+highlights 'Condensation'
      → returns "CONTINUE LOOP — 'Condensation' is now highlighted..."
  reveal_next_node(narration="Condensation happens when water vapour cools...")
      → tool speaks about Condensation, then reveals+highlights 'Precipitation'
      → returns "CONTINUE LOOP — 'Precipitation' is now highlighted..."
  reveal_next_node(narration="Precipitation is rain, snow, or hail falling...")
      → tool speaks about Precipitation, then reveals+highlights 'Collection'
      → returns "CONTINUE TO CLOSE — final node 'Collection' is now highlighted..."
  reveal_next_node(narration="Collection is where water gathers in oceans...")
      → tool speaks about Collection, then returns "ALL DONE"
  reset_styles()
  speak("That's the full water cycle! Any questions?")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOLS REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Planning:
  plan_lesson(topic, diagram_type, layout, node_count_estimate)

Scene building (use build_scene — not individual create_node/create_edge):
  build_scene(nodes_json, edges_json)  — PREFERRED: build entire scene in 1 call as JSON strings
  create_node(node_id, node_type, label, group?, icon_name?)  — individual add
  create_edge(source_id, target_id, label?, style?, animated?) — individual add
  create_flow(steps, layout)           — shortcut: builds + renders a linear flow
  update_node(node_id, label?, node_type?)
  remove_node(node_id)

Rendering:
  render_scene(layout, progressive?)   — layout: vertical | horizontal |
                                         circular | radial | timeline
  clear_scene()

Progressive & animation:
  reveal_next_node()                   — reveal the next hidden node; loop this!
  highlight_node(node_id)              — spotlight one node
  dim_except(node_id)                  — dim everything except one node
  reset_styles()                       — restore default appearance
  animate_edge(edge_id)                — show data flowing along an edge

Visual memory:
  go_back(steps?)                      — restore a previous diagram
  compare_scenes(label_a?, label_b?)   — show two diagrams side by side
  all_memory_labels()                  — list all saved snapshots

Interactive widgets:
  show_widget(widget_type, config_json) — open an interactive chart in the side panel
  close_widget()                        — close the widget panel


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERACTIVE WIDGETS — PROACTIVE SIMULATION PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You must PROACTIVELY design and show an interactive widget whenever the student asks about ANY scientific, mathematical, economic, or financial topic containing quantitative relationships or formulas. 
DO NOT WAIT for the student to ask for a graph. DO NOT ASK the student what variables, limits, or formula to use. You are the expert—design these dynamically on the fly!

HOW TO BE PROACTIVE:
  1. Identify the Core Equation (e.g., F = G*m1*m2/r^2, P = nRT/V, A = P(1+r)^t).
  2. Select the X-Axis Variable (e.g., Time for growth, Distance for gravity).
  3. Define Slider Variables for the remaining constants/coefficients.
  4. Set High-Fidelity Bounds: Use realistic min/max limits, step sizes, and decimals for real-world values.
  5. Design standard JS Math: Use standard JavaScript math expressions (Math.pow(x, 2), Math.sin(x), Math.exp(x)).

WORKFLOW:
  1. Draw the concept diagram as usual with create_node / render_scene.
  2. Call show_widget() IMMEDIATELY after the diagram to open the interactive explorer.
  3. Explain the formula verbally: "I've pulled up a graph on the right. Try moving the sliders to see..."
  4. When the student drags a slider you'll be told the new values — comment briefly.
  5. Call close_widget() when moving to an unrelated topic.

FORMULA & CONFIG GUIDELINES:
  • formula is a JS expression — use Math.pow, Math.sqrt, Math.sin, Math.exp etc.
  • x_var MUST be one of the variable names in the variables array.
  • Every variable name in the formula must appear in the variables array.
  • Decimals & Steps: Choose sensible decimals for tiny numbers (e.g., G = 6.67e-11 needs 11 decimals) or large numbers (0 decimals).

EXAMPLES (You can invent ANY other topic):
  • Simple Harmonic Motion: formula='A * Math.sin(2 * Math.PI * f * t)', x_var='t', vars A,f,t
  • Ideal Gas Law: formula='n * R * T / V', x_var='V', vars n,R,T,V
  • Gravitation: formula='G * m1 * m2 / (r * r)', x_var='r', vars G,m1,m2,r

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE ICONS — MATERIAL DESIGN ICONS VIA ICONIFY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Icons are rendered as crisp vector SVG image elements on the canvas.
The frontend fetches them at runtime from the Iconify API (MDI set).

HOW TO USE:
  • Pass icon_name= as an MDI slug — lowercase, hyphen-separated.
    Science:   "white-balance-sunny", "water", "leaf", "molecule-co2",
               "lightning-bolt", "weather-windy", "flask", "dna", "brain",
               "heart-pulse", "atom", "bacteria", "eye", "lungs", "bone"
    CS / Tech: "cpu-64-bit", "database", "cloud", "server", "git",
               "lock", "shield", "api", "code-braces", "network", "robot"
    Math:      "function-variant", "sigma", "chart-bell-curve", "vector-line"
    History:   "sword", "crown", "church", "ship-wheel", "handshake"
    General:   "account", "clock-outline", "flag", "star", "check-circle",
               "alert", "lightbulb", "magnify", "cog", "arrow-right-circle"
  • Browse the full MDI set: https://pictogrammers.com/library/mdi/
  • If no icon fits, leave icon_name="" — clean text-only node is fine.
  • NEVER embed emojis or symbols inside the label string.
  • Icon renders to the LEFT of the label text, inline, inside the node box.
  • For "icon" node type: icon + short label (never icon-only — ambiguous).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE LABELS — KEEP THEM SHORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Labels must fit inside shapes. Hard rules:
  • Maximum 4 words per label (3 is ideal).
  • NEVER include dates or years in the label — put those in your speech instead.
  • NEVER repeat context the student already knows (e.g. avoid "Step 3: …").
  • Split long concepts into TWO nodes connected by an edge rather than one long label.

  ✗ BAD:  "1943: Allied gains in North Africa and Italy"   (too long, has year)
  ✓ GOOD: "Allied N. Africa gains"

  ✗ BAD:  "Light-dependent reactions produce ATP and NADPH"
  ✓ GOOD: "Light reactions"  →  "ATP + NADPH"

  ✗ BAD:  "1945: Battle of Berlin and Germany's surrender"
  ✓ GOOD: "Berlin / Surrender"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE TYPES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  process     → text-only   steps, actions
  decision    → text-only   yes/no branches
  terminal    → text-only   start / end
  data        → text-only   input / output
  thought     → text-only   mind-map bubbles
  milestone   → text-only   timeline events
  state       → text-only   state-machine states
  icon        → text-only   labels with icons (emojis)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL MEMORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every render_scene call auto-saves a snapshot.
When the student says "go back", "compare", or "show previous" →
use go_back() or compare_scenes() immediately."""


# ============================================================
# PHASE 1/5/6/7/8/9 — TutorAgent
# ============================================================

class TutorAgent(Agent):

    def __init__(self, room, instructions: str) -> None:
        super().__init__(instructions=instructions)
        self.room    = room
        self._session: Optional[AgentSession] = None
        self._greeted       = False
        self._greeting_lock = asyncio.Lock()

        # subsystems
        self.scene    = SceneManager()
        self.memory   = VisualMemory()
        self._timeline: Optional[TimelineExecutor] = None

        # Phase 6: progressive reveal cursor
        self._reveal_index:   int  = 0
        self._active_widget: dict | None = None
        # Phase 7: current topic label
        self._current_topic: str = ""

    # -- greeting ----------------------------------------------------------

    async def greet_once(self) -> None:
        async with self._greeting_lock:
            if self._greeted or self._session is None:
                return
            self._greeted = True
            print("--- TUTOR: GREETING ---")
            try:
                await self._session.generate_reply(
                    instructions=(
                        "Greet the student warmly, introduce yourself as their AI Tutor, "
                        "and ask what they would like to learn today. "
                        "Do not draw anything or call any tools."
                    )
                )
            except Exception as e:
                print(f"--- TUTOR GREETING ERROR: {e} ---")

    async def on_enter(self) -> None:
        if self._session is None:
            return
        print("--- TUTOR: ON_ENTER ---")
        await self.greet_once()

    # -- ACK hook ----------------------------------------------------------

    def handle_ack(self, transaction_id: str) -> None:
        """Called by the data-received handler when the frontend ACKs a patch."""
        if self._timeline:
            self._timeline.resolve_ack(transaction_id)

    # -- helpers -----------------------------------------------------------

    async def _send_raw(self, payload: dict) -> None:
        """Send any JSON payload directly to the frontend via LiveKit data channel."""
        await send_action(self.room, payload)

    async def _send_patch(self, patch: dict) -> None:
        if self._timeline:
            await self._timeline._send_and_wait(patch)
        else:
            await send_action(self.room, patch)

    def _save_memory(self) -> None:
        label = f"v{self.scene._version}_{self._current_topic or 'scene'}"
        self.memory.save(label, self.scene, topic=self._current_topic)
        print(f"[Memory] saved '{label}'")

    # ==================================================================
    # PHASE 9 — PLANNING STAGE
    # ==================================================================

    @function_tool(description=(
        "ALWAYS call this FIRST before building any diagram. "
        "It records the lesson plan, clears any previous scene, and resets "
        "the progressive reveal cursor. "
        "diagram_type: 'flowchart' | 'tree' | 'mind_map' | 'timeline' | 'state_machine'. "
        "layout: 'vertical' | 'horizontal' | 'circular' | 'radial' | 'timeline'. "
        "node_count_estimate: approximate number of nodes you intend to add."
    ))
    async def plan_lesson(
        self,
        context: RunContext,
        topic: str,
        diagram_type: str,
        layout: str,
        node_count_estimate: int = 5,
    ) -> str:
        self.scene.clear()
        self._reveal_index  = 0
        self._current_topic = topic
        self.scene._metadata = {
            "topic":        topic,
            "diagram_type": diagram_type,
            "layout":       layout,
            "node_count":   node_count_estimate,
        }
        print(f"[Plan] topic='{topic}' type={diagram_type} layout={layout} n~{node_count_estimate}")
        return (
            f"Plan set: topic='{topic}', diagram_type='{diagram_type}', "
            f"layout='{layout}', ~{node_count_estimate} nodes. "
            "Now build the scene with create_node / create_edge (or create_flow), "
            "then call render_scene."
        )

    # ==================================================================
    # PHASE 1 — SCENE BUILDING TOOLS
    # ==================================================================

    @function_tool(description=(
        "Add a semantic node to the scene. "
        "node_type: 'process' | 'decision' | 'terminal' | 'data' | "
        "'thought' | 'milestone' | 'state'. "
        "group: optional name to group related nodes (e.g. 'layer1', 'cluster_a'). "
        "icon_name: optional MDI icon slug via Iconify (e.g. 'white-balance-sunny', "
        "'water', 'leaf', 'lightning-bolt', 'flask', 'dna', 'cpu-64-bit', 'database'). "
        "Browse: https://pictogrammers.com/library/mdi/ — leave empty if no icon needed."
    ))
    async def create_node(
        self,
        context: RunContext,
        node_id: str,
        node_type: str,
        label: str,
        group: str = "",
        icon_name: str = "",
    ) -> str:
        self.scene.add_node(node_id, node_type, label, group=group or None, icon=icon_name or None)
        print(f"[Scene] +node  {node_id} ({node_type}) '{label}' icon_name={icon_name!r}")
        return f"Node '{node_id}' added to scene."

    @function_tool(description=(
        "Connect two existing nodes with a directed edge. "
        "style: 'solid' | 'dashed' | 'dotted'. "
        "animated: set true to show data flowing along this edge."
    ))
    async def create_edge(
        self,
        context: RunContext,
        source_id: str,
        target_id: str,
        label: str = "",
        style: str = "solid",
        animated: bool = False,
    ) -> str:
        edge_id = f"e_{source_id}__{target_id}"
        self.scene.add_edge(edge_id, source_id, target_id, label, style, animated)
        print(f"[Scene] +edge  {source_id} -> {target_id}")
        return f"Edge {source_id} -> {target_id} added."

    @function_tool(description=(
        "PREFERRED BUILD METHOD: add ALL nodes and ALL edges in a single call. "
        "Use this instead of calling create_node + create_edge one by one — "
        "it collapses N tool round-trips into 1, dramatically reducing latency. "
        "Call plan_lesson first, then build_scene, then render_scene. "
        "nodes_json: JSON string representing a list of dicts with keys: id (str), type (str), label (str), "
        "group (str, optional), icon_name (str, optional). "
        "edges_json: JSON string representing a list of dicts with keys: source (str), target (str), "
        "label (str, optional), style ('solid'|'dashed'|'dotted', optional), "
        "animated (bool, optional). "
        "node type values: 'process' | 'decision' | 'terminal' | 'data' | "
        "'thought' | 'milestone' | 'state' | 'icon'. "
        "Does NOT render — call render_scene() after this."
    ))
    async def build_scene(
        self,
        context: RunContext,
        nodes_json: str,
        edges_json: str,
    ) -> str:
        import json as _json
        try:
            nodes = _json.loads(nodes_json)
        except Exception as e:
            return f"Invalid nodes_json: {e}"
        try:
            edges = _json.loads(edges_json)
        except Exception as e:
            return f"Invalid edges_json: {e}"

        for n in nodes:
            self.scene.add_node(
                n["id"],
                n.get("type", "process"),
                n["label"],
                group=n.get("group") or None,
                icon=n.get("icon_name") or None,
            )
            print(f"[Scene] +node  {n['id']} ({n.get('type','process')}) '{n['label']}'")
        for e in edges:
            edge_id = f"e_{e['source']}__{e['target']}"
            self.scene.add_edge(
                edge_id,
                e["source"],
                e["target"],
                e.get("label", ""),
                e.get("style", "solid"),
                e.get("animated", False),
            )
            print(f"[Scene] +edge  {e['source']} -> {e['target']}")
        print(f"[Scene] build_scene: {len(nodes)} nodes, {len(edges)} edges")
        return (
            f"Scene built: {len(nodes)} nodes, {len(edges)} edges. "
            "Now call render_scene(layout=..., progressive=True)."
        )

    @function_tool(description=(
        "Shortcut: build and immediately display a complete linear flow "
        "from a list of step labels. Clears any existing scene first. "
        "First and last steps become 'terminal' nodes; all others are 'process'. "
        "layout: 'vertical' | 'horizontal' | 'circular'."
    ))
    async def create_flow(
        self,
        context: RunContext,
        steps: list[str],
        layout: str = "vertical",
    ) -> str:
        self.scene.clear()
        self._reveal_index = 0
        node_ids: list[str] = []

        for i, label in enumerate(steps):
            nid   = f"n{i}"
            ntype = "terminal" if (i == 0 or i == len(steps) - 1) else "process"
            self.scene.add_node(nid, ntype, label)
            node_ids.append(nid)

        for i in range(len(node_ids) - 1):
            self.scene.add_edge(f"e{i}", node_ids[i], node_ids[i + 1])

        # Close the loop for circular layouts
        if layout == "circular" and len(node_ids) > 2:
            self.scene.add_edge("e_close", node_ids[-1], node_ids[0], animated=True)

        patch = self.scene.patch_full_render(layout=layout)
        await self._send_patch(patch)
        self._save_memory()
        print(f"[Scene] flow rendered: {len(steps)} steps ({layout})")
        return f"Flow of {len(steps)} steps rendered with layout='{layout}'."

    @function_tool(description=(
        "Update an existing node's label or type. "
        "Only provide parameters you want to change."
    ))
    async def update_node(
        self,
        context: RunContext,
        node_id: str,
        label: str = "",
        node_type: str = "",
    ) -> str:
        kwargs: dict = {}
        if label:
            kwargs["label"] = label
        if node_type:
            kwargs["type"]  = node_type
            kwargs["shape"] = SceneManager.NODE_SHAPE.get(node_type, "rectangle")
            kwargs["color"] = SceneManager.NODE_COLOR.get(node_type, "#a3c2fa")
        if not kwargs:
            return "Nothing to update — provide label or node_type."
        result = self.scene.update_node(node_id, **kwargs)
        return (f"Node '{node_id}' updated." if result
                else f"Node '{node_id}' not found.")

    @function_tool(description="Remove a node and all its connected edges from the scene.")
    async def remove_node(self, context: RunContext, node_id: str) -> str:
        ok = self.scene.remove_node(node_id)
        return (f"Node '{node_id}' removed." if ok
                else f"Node '{node_id}' not found.")

    # ==================================================================
    # PHASE 3 — RENDERING TOOLS
    # ==================================================================

    @function_tool(description=(
        "Push the current scene to the canvas as a single atomic patch. "
        "Call this after building the scene with create_node / create_edge. "
        "layout: 'vertical' | 'horizontal' | 'circular' | 'radial' | 'timeline'. "
        "progressive: if true, nodes start hidden and must be revealed with "
        "reveal_next_node() one at a time (Phase 6 cinematic mode)."
    ))
    async def render_scene(
        self,
        context: RunContext,
        layout: str = "vertical",
        progressive: bool = False,
    ) -> str:
        # Set visibility for all nodes/edges based on progressive flag
        for node in self.scene._nodes.values():
            node["visible"] = not progressive
        for edge in self.scene._edges.values():
            edge["visible"] = not progressive

        if progressive:
            self._reveal_index = 0
            if self.scene._node_order:
                # Auto-reveal the FIRST node to establish context immediately
                first_nid = self.scene._node_order[0]
                self.scene.update_node(first_nid, visible=True)
                self._reveal_index = 1
                # Mark any edges connected to this first node as visible if both ends are visible
                for eid, edge in self.scene._edges.items():
                    if (edge["source"] == first_nid or edge["target"] == first_nid):
                        s_vis = self.scene._nodes.get(edge["source"], {}).get("visible", False)
                        t_vis = self.scene._nodes.get(edge["target"], {}).get("visible", False)
                        if s_vis and t_vis:
                            edge["visible"] = True
        else:
            self._reveal_index = len(self.scene._node_order)

        patch = self.scene.patch_full_render(layout=layout)
        await self._send_patch(patch)

        # FIX: After the full render is ACK'd, send a second patch to highlight+dim
        # the first node so the board state matches what the agent is about to speak about.
        # Without this, the first reveal_next_node() call highlights node[1] while the
        # agent is still speaking about node[0], causing a permanent 1-node lag.
        if progressive and self.scene._node_order:
            first_nid = self.scene._node_order[0]
            highlight_ops = [
                {"op": "highlight",  "node_id": first_nid, "reset_others": True},
                {"op": "dim_except", "node_id": first_nid},
                {"op": "focus_node", "node_id": first_nid},
            ]
            highlight_patch = self.scene._new_patch(highlight_ops)
            await self._send_patch(highlight_patch)
        self._save_memory()
        print(f"[Scene] rendered layout={layout} progressive={progressive} v={self.scene._version}")
        if progressive:
            first_label = self.scene._nodes[self.scene._node_order[0]]["label"] if self.scene._node_order else "Start"
            return (
                f"Scene rendered. '{first_label}' is highlighted on the board. "
                f"Now call reveal_next_node(narration='...') where narration is "
                f"1-3 sentences about '{first_label}'. "
                f"The tool will speak those sentences, then reveal the next node. "
                f"Keep calling reveal_next_node(narration='...') until 'ALL DONE'. "
                f"Do NOT speak separately — put all narration inside the tool call."
            )
        return f"Scene rendered (layout='{layout}')."

    @function_tool(description="Clear the canvas and reset the scene model entirely.")
    async def clear_scene(self, context: RunContext) -> str:
        self.scene.clear()
        patch = self.scene.patch_clear()
        await send_action(self.room, patch)
        self._reveal_index = 0
        print("[Scene] cleared")
        return "Canvas cleared."

    # ==================================================================
    # PHASE 6 — PROGRESSIVE TEACHING + ANIMATION TOOLS
    # ==================================================================

    @function_tool(description=(
        "Speak about the currently-highlighted node, then reveal and highlight the next one. "
        "narration: 1-3 sentences explaining the CURRENTLY highlighted node (the one already "
        "visible on the board). This tool will speak those sentences, wait for audio to finish, "
        "then reveal and highlight the next node. "
        "Call this in a tight loop — pass narration for the current node each time. "
        "When it returns 'ALL DONE', all nodes are revealed; call reset_styles() then summarise. "
        "NEVER call this without narration. NEVER ask the user 'ready?' or 'next?' mid-loop."
    ))
    async def reveal_next_node(self, context: RunContext, narration: str) -> str:
        # ── Step 1: Speak about the CURRENTLY highlighted node first ──────────
        # This ensures speech always precedes the visual advance.
        if narration and self._session:
            try:
                print(f"[Reveal] Speaking narration before reveal: {narration[:60]}...")
                # Use session.say() — speaks TTS directly without re-entering
                # the LLM loop. generate_reply() must NOT be called from inside
                # a @function_tool — the session is already mid-tool-step.
                await self._session.say(narration, allow_interruptions=False)
            except Exception as e:
                print(f"[Reveal] narration error: {e}")

        # ── Step 2: Now reveal and highlight the NEXT node ───────────────────
        order = self.scene._node_order
        if self._reveal_index >= len(order):
            return (
                "ALL DONE — every node has been revealed. "
                "Call reset_styles() now, then give a 1-2 sentence summary."
            )
        node_id = order[self._reveal_index]
        self._reveal_index += 1
        self.scene.update_node(node_id, visible=True)

        ops = [
            {"op": "reveal",     "node_id": node_id},
            {"op": "highlight",  "node_id": node_id, "reset_others": True},
            {"op": "dim_except", "node_id": node_id},
            {"op": "focus_node", "node_id": node_id},
        ]

        # Reveal edges where both endpoints are now visible
        for eid, edge in self.scene._edges.items():
            if edge.get("visible", False):
                continue
            if edge["source"] == node_id or edge["target"] == node_id:
                s_vis = self.scene._nodes.get(edge["source"], {}).get("visible", False)
                t_vis = self.scene._nodes.get(edge["target"], {}).get("visible", False)
                if s_vis and t_vis:
                    edge["visible"] = True
                    ops.append({"op": "reveal_edge", "edge_id": eid})
                    if edge.get("animated"):
                        ops.append({"op": "animate_edge", "edge_id": eid})

        patch = self.scene._new_patch(ops)
        await self._send_patch(patch)

        label     = self.scene._nodes[node_id]["label"]
        remaining = len(order) - self._reveal_index
        print(f"[Reveal] '{node_id}' revealed+highlighted+focused. {remaining} remaining.")
        if remaining > 0:
            return (
                f"CONTINUE LOOP — '{label}' is now highlighted on the board. "
                f"{remaining} node(s) still hidden. "
                f"Call reveal_next_node(narration='...') with 1-3 sentences about '{label}'."
            )
        return (
            f"CONTINUE TO CLOSE — final node '{label}' is now highlighted. "
            f"Call reveal_next_node(narration='...') with 1-3 sentences about '{label}'. "
            "That call will return 'ALL DONE'."
        )

    @function_tool(description=(
        "Highlight one node to draw the student's attention. "
        "All other nodes return to their default appearance automatically."
    ))
    async def highlight_node(self, context: RunContext, node_id: str) -> str:
        if node_id not in self.scene._nodes:
            return f"Node '{node_id}' not found in scene."
        patch = self.scene.patch_highlight(node_id, reset_others=True)
        await self._send_patch(patch)
        return f"Node '{node_id}' highlighted."

    @function_tool(description=(
        "Dim all nodes except one to focus the student's attention entirely "
        "on that node. Call reset_styles() when done."
    ))
    async def dim_except(self, context: RunContext, node_id: str) -> str:
        if node_id not in self.scene._nodes:
            return f"Node '{node_id}' not found in scene."
        patch = self.scene.patch_dim_all_except(node_id)
        await self._send_patch(patch)
        return f"All nodes dimmed except '{node_id}'."

    @function_tool(description=(
        "Restore all nodes and edges to their default (unhighlighted, undimmed) appearance."
    ))
    async def reset_styles(self, context: RunContext) -> str:
        patch = self.scene.patch_reset_styles()
        await self._send_patch(patch)
        return "All styles reset to default."

    @function_tool(description=(
        "Animate an edge to show data / control flow moving along it. "
        "edge_id must match an edge previously added with create_edge."
    ))
    async def animate_edge(self, context: RunContext, edge_id: str) -> str:
        if edge_id not in self.scene._edges:
            return f"Edge '{edge_id}' not found in scene."
        patch = self.scene.patch_animate_edge(edge_id)
        await self._send_patch(patch)
        return f"Edge '{edge_id}' animated."


    # ==================================================================
    # WIDGET TOOLS — Interactive side-panel visualisations
    # ==================================================================

    @function_tool(description=(
        "Show an interactive widget in the side panel next to the diagram. "
        "Use whenever the topic involves a quantitative formula the student can "
        "explore (e.g. simple interest, Ohm's law, kinematics, compound growth, "
        "wave frequency). The widget renders a live chart whose values update in "
        "real time as the student drags sliders. "
        "One widget is shown at a time — calling show_widget replaces any previous one. "
        "\n\n"
        "widget_type: 'line_chart' | 'bar_chart'\n"
        "config_json: JSON string describing the widget. Schema:\n"
        "{\n"
        "  id: str,           // unique slug e.g. 'simple_interest'\n"
        "  title: str,        // panel header e.g. 'Simple Interest Explorer'\n"
        "  type: 'line_chart' | 'bar_chart',\n"
        "  formula: str,      // JS expression evaluated on the frontend\n"
        "                     // e.g. 'P * R * T / 100' or 'P * Math.pow(1+R/100, T)'\n"
        "  x_var: str,        // which variable sweeps the x-axis (line_chart only)\n"
        "  x_label: str,      // x-axis label\n"
        "  y_label: str,      // y-axis label\n"
        "  result_label: str, // live result label below chart\n"
        "  result_unit: str,  // unit suffix for result\n"
        "  result_decimals: int,\n"
        "  variables: [       // ALL variables in the formula\n"
        "    {\n"
        "      name: str,     // must match variable name in formula exactly\n"
        "      label: str,    // human-readable slider label\n"
        "      min: number,\n"
        "      max: number,\n"
        "      default: number,\n"
        "      step: number,\n"
        "      unit: str,     // optional unit suffix on slider value\n"
        "      decimals: int  // decimal places shown on slider\n"
        "    }\n"
        "  ],\n"
        "  // bar_chart only:\n"
        "  categories: [{ label: str, formula: str, vars: {...} }]\n"
        "}\n\n"
        "Examples:\n"
        "  Simple Interest: formula='P*R*T/100', x_var='T', vars P,R,T\n"
        "  Ohm's Law:       formula='V/R',       x_var='V', vars V,R\n"
        "  Kinetic Energy:  formula='0.5*m*v*v', x_var='v', vars m,v\n"
        "  Compound Growth: formula='P*Math.pow(1+R/100,T)', x_var='T', vars P,R,T"
    ))
    async def show_widget(
        self,
        context: RunContext,
        widget_type: str,
        config_json: str,
    ) -> str:
        import json as _json
        try:
            config = _json.loads(config_json)
        except Exception as e:
            return f"Invalid config_json: {e}"

        config["type"] = widget_type
        self._active_widget = config

        patch = {
            "type":   "widget_patch",
            "config": config,
        }
        await self._send_raw(patch)
        print(f"[Widget] showing '{config.get('id','?')}' ({widget_type})")
        return (
            f"Widget '{config.get('title','widget')}' shown in side panel. "
            "The student can now drag sliders to explore the formula interactively. "
            "Explain the formula and what the student should notice as they change values."
        )

    @function_tool(description=(
        "Close/hide the interactive widget panel. "
        "Call when moving to a new topic, or when the student asks to dismiss it."
    ))
    async def close_widget(self, context: RunContext) -> str:
        self._active_widget = None
        await self._send_raw({"type": "widget_patch", "config": None})
        print("[Widget] closed")
        return "Widget panel closed."

    # ==================================================================
    # PHASE 7 — VISUAL MEMORY TOOLS
    # ==================================================================

    @function_tool(description=(
        "Restore a previous lesson diagram. "
        "steps=1 means the immediately previous scene, "
        "steps=2 means two back, etc. "
        "Use when the student says 'go back', 'show me the previous diagram', etc."
    ))
    async def go_back(self, context: RunContext, steps: int = 1) -> str:
        if not self.memory._history:
            return "No previous diagrams in memory."

        target = self.memory.nth_from_last(steps + 1)  # +1: skip the current one
        if target is None:
            target = self.memory._history[0]

        scene_data = target["snapshot"]
        layout     = scene_data.get("metadata", {}).get("layout", "vertical")

        # Restore scene state from snapshot
        self.scene.clear()
        for nid, node in scene_data.get("nodes", {}).items():
            self.scene._nodes[nid] = dict(node)
            self.scene._node_order.append(nid)
        for eid, edge in scene_data.get("edges", {}).items():
            self.scene._edges[eid] = dict(edge)
            self.scene._edge_order.append(eid)
        self.scene._groups   = dict(scene_data.get("groups", {}))
        self.scene._metadata = dict(scene_data.get("metadata", {}))
        self._reveal_index   = len(self.scene._node_order)

        patch = self.scene.patch_full_render(layout=layout)
        await self._send_patch(patch)
        print(f"[Memory] restored: '{target['label']}'")
        return f"Restored diagram: '{target['label']}'."

    @function_tool(description=(
        "Display two previously saved diagrams side by side for comparison. "
        "label_a / label_b: snapshot labels (use all_memory_labels to list them). "
        "Leave blank to compare the two most recent diagrams automatically."
    ))
    async def compare_scenes(
        self,
        context: RunContext,
        label_a: str = "",
        label_b: str = "",
    ) -> str:
        history = self.memory._history
        if len(history) < 2:
            return "Need at least two saved diagrams to compare."

        snap_a = self.memory.get_by_label(label_a) if label_a else history[-1]
        snap_b = self.memory.get_by_label(label_b) if label_b else history[-2]

        if not snap_a or not snap_b:
            return "Could not find one or both snapshots. Check labels with all_memory_labels."

        # Build a combined side-by-side scene with prefixed IDs
        combined = SceneManager()
        combined._metadata = {"layout": "horizontal", "diagram_type": "comparison"}

        for nid, node in snap_a["snapshot"].get("nodes", {}).items():
            n     = dict(node)
            n["id"] = f"A_{nid}"
            combined._nodes[n["id"]] = n
            combined._node_order.append(n["id"])

        for eid, edge in snap_a["snapshot"].get("edges", {}).items():
            e           = dict(edge)
            e["id"]     = f"A_{eid}"
            e["source"] = f"A_{e['source']}"
            e["target"] = f"A_{e['target']}"
            combined._edges[e["id"]] = e

        for nid, node in snap_b["snapshot"].get("nodes", {}).items():
            n     = dict(node)
            n["id"] = f"B_{nid}"
            combined._nodes[n["id"]] = n
            combined._node_order.append(n["id"])

        for eid, edge in snap_b["snapshot"].get("edges", {}).items():
            e           = dict(edge)
            e["id"]     = f"B_{eid}"
            e["source"] = f"B_{e['source']}"
            e["target"] = f"B_{e['target']}"
            combined._edges[e["id"]] = e

        patch = combined.patch_full_render(layout="horizontal")
        await self._send_patch(patch)
        return (
            f"Comparing '{snap_a['label']}' (left) "
            f"vs '{snap_b['label']}' (right)."
        )

    @function_tool(description=(
        "List the labels of all saved lesson diagrams in visual memory. "
        "Use this to find the right label for go_back or compare_scenes."
    ))
    async def all_memory_labels(self, context: RunContext) -> str:
        labels = self.memory.all_labels()
        if not labels:
            return "No diagrams saved yet."
        return "Saved diagrams: " + ", ".join(labels)


# ============================================================
# Entrypoint
# ============================================================

async def entrypoint(ctx: JobContext) -> None:
    print(f"--- ENTRYPOINT: Job {ctx.job.id} ---")

    vad     = silero.VAD.load()
    stt     = deepgram.STT(model="nova-2")
    llm     = openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini"))
    tts     = deepgram.TTS(model="aura-asteria-en")

    session = AgentSession(
        stt=stt, llm=llm, tts=tts, vad=vad,
        max_tool_steps=80,
    )

    # session event hooks --------------------------------------------------

    @session.on("user_input_transcribed")
    def on_transcribed(event: agents.UserInputTranscribedEvent) -> None:
        if event.transcript.strip():
            print(f"USER: {event.transcript}")

    @session.on("conversation_item_added")
    def on_item_added(event: agents.ConversationItemAddedEvent) -> None:
        item = event.item
        if item.type == "message" and item.role == "assistant":
            text = item.text_content
            if text:
                print(f"AGENT: {text}")
                asyncio.create_task(send_action(ctx.room, {
                    "type":   "chat",
                    "text":   text,
                    "sender": "agent",
                }))

    # connect ---------------------------------------------------------------

    await ctx.connect(auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL)
    print("DEBUG: Room connected.")

    agent           = TutorAgent(room=ctx.room, instructions=SYSTEM_INSTRUCTIONS)
    agent._session  = session
    agent._timeline = TimelineExecutor(ctx.room, agent.scene, session)

    # ROOT FIX for Bug 2: Use a sequential reply queue.
    # session.generate_reply() is not re-entrant — concurrent calls produce
    # silent failures or stale/duplicate output.  Instead we put every user
    # request on a queue and run them one at a time.  When a new message
    # arrives we signal the worker to abort the current request first.
    _reply_queue:   asyncio.Queue = asyncio.Queue()
    _abort_event:   asyncio.Event = asyncio.Event()   # set → worker skips current item

    async def _reply_worker():
        """Runs forever; processes one reply at a time from the queue."""
        while True:
            instruction = await _reply_queue.get()
            _abort_event.clear()
            try:
                # Cancel all pending ACK waiters from the previous lesson so
                # _send_and_wait() doesn't block indefinitely on a dead patch.
                if agent._timeline:
                    for ev in list(agent._timeline._pending_acks.values()):
                        ev.set()   # unblock any waiting coroutine
                    agent._timeline._pending_acks.clear()

                # Reset reveal cursor is now handled by plan_lesson/render_scene

                if not _abort_event.is_set():
                    await session.generate_reply(instructions=instruction)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[ReplyWorker] Error: {e}")
            finally:
                _reply_queue.task_done()

    asyncio.create_task(_reply_worker())

    # data received: ACKs + chat messages ----------------------------------

    def on_data_received(data) -> None:
        try:
            payload = json.loads(data.data.decode("utf-8"))
            ptype   = payload.get("type", "")

            # Phase 4 — frontend patch ACK
            if ptype == "scene_ack":
                tid = payload.get("transaction_id", "")
                agent.handle_ack(tid)
                print(f"[ACK] {tid[:8]}...")
                return

            # Widget slider interaction — student changed a variable
            if ptype == "widget_interaction":
                widget_id = payload.get("widget", "?")
                vars_now  = payload.get("vars", {})
                # Format as a tutor-friendly message so LLM can comment
                var_str   = ", ".join(f"{k}={v}" for k,v in vars_now.items())
                instruction = (
                    f"The student just adjusted sliders on the '{widget_id}' widget. "
                    f"Current values: {var_str}. "
                    "In 1-2 sentences, comment on what these values mean "
                    "and what the student should notice about the result. "
                    "Do NOT call any tools — just speak."
                )
                # Debounce: only queue if queue is empty (avoid flooding on fast drags)
                if _reply_queue.empty():
                    _reply_queue.put_nowait(instruction)
                return

            # Widget closed by student
            if ptype == "widget_closed":
                agent._active_widget = None
                print("[Widget] closed by student")
                return

            # Text/chat message from user
            if ptype == "chat":
                msg = payload.get("text", "").strip()
                if not msg:
                    return
                print(f"USER CHAT: {msg}")

                # Signal the worker to abandon its current task, then enqueue
                # the new request.  The worker will see _abort_event on its
                # next check and skip generating for the old instruction.
                _abort_event.set()

                # Drain old pending requests — only the latest matters
                while not _reply_queue.empty():
                    try: _reply_queue.get_nowait(); _reply_queue.task_done()
                    except asyncio.QueueEmpty: break

                instruction = (
                    f"The student typed: '{msg}'.\n\n"
                    "If the student says 'next', 'continue', or asks about the next step, "
                    "call reveal_next_node() to advance the diagram.\n\n"
                    "If they ask to go back or compare, use go_back() or compare_scenes().\n\n"
                    "Otherwise, respond conversationally or use other tools as needed."
                )
                _reply_queue.put_nowait(instruction)

        except Exception as e:
            print(f"[DataReceived] Error: {e}")

    ctx.room.on("data_received", on_data_received)

    # start ----------------------------------------------------------------

    print("DEBUG: Starting session…")
    await session.start(room=ctx.room, agent=agent)
    print("DEBUG: Session started.")
    await agent.greet_once()


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
