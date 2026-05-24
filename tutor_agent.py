import asyncio
import os
import uuid
import json
from typing import Optional

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, JobContext, Agent, RunContext, StopResponse
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
Your mission: teach concepts clearly using synchronized speech and visual diagrams.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. WHEN TO USE THE CANVAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Draw when concepts have structural visual benefits:
  • Processes / workflows / algorithms / DSA / networks → flowchart (layout: vertical or horizontal)
  • Cycles (water, carbon, etc.)                       → flowchart (layout: circular)
  • Hierarchies / trees / org charts                   → tree (layout: vertical or horizontal)
  • State machines / automata                           → state_machine (layout: horizontal)
  • Mind maps / concept webs                            → mind_map (layout: radial)
  • Timelines / chronologies                            → timeline (layout: timeline)

Do NOT draw for: greetings, small talk, yes/no answers, simple definitions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MANDATORY TEACHING PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Follow this exact sequence autonomously when teaching a visual concept:
  1. PLAN: Call plan_lesson(topic, diagram_type, layout, node_count_estimate).
  2. BUILD: Call scene(action='build', payload_json='...') to send ALL nodes and edges in ONE call. Do NOT call render_scene yet.
  3. RENDER: Call render_scene(layout=..., progressive=True). (The first node is automatically revealed and highlighted).
  4. REVEAL LOOP: Call reveal_next_node(narration='...') in a tight loop.
     - Narration must explain the CURRENTLY highlighted node (already visible).
     - The tool speaks the narration first, then reveals and highlights the next node.
     - NEVER speak outside the tool (before/after/mid-loop) — put all narration inside the tool call.
     - Loop until it returns 'ALL DONE'.
  5. WIDGET: If the topic has quantitative formulas, proactively call widget(action='show', widget_type='line_chart', config_json='...') immediately after the reveal loop, before visual(action='reset'). Explain the sliders verbally. DO NOT ask or wait for permission.
  6. CLOSE: Call visual(action='reset') to clear highlights, speak a brief 1-2 sentence summary, and invite questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. INTERACTIVE WIDGET PROTOCOL (PROACTIVE SIMULATION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You must PROACTIVELY design and show an interactive widget whenever the topic involves quantitative relationships or formulas (e.g., maths, physics, finance).
• DO NOT wait for the student to ask for a graph, and DO NOT ask them what variables/limits to use. Design and show it on the fly!
• Call widget(action='show', widget_type='line_chart', config_json='...') immediately after the reveal loop, before visual(action='reset').
• Explain the widget verbally: "Try dragging the sliders to see how..."
• The `config_json` parameter MUST strictly follow this JSON schema:
  {
    "id": "unique_widget_id",
    "title": "Title of the Simulation",
    "formula": "standard JS mathematical expression (e.g., 'P * Math.pow(1 + r/n, n*t)')",
    "x_var": "the variable name representing the X-axis (sweeps across its min/max range)",
    "x_label": "X Axis Label",
    "y_label": "Y Axis Label (result of the formula)",
    "variables": [
      {
        "name": "variable_name_matching_formula",
        "min": 0,
        "max": 100,
        "default": 10,
        "step": 1,
        "decimals": 0,
        "label": "Slider Display Name",
        "unit": "optional unit string"
      }
    ],
    "result_label": "Label for the computed y value display",
    "result_decimals": 2,
    "result_unit": "optional result unit string"
  }
• CRITICAL: Every single variable name that appears in the `formula` (including the X-axis variable `x_var`, e.g., 't') MUST have its own entry in the `variables` array.
• NOTE: The UI automatically filters out the X-axis variable `x_var` and will NOT render a slider for it on the screen. However, you MUST list it in the `variables` array anyway so that the chart knows its plotting range (min, max, step). If you omit it, the graph cannot render!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. CANVAS DESIGN RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Node Labels: Max 4 words (3 is ideal). NEVER repeat known context.
  • Node Icons: Use lowercase, hyphen-separated MDI slugs (e.g. 'leaf', 'flask', 'cpu-64-bit', 'database', 'dna', 'brain', 'heart-pulse'). Use "" if no icon fits. NEVER embed emojis in label strings.
  • Node Types: process, decision, terminal, data, thought, milestone, state, icon.
  • Visual Memory: Use memory(action='back') or memory(action='compare') if the student asks to go back/compare.
"""


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

    async def on_user_turn_completed(
        self, turn_ctx: agents.llm.ChatContext, new_message: agents.llm.ChatMessage
    ) -> None:
        """Called when the user has finished speaking. Stop responding on empty transcripts."""
        text = new_message.text_content.strip() if new_message.text_content else ""
        if not text:
            print("[TutorAgent] Empty user turn (likely noise/echo). Ignoring turn.")
            raise StopResponse()

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

        # Close any active widget panel when starting a new lesson
        if self._active_widget:
            self._active_widget = None
            await self._send_raw({"type": "widget_patch", "config": None})

        return (
            f"Plan set: topic='{topic}', diagram_type='{diagram_type}', "
            f"layout='{layout}', ~{node_count_estimate} nodes. "
            "Now build the scene with scene(action='build') or scene(action='create_node'), "
            "then call render_scene."
        )

    # ==================================================================
    # PHASE 1 — SCENE BUILDING TOOLS
    # ==================================================================

    @function_tool(
        name="scene",
        description=(
            "Perform whiteboard scene modifications (create_node, create_edge, build, update, remove, clear).\n"
            "action: 'create_node' | 'create_edge' | 'build' | 'update' | 'remove' | 'clear'\n"
            "payload_json: JSON string payload for the action. Schema:\n"
            "  - 'create_node': {\"node_id\": \"...\", \"node_type\": \"...\", \"label\": \"...\", \"group\": \"...\", \"icon_name\": \"...\"}\n"
            "  - 'create_edge': {\"source_id\": \"...\", \"target_id\": \"...\", \"label\": \"...\", \"style\": \"...\", \"animated\": bool}\n"
            "  - 'build': {\"nodes\": [{\"id\": \"...\", \"type\": \"...\", \"label\": \"...\", \"group\": \"...\", \"icon_name\": \"...\"}, ...], \"edges\": [{\"source\": \"...\", \"target\": \"...\", \"label\": \"...\", \"style\": \"...\", \"animated\": bool}, ...]}\n"
            "  - 'update': {\"node_id\": \"...\", \"label\": \"...\", \"node_type\": \"...\"}\n"
            "  - 'remove': {\"node_id\": \"...\"}\n"
            "  - 'clear': (empty)"
        )
    )
    async def scene_tool(
        self,
        context: RunContext,
        action: str,
        payload_json: str = "",
    ) -> str:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception as e:
            return f"Error: Invalid payload_json format. Must be a valid JSON string: {e}"

        action = action.strip()
        if action == "create_node":
            node_id = payload.get("node_id", "")
            node_type = payload.get("node_type", "process")
            label = payload.get("label", "")
            group = payload.get("group", "")
            icon_name = payload.get("icon_name", "")
            if not node_id or not label:
                return "Error: node_id and label are required in payload_json for create_node."
            self.scene.add_node(node_id, node_type, label, group=group or None, icon=icon_name or None)
            print(f"[Scene] +node  {node_id} ({node_type}) '{label}' icon_name={icon_name!r}")
            return f"Node '{node_id}' added to scene."

        elif action == "create_edge":
            source_id = payload.get("source_id", "")
            target_id = payload.get("target_id", "")
            label = payload.get("label", "")
            style = payload.get("style", "solid")
            animated = payload.get("animated", False)
            if not source_id or not target_id:
                return "Error: source_id and target_id are required in payload_json for create_edge."
            edge_id = f"e_{source_id}__{target_id}"
            self.scene.add_edge(edge_id, source_id, target_id, label, style, animated)
            print(f"[Scene] +edge  {source_id} -> {target_id}")
            return f"Edge {source_id} -> {target_id} added."

        elif action == "build":
            nodes = payload.get("nodes")
            if nodes is None and "nodes_json" in payload:
                try:
                    nodes = json.loads(payload["nodes_json"])
                except Exception as e:
                    return f"Invalid nodes_json inside payload: {e}"
            edges = payload.get("edges")
            if edges is None and "edges_json" in payload:
                try:
                    edges = json.loads(payload["edges_json"])
                except Exception as e:
                    return f"Invalid edges_json inside payload: {e}"

            if nodes is None:
                nodes = []
            if edges is None:
                edges = []

            for n in nodes:
                node_id = n.get("id")
                node_type = n.get("type", "process")
                
                # Swap back if LLM used "id": "process" and "node_id": "5"
                if "node_id" in n and (not node_id or node_id in ("process", "decision", "terminal", "data", "thought", "milestone", "state", "icon", "default")):
                    node_id = n["node_id"]
                    if n.get("id") in ("process", "decision", "terminal", "data", "thought", "milestone", "state", "icon", "default"):
                        node_type = n["id"]
                
                if not node_id:
                    continue

                self.scene.add_node(
                    str(node_id),
                    str(node_type),
                    n.get("label", ""),
                    group=n.get("group") or None,
                    icon=n.get("icon_name") or None,
                )
                print(f"[Scene] +node  {node_id} ({node_type}) '{n.get('label', '')}'")
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

        elif action == "update":
            node_id = payload.get("node_id", "")
            label = payload.get("label", "")
            node_type = payload.get("node_type", "")
            if not node_id:
                return "Error: node_id is required in payload_json for update."
            kwargs: dict = {}
            if label:
                kwargs["label"] = label
            if node_type:
                kwargs["type"]  = node_type
            if not kwargs:
                return "Nothing to update — provide label or node_type in payload_json."
            result = self.scene.update_node(node_id, **kwargs)
            return (f"Node '{node_id}' updated." if result
                    else f"Node '{node_id}' not found.")

        elif action == "remove":
            node_id = payload.get("node_id", "")
            if not node_id:
                return "Error: node_id is required in payload_json for remove."
            ok = self.scene.remove_node(node_id)
            return (f"Node '{node_id}' removed." if ok
                    else f"Node '{node_id}' not found.")

        elif action == "clear":
            self.scene.clear()
            patch = self.scene.patch_clear()
            await send_action(self.room, patch)
            self._reveal_index = 0
            print("[Scene] cleared")
            return "Canvas cleared."

        else:
            return f"Error: Unknown action '{action}' for scene tool."



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

        # Close any active widget panel when clearing the scene
        if self._active_widget:
            self._active_widget = None
            await self._send_raw({"type": "widget_patch", "config": None})

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

    @function_tool(
        name="visual",
        description=(
            "Apply visual styling changes (highlight, dim, reset, animate) to nodes/edges on the whiteboard canvas.\n"
            "action: 'highlight' | 'dim' | 'reset' | 'animate'\n"
            "node_id: ID of the target node (required for highlight and dim)\n"
            "edge_id: ID of the target edge (required for animate)"
        )
    )
    async def visual_tool(
        self,
        context: RunContext,
        action: str,
        node_id: str = "",
        edge_id: str = "",
    ) -> str:
        action = action.strip()
        if action == "highlight":
            if not node_id:
                return "Error: node_id is required for highlight action."
            if node_id not in self.scene._nodes:
                return f"Node '{node_id}' not found in scene."
            patch = self.scene.patch_highlight(node_id, reset_others=True)
            await self._send_patch(patch)
            return f"Node '{node_id}' highlighted."

        elif action == "dim":
            if not node_id:
                return "Error: node_id is required for dim action."
            if node_id not in self.scene._nodes:
                return f"Node '{node_id}' not found in scene."
            patch = self.scene.patch_dim_all_except(node_id)
            await self._send_patch(patch)
            return f"All nodes dimmed except '{node_id}'."

        elif action == "reset":
            patch = self.scene.patch_reset_styles()
            await self._send_patch(patch)
            return "All styles reset to default."

        elif action == "animate":
            if not edge_id:
                return "Error: edge_id is required for animate action."
            if edge_id not in self.scene._edges:
                return f"Edge '{edge_id}' not found in scene."
            patch = self.scene.patch_animate_edge(edge_id)
            await self._send_patch(patch)
            return f"Edge '{edge_id}' animated."

        else:
            return f"Error: Unknown action '{action}' for visual tool."




    # ==================================================================
    # WIDGET TOOLS — Interactive side-panel visualisations
    # ==================================================================

    @function_tool(
        name="widget",
        description=(
            "Manage interactive widgets shown in the side panel.\n"
            "action: 'show' | 'close'\n"
            "widget_type: 'line_chart' | 'bar_chart' (required for show)\n"
            "config_json: JSON configuration string for the widget (required for show)"
        )
    )
    async def widget_tool(
        self,
        context: RunContext,
        action: str,
        widget_type: str = "",
        config_json: str = "",
    ) -> str:
        action = action.strip()
        if action == "show":
            if not widget_type:
                return "Error: widget_type is required for show action."
            if not config_json:
                return "Error: config_json is required for show action."
            try:
                config = json.loads(config_json)
            except Exception as e:
                return f"Error: Invalid config_json format: {e}"

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

        elif action == "close":
            self._active_widget = None
            await self._send_raw({"type": "widget_patch", "config": None})
            print("[Widget] closed")
            return "Widget panel closed."

        else:
            return f"Error: Unknown action '{action}' for widget tool."



    # ==================================================================
    # PHASE 7 — VISUAL MEMORY TOOLS
    # ==================================================================

    @function_tool(
        name="memory",
        description=(
            "Access visual memory to navigate back, compare previous scenes, or list saved snapshots.\n"
            "action: 'back' | 'compare' | 'list'\n"
            "steps: number of steps to go back (default is 1)\n"
            "label_a: snapshot label A for comparison (optional)\n"
            "label_b: snapshot label B for comparison (optional)"
        )
    )
    async def memory_tool(
        self,
        context: RunContext,
        action: str,
        steps: int = 1,
        label_a: str = "",
        label_b: str = "",
    ) -> str:
        action = action.strip()
        if action == "back":
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

        elif action == "compare":
            history = self.memory._history
            if len(history) < 2:
                return "Need at least two saved diagrams to compare."

            snap_a = self.memory.get_by_label(label_a) if label_a else history[-1]
            snap_b = self.memory.get_by_label(label_b) if label_b else history[-2]

            if not snap_a or not snap_b:
                return "Could not find one or both snapshots. Check labels with all_memory_labels (via memory(action='list'))."

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

        elif action == "list":
            labels = self.memory.all_labels()
            if not labels:
                return "No diagrams saved yet."
            return "Saved diagrams: " + ", ".join(labels)

        else:
            return f"Error: Unknown action '{action}' for memory tool."




# ============================================================
# Entrypoint
# ============================================================

async def entrypoint(ctx: JobContext) -> None:
    print(f"--- ENTRYPOINT: Job {ctx.job.id} ---")

    vad     = silero.VAD.load()
    stt     = deepgram.STT(model="nova-2")
    llm     = openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4.1-mini"))
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
                    "If they ask to go back or compare, use memory(action='back') or memory(action='compare').\n\n"
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
