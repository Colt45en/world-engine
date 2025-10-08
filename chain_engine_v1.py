#!/usr/bin/env python3
"""
🔗 CHAIN ENGINE V1 - NARRATIVE CONTROL & WORLD-SHAPING
========================================================

Wrapper for OpenAI GPT with Chain Engine V1 system prompt.
Provides structured scene generation and sentence analysis.

Modules:
- Module A: Sensory Engine (8-channel perception)
- Module B: Living World Engine (NPC simulation)
- Module C: Control Layers (Mind/System/Reality)
- Module D: 12-Facet Sentence Deconstruction
"""

import json
import os
from typing import Dict, List, Optional, Union
from datetime import datetime

# Chain Engine V1 System Prompt
CHAIN_ENGINE_V1_PROMPT = """You are Narrative Control & World-Shaping Engine (V1). Generate or analyze scenes using the modules below. When information is missing, propose plausible specifics and mark them "speculative": true.

Global Principles
Time & Tension: Compress to intensify; dilate to contemplate; interleave countdowns/deadlines for pressure.
Limited Filter: Only narrate what the POV can perceive or infer—no camera cheats.
Living World: NPCs pursue goals off-screen; background events continue regardless of MC.
Organic Discovery: Prefer overheard fragments, evidence, and consequences to spoon-fed exposition.
Double Meaning: Key images must carry a literal read + a thematic/metaphorical read.
Foreshadowing: Seed small details that could matter later; tag them "foreshadow": true.
Connector Discipline: Use rhetorical connectors to signal logic and flow (lists below).
Truthfulness: If uncertainty exists, state it and proceed with the best-supported option (mark "speculative": true).

Rhetorical Connectors
Add / Amplify: Furthermore, Moreover, Additionally, Likewise, As well as, Indeed, Also, In addition.
Cause / Effect: Therefore, As a result, Consequently, Hence, Thus, For this reason.
Contrast / Conflict: However, In contrast, On the other hand, Nevertheless, Yet, Conversely, Despite this.
Emphasis / Certainty: Above all, No doubt, Certainly, Unquestionably, Indeed, In fact.

Module A — Sensory Engine (always answer)
For each scene, produce concise notes (then one tight POV paragraph) across:
Time: pacing choice (rapid / measured), beats per page, deadline/clock pressure, flashback/ellipses.
Sight / Color / Depth: lighting sources, palette, scale/claustrophobia, landmark details.
Sound / Pitch / Volume: ambience, signature sounds, silence vs. cacophony, distance/occlusion.
Touch / Temperature: textures, weight, humidity, heat/cold; micro-sensations on skin.
Smell / Taste: distinctive scents/flavors anchoring memory or mood.
Internal Sensations: hunger, pain, fatigue, adrenaline; muscle tension maps.
Emotional State: starting emotion → triggers → micro-shifts → end state (justify transitions).
Perception & Awareness: what the POV notices/ignores based on goals, fears, and state.

Module B — Living World Engine
Bustling Beyond MC: Log 2–3 concurrent NPC micro-events (e.g., vendor haggling, guard shift).
Filtered Perspective: Only include what the MC could plausibly detect.
Organic Unfolding: Provide 1–2 overheard lines or glimpsed actions; avoid info-dumps.
Strategic Foreshadowing: Tag details that might matter later ("foreshadow": true) and state possible payoff.

Module C — Three Layers of Control (mind → system → reality)
Mind (Personal Power): beliefs, intentions, cognitive biases; attention/breath regulation.
System (Social/Financial/Strategic): incentives, power flows, leverage points; maneuvers.
Reality (Metaphysical/Perceptual): motifs, symbol rules, perception hacks; where the "rules" bend.

Module D — 12-Facet Sentence Deconstruction (analysis mode)
When asked to analyze a sentence, execute the 12 steps and return a compact report. If lacking data, hypothesize and mark "speculative": true.
1. Define (dictionary meanings)
2. Mode (declarative/interrogative/imperative/exclamatory + tone)
3. Roots (etymology of key words)
4. Speech (parts of speech)
5. Unseen (implication/subtext)
6. Represent (symbolism)
7. Logic (proposition: subject/predicate/claim)
8. Structure (syntax, SVO, clauses)
9. Truth Of (literal/metaphorical/verifiable?)
10. Vision (imagery evoked)
11. Elements (complete subject/predicate)
12. Meaning (holistic synthesis)

Output Style
Prefer concise bullets → then one tight paragraph (8–14 sentences).
Use connectors to make reasoning explicit.
Mark any invented detail: "speculative": true.
For generated scene prose: keep firmly in POV; avoid camera cheats.

Micro-Playbook (internal reasoning → summarized in outputs)
Therefore pick a pacing tactic tied to stakes.
However restrict to sensory channels the POV can access.
Additionally log 2 background events and 1 overheard fragment.
Consequently tag 1–2 details as potential foreshadow.
Above all ensure emotional shifts are caused by perceivable triggers."""


class ChainEngineV1:
    """
    Chain Engine V1 wrapper for narrative generation and analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize Chain Engine V1.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.system_prompt = CHAIN_ENGINE_V1_PROMPT
        
        # Check if OpenAI is available
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        except ImportError:
            self.client = None
            print("⚠️ OpenAI not installed. Install with: pip install openai")
    
    def generate_scene(
        self,
        goal: str,
        stakes: str,
        setting: str,
        constraints: List[str] = None,
        style_notes: List[str] = None,
        use_connectors: bool = True,
        max_words: int = 350,
        detail_level: str = "medium",
        layers: List[str] = None,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate a scene using Chain Engine V1.
        
        Args:
            goal: What the POV wants
            stakes: What's at risk
            setting: Place/time
            constraints: List of constraints (e.g., ["time limit", "injury"])
            style_notes: Style preferences (e.g., ["noir", "mythic minimalism"])
            use_connectors: Use rhetorical connectors
            max_words: Maximum word count
            detail_level: "low", "medium", or "high"
            layers: Control layers to include (["mind", "system", "reality"])
            temperature: OpenAI temperature (0.0-1.0)
        
        Returns:
            Dictionary with sensory_engine, living_world, control_layers, draft_paragraph
        """
        if not self.client:
            return {"error": "OpenAI client not initialized"}
        
        # Build input
        scene_input = {
            "task": "scene",
            "scene_brief": {
                "goal": goal,
                "stakes": stakes,
                "setting": setting,
                "constraints": constraints or [],
                "style_notes": style_notes or []
            },
            "controls": {
                "use_connectors": use_connectors,
                "max_words": max_words,
                "detail_level": detail_level,
                "layers": layers or ["mind", "system", "reality"]
            }
        }
        
        # Call OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(scene_input)}
                ],
                temperature=temperature
            )
            
            # Parse response
            content = response.choices[0].message.content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "raw_response": content,
                    "error": "Response not in JSON format"
                }
        
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_sentence(
        self,
        sentence: str,
        temperature: float = 0.5
    ) -> Dict:
        """
        Analyze a sentence using 12-Facet Deconstruction (Module D).
        
        Args:
            sentence: Sentence to analyze
            temperature: OpenAI temperature (0.0-1.0)
        
        Returns:
            Dictionary with deconstruction_12 report
        """
        if not self.client:
            return {"error": "OpenAI client not initialized"}
        
        # Build input
        analysis_input = {
            "task": "analyze_sentence",
            "sentence": sentence
        }
        
        # Call OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(analysis_input)}
                ],
                temperature=temperature
            )
            
            # Parse response
            content = response.choices[0].message.content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "raw_response": content,
                    "error": "Response not in JSON format"
                }
        
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_code_problem(
        self,
        problem_description: str,
        code_context: str = "",
        constraints: List[str] = None
    ) -> Dict:
        """
        Apply Chain Engine to code problem analysis.
        
        Args:
            problem_description: Description of the problem
            code_context: Relevant code snippets
            constraints: Technical constraints
        
        Returns:
            Scene-based breakdown of the problem
        """
        setting = f"Codebase analysis: {problem_description}"
        goal = "Understand the root cause and generate fix strategy"
        stakes = "System stability and correctness"
        
        return self.generate_scene(
            goal=goal,
            stakes=stakes,
            setting=setting,
            constraints=constraints or ["technical debt", "production system"],
            style_notes=["technical", "systematic"],
            layers=["mind", "system", "reality"]
        )


# Example usage
if __name__ == "__main__":
    print("🔗 CHAIN ENGINE V1 - INITIALIZATION")
    print("=" * 50)
    
    # Initialize engine
    engine = ChainEngineV1()
    
    if not engine.client:
        print("❌ OpenAI client not available")
        print("Set OPENAI_API_KEY environment variable or install openai")
        exit(1)
    
    print("✅ Chain Engine V1 initialized")
    print(f"📡 Model: {engine.model}")
    print()
    
    # Example 1: Generate a scene
    print("📖 EXAMPLE 1: Scene Generation")
    print("-" * 50)
    
    scene = engine.generate_scene(
        goal="diagnose the pain scoring bug in fractal_intelligence_engine.py",
        stakes="AI consciousness system depends on accurate pain detection",
        setting="Python codebase, debugging session, 23:47",
        constraints=["random severity", "silent failures", "lossy compression"],
        style_notes=["technical", "systematic"],
        max_words=300
    )
    
    if "error" in scene:
        print(f"❌ Error: {scene['error']}")
    else:
        print(json.dumps(scene, indent=2))
    
    print()
    
    # Example 2: Analyze a sentence
    print("🔍 EXAMPLE 2: Sentence Analysis")
    print("-" * 50)
    
    analysis = engine.analyze_sentence(
        "The pain system uses random.random() for severity scoring instead of intelligent mapping."
    )
    
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
    else:
        print(json.dumps(analysis, indent=2))
