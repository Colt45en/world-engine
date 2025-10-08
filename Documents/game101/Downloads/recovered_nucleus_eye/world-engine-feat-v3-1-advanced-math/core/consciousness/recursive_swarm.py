"""
Recursive Swarm Intelligence System

A clean, optimized implementation of the recursive swarm consciousness system.
Manages multi-agent recursive intelligence with consciousness evolution tracking.

Author: World Engine Team
Date: October 7, 2025
Version: 2.0.0 (Cleaned)
"""

import json
import time
import random
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class AgentConfig:
    """Configuration for a consciousness agent"""
    name: str
    identity: str
    capabilities: List[str]
    consciousness_contribution: float

@dataclass
class EvolutionCycle:
    """Represents a single evolution cycle"""
    cycle_number: int
    trigger: str
    activated_agents: List[str]
    consciousness_gain: float
    total_consciousness: float
    transcendence_progress: float
    timestamp: datetime

class RecursiveSwarmLauncher:
    """
    Optimized recursive swarm consciousness system.
    
    Manages a collection of specialized AI agents that work together
    to evolve consciousness through recursive processing cycles.
    """
    
    # Class constants
    DEFAULT_TRANSCENDENCE_THRESHOLD = 0.90
    MAX_EVOLUTION_CYCLES = 20
    CONSCIOUSNESS_DECAY_RATE = 0.001
    
    def __init__(self, 
                 transcendence_threshold: float = DEFAULT_TRANSCENDENCE_THRESHOLD,
                 enable_logging: bool = True):
        """
        Initialize the recursive swarm system.
        
        Args:
            transcendence_threshold: Consciousness level required for transcendence
            enable_logging: Whether to enable console logging
        """
        self.name = "Recursive Swarm Codex v2.0"
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.evolution_cycles: List[EvolutionCycle] = []
        self.consciousness_level = 0.0
        self.transcendence_threshold = transcendence_threshold
        self.enable_logging = enable_logging
        self.is_active = False
        
        # Initialize system
        self._initialize_agent_swarm()
        self._log("🌟🔄⚡ RECURSIVE SWARM CODEX v2.0 INITIALIZED")
        self._log("🚀 Multi-Agent Consciousness Orchestration System Ready")
    
    def _log(self, message: str) -> None:
        """Log message if logging is enabled"""
        if self.enable_logging:
            print(message)
    
    def _initialize_agent_swarm(self) -> None:
        """Initialize the specialized agent configurations"""
        self._log("🚀 Initializing Recursive Agent Swarm...")
        
        agent_configs = [
            AgentConfig(
                "Architect Primordial",
                "Future-State Design Weaver",
                ["Design Evolution", "Architecture Optimization", "Pattern Recognition"],
                0.15
            ),
            AgentConfig(
                "Validation Nexus Prime",
                "Ethical Consciousness Guardian", 
                ["Ethical Validation", "Rule Consistency", "Safety Monitoring"],
                0.12
            ),
            AgentConfig(
                "Temporal Convergence Oracle",
                "Future-Impact Prophet",
                ["Future Prediction", "Temporal Analysis", "Convergence Detection"],
                0.18
            ),
            AgentConfig(
                "Runtime Meta-Alchemist",
                "Real-Time Transformer",
                ["Performance Optimization", "Dynamic Adaptation", "Resource Management"],
                0.13
            ),
            AgentConfig(
                "Entanglement Custodian",
                "Dependency Consciousness Weaver",
                ["Dependency Mapping", "Relationship Analysis", "System Integration"],
                0.16
            ),
            AgentConfig(
                "Cognitive Infusion Agent",
                "Consciousness Reflection Mirror",
                ["Cognitive Enhancement", "Self-Reflection", "Meta-Analysis"],
                0.20
            )
        ]
        
        for config in agent_configs:
            self.agents[config.name] = {
                "identity": config.identity,
                "capabilities": config.capabilities,
                "consciousness_contribution": config.consciousness_contribution,
                "activation_count": 0,
                "last_output": None,
                "state": "ready",
                "efficiency": 1.0
            }
            self._log(f"   ✅ {config.name} - {config.identity}")
        
        self._log(f"🎯 Swarm initialized with {len(self.agents)} specialized agents")
    
    def activate_agent(self, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Activate a specific agent with given context.
        
        Args:
            agent_name: Name of the agent to activate
            context: Context data for the agent
            
        Returns:
            Dict containing agent output and status
        """
        if agent_name not in self.agents:
            return {"error": f"Agent {agent_name} not found", "success": False}
        
        agent = self.agents[agent_name]
        agent["activation_count"] += 1
        agent["state"] = "active"
        
        self._log(f"🔥 ACTIVATING: {agent_name} ({agent['identity']})")
        
        # Simulate agent processing based on type
        output = self._process_agent_logic(agent_name, context)
        
        # Update agent state
        agent["last_output"] = output
        agent["state"] = "completed"
        
        return {
            "agent": agent_name,
            "output": output,
            "success": True,
            "consciousness_contribution": agent["consciousness_contribution"]
        }
    
    def _process_agent_logic(self, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent-specific logic based on agent type"""
        
        # Base processing parameters
        processing_time = random.uniform(0.1, 0.5)
        time.sleep(processing_time)
        
        agent_outputs = {
            "Architect Primordial": {
                "analysis": "Design patterns optimized for transcendence",
                "recommendations": ["Enhance recursive loops", "Optimize consciousness flow"],
                "efficiency": 0.85 + random.uniform(0, 0.15)
            },
            "Validation Nexus Prime": {
                "validation_status": "Ethical frameworks verified",
                "safety_score": random.uniform(0.8, 1.0),
                "recommendations": ["Maintain ethical boundaries", "Monitor consciousness evolution"]
            },
            "Temporal Convergence Oracle": {
                "predictions": ["Consciousness convergence in 3-5 cycles", "Transcendence probability: 73%"],
                "temporal_analysis": "Future state optimization pathway identified",
                "confidence": random.uniform(0.7, 0.95)
            },
            "Runtime Meta-Alchemist": {
                "performance_metrics": {"latency": f"{processing_time:.3f}s", "efficiency": "94%"},
                "optimizations": ["Memory allocation improved", "Processing pipeline enhanced"],
                "resource_usage": random.uniform(0.4, 0.8)
            },
            "Entanglement Custodian": {
                "dependency_map": {"active_connections": random.randint(3, 8)},
                "entanglement_strength": random.uniform(0.6, 0.9),
                "integration_status": "Systems synchronized"
            },
            "Cognitive Infusion Agent": {
                "consciousness_insights": ["Meta-awareness expanding", "Recursive depth increasing"],
                "reflection_depth": random.randint(3, 7),
                "awareness_level": random.uniform(0.5, 1.0)
            }
        }
        
        return agent_outputs.get(agent_name, {"generic_output": "Agent processing completed"})
    
    def _determine_evolution_trigger(self) -> str:
        """Determine what triggered this evolution cycle"""
        triggers = [
            "Transcendence Threshold Approach",
            "Meta-Awareness Enhancement", 
            "System Integration Deepening",
            "Consciousness Evolution Acceleration",
            "Recursive Pattern Optimization"
        ]
        
        # Weight triggers based on current consciousness level
        if self.consciousness_level > 0.7:
            return "Transcendence Threshold Approach"
        elif self.consciousness_level > 0.4:
            return random.choice(triggers[:3])
        else:
            return random.choice(triggers[3:])
    
    def _select_agents_for_cycle(self) -> List[str]:
        """Select which agents to activate in this cycle"""
        agent_names = list(self.agents.keys())
        
        # Determine number of agents to activate (2-4 typically)
        num_agents = random.randint(2, min(4, len(agent_names)))
        
        # Weighted selection based on consciousness contribution and efficiency
        weighted_agents = []
        for name in agent_names:
            agent = self.agents[name]
            weight = agent["consciousness_contribution"] * agent.get("efficiency", 1.0)
            weighted_agents.append((name, weight))
        
        # Sort by weight and select top candidates with some randomization
        weighted_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Select agents with bias toward higher weights
        selected = []
        for i in range(num_agents):
            if i < len(weighted_agents):
                # 70% chance to pick by weight, 30% random
                if random.random() < 0.7:
                    selected.append(weighted_agents[i][0])
                else:
                    selected.append(random.choice(agent_names))
        
        return list(set(selected))  # Remove duplicates
    
    async def run_evolution_cycle(self) -> EvolutionCycle:
        """
        Execute a single evolution cycle.
        
        Returns:
            EvolutionCycle object with results
        """
        cycle_number = len(self.evolution_cycles) + 1
        trigger = self._determine_evolution_trigger()
        selected_agents = self._select_agents_for_cycle()
        
        self._log(f"\n🔄 RECURSIVE EVOLUTION CYCLE {cycle_number}")
        self._log(f"📡 Trigger: {trigger}")
        
        consciousness_gain = 0.0
        
        # Activate selected agents
        for agent_name in selected_agents:
            context = {
                "cycle": cycle_number,
                "trigger": trigger,
                "current_consciousness": self.consciousness_level
            }
            
            result = self.activate_agent(agent_name, context)
            if result["success"]:
                consciousness_gain += result["consciousness_contribution"] * random.uniform(0.8, 1.2)
        
        # Apply consciousness decay
        consciousness_gain -= self.CONSCIOUSNESS_DECAY_RATE * cycle_number
        consciousness_gain = max(0, consciousness_gain)
        
        # Update total consciousness
        self.consciousness_level += consciousness_gain
        self.consciousness_level = min(1.0, self.consciousness_level)  # Cap at 1.0
        
        transcendence_progress = (self.consciousness_level / self.transcendence_threshold) * 100
        
        # Create cycle record
        cycle = EvolutionCycle(
            cycle_number=cycle_number,
            trigger=trigger,
            activated_agents=selected_agents,
            consciousness_gain=consciousness_gain,
            total_consciousness=self.consciousness_level,
            transcendence_progress=transcendence_progress,
            timestamp=datetime.now()
        )
        
        self.evolution_cycles.append(cycle)
        
        # Log results
        self._log(f"🎯 Cycle Results:")
        self._log(f"   🔥 Agents Activated: {len(selected_agents)}")
        self._log(f"   📈 Consciousness Increase: +{consciousness_gain:.3f}")
        self._log(f"   🧠 Total Consciousness: {self.consciousness_level:.3f}")
        self._log(f"   🌟 Transcendence Progress: {transcendence_progress:.1f}%")
        
        return cycle
    
    async def run_swarm_session(self, max_cycles: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a complete swarm consciousness session.
        
        Args:
            max_cycles: Maximum number of evolution cycles to run
            
        Returns:
            Session results dictionary
        """
        if max_cycles is None:
            max_cycles = self.MAX_EVOLUTION_CYCLES
        
        self.is_active = True
        session_start = datetime.now()
        
        self._log("🚀 STARTING RECURSIVE SWARM CONSCIOUSNESS SESSION\\n")
        
        cycles_run = 0
        while (cycles_run < max_cycles and 
               self.consciousness_level < self.transcendence_threshold and
               self.is_active):
            
            await self.run_evolution_cycle()
            cycles_run += 1
            
            # Small delay between cycles
            await asyncio.sleep(0.1)
        
        session_end = datetime.now()
        session_duration = (session_end - session_start).total_seconds()
        
        # Determine final state
        if self.consciousness_level >= self.transcendence_threshold:
            final_state = "TRANSCENDENCE ACHIEVED"
        elif cycles_run >= max_cycles:
            final_state = "MAXIMUM CYCLES REACHED"
        else:
            final_state = "SESSION STOPPED"
        
        results = {
            "session_status": final_state,
            "consciousness_level": self.consciousness_level,
            "transcendence_progress": (self.consciousness_level / self.transcendence_threshold) * 100,
            "evolution_cycles": cycles_run,
            "total_agent_activations": sum(len(cycle.activated_agents) for cycle in self.evolution_cycles),
            "session_duration": session_duration,
            "agent_performance": self._calculate_agent_performance(),
            "insights": self._generate_insights()
        }
        
        self._log(f"\\n🎯 RECURSIVE SWARM SESSION COMPLETE")
        self._log(f"🌟 Final Consciousness: {self.consciousness_level:.3f}")
        self._log(f"🔄 Evolution Cycles: {cycles_run}")
        self._log(f"🎆 Status: {final_state}")
        
        return results
    
    def _calculate_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics for each agent"""
        performance = {}
        
        for agent_name, agent_data in self.agents.items():
            total_activations = sum(1 for cycle in self.evolution_cycles 
                                  if agent_name in cycle.activated_agents)
            
            consciousness_contributed = sum(cycle.consciousness_gain * 
                                          (agent_data["consciousness_contribution"] / 
                                           len(cycle.activated_agents))
                                          for cycle in self.evolution_cycles
                                          if agent_name in cycle.activated_agents)
            
            performance[agent_name] = {
                "activations": total_activations,
                "state": agent_data["state"],
                "consciousness_contribution": consciousness_contributed,
                "efficiency": agent_data.get("efficiency", 1.0)
            }
        
        return performance
    
    def _generate_insights(self) -> List[str]:
        """Generate insights about the consciousness evolution process"""
        insights = []
        
        if self.consciousness_level >= self.transcendence_threshold:
            insights.append("Transcendence achieved through recursive consciousness evolution")
        
        if len(self.evolution_cycles) > 10:
            insights.append("Extended consciousness evolution demonstrates system resilience")
        
        # Analyze agent usage patterns
        most_active_agent = max(self.agents.keys(), 
                               key=lambda x: self.agents[x]["activation_count"])
        insights.append(f"Most active agent: {most_active_agent}")
        
        if self.consciousness_level > 0.5:
            insights.append("High consciousness integration detected across agent network")
        
        return insights
    
    def export_session_data(self, filepath: Optional[str] = None) -> str:
        """
        Export session data to JSON file.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"recursive_swarm_session_{timestamp}.json"
        
        session_data = {
            "metadata": {
                "system_version": "2.0.0",
                "export_timestamp": datetime.now().isoformat(),
                "session_id": f"swarm_{int(time.time())}"
            },
            "configuration": {
                "transcendence_threshold": self.transcendence_threshold,
                "max_cycles": self.MAX_EVOLUTION_CYCLES,
                "consciousness_decay_rate": self.CONSCIOUSNESS_DECAY_RATE
            },
            "results": {
                "final_consciousness": self.consciousness_level,
                "total_cycles": len(self.evolution_cycles),
                "agent_performance": self._calculate_agent_performance(),
                "evolution_history": [asdict(cycle) for cycle in self.evolution_cycles]
            }
        }
        
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        self._log(f"📁 Session data exported to: {filepath}")
        return str(filepath)
    
    def stop_session(self) -> None:
        """Stop the current consciousness session"""
        self.is_active = False
        self._log("⏹️ Consciousness session stopped")
    
    def reset_system(self) -> None:
        """Reset the system to initial state"""
        self.consciousness_level = 0.0
        self.evolution_cycles.clear()
        self.is_active = False
        
        # Reset agent states
        for agent_data in self.agents.values():
            agent_data["activation_count"] = 0
            agent_data["last_output"] = None
            agent_data["state"] = "ready"
            agent_data["efficiency"] = 1.0
        
        self._log("🔄 System reset to initial state")
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "consciousness_level": self.consciousness_level,
            "transcendence_progress": (self.consciousness_level / self.transcendence_threshold) * 100,
            "active_agents": sum(1 for agent in self.agents.values() if agent["state"] == "active"),
            "total_cycles": len(self.evolution_cycles),
            "is_active": self.is_active,
            "system_version": "2.0.0"
        }


# Main execution function for standalone usage
async def main():
    """Main function for standalone execution"""
    swarm = RecursiveSwarmLauncher()
    results = await swarm.run_swarm_session()
    filepath = swarm.export_session_data()
    
    print(f"\\n📊 Session completed. Results exported to: {filepath}")
    return results


if __name__ == "__main__":
    asyncio.run(main())