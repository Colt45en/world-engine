#!/usr/bin/env python3
"""
Quantum Protocol Unity Integration Example
=========================================

Demonstrates the complete quantum protocol system working with Unity-style
agent management, environmental events, recursive analysis, and swarm mind coordination.
"""

import asyncio
import json
import logging
import math
import random
import time
from typing import List, Dict, Any

# Import our quantum protocol system
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

try:
    from quantum_protocol_daemon import (
        QuantumProtocolDaemon, Vector3, MathFunctionType, QuantumEventType,
        EnvironmentalEventType, QuantumAgent, QuantumStep, SwarmAnalysis
    )
except ImportError:
    print("Error: quantum_protocol_daemon not found. Please ensure it's in the same directory.")
    sys.exit(1)

class UnityQuantumIntegrationDemo:
    """
    Demonstrates Unity-style quantum protocol integration with:
    - Multi-agent path simulation
    - Environmental event spawning
    - Real-time collapse detection
    - Swarm mind analysis
    - Memory ghost replay
    - Recursive infrastructure evolution
    """

    def __init__(self):
        self.daemon = QuantumProtocolDaemon()
        self.agents: List[str] = []
        self.simulation_running = False
        self.demo_scenarios = [
            "radial_convergence",
            "sinusoidal_waves",
            "chaotic_dispersion",
            "recursive_nexus"
        ]

        # Set up event handlers
        self.daemon.event_callbacks.append(self.handle_protocol_event)

        # Configure for demonstration
        self.daemon.update_frequency = 60.0  # Higher frequency for demo
        self.daemon.auto_collapse_enabled = True
        self.daemon.collapse_threshold = 0.7

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def handle_protocol_event(self, event_type: QuantumEventType, data: Dict[str, Any]):
        """Handle protocol events Unity-style"""
        timestamp = time.strftime('%H:%M:%S')

        if event_type == QuantumEventType.AGENT_COLLAPSE:
            agent_id = data.get('agent_id', 'unknown')
            print(f"\n🌀 [{timestamp}] QUANTUM COLLAPSE!")
            print(f"   Agent: {agent_id}")
            print(f"   Initiating memory ghost spawn...")
            print(f"   Displaying score glyph...")
            print(f"   Triggering particle burst...")
            self._spawn_memory_ghost(agent_id)

        elif event_type == QuantumEventType.SWARM_CONVERGENCE:
            print(f"\n🧠 [{timestamp}] SWARM CONVERGENCE DETECTED!")
            print(f"   Collective mind awakening...")
            print(f"   Replaying all quantum paths...")
            print(f"   Fading reality matrix...")
            self._trigger_global_effects()

        elif event_type == QuantumEventType.FUNCTION_CHANGE:
            func_type = data.get('function_type', 'unknown')
            print(f"\n⚡ [{timestamp}] REALITY FUNCTION SHIFT!")
            print(f"   New function: {func_type}")
            print(f"   Updating global shader parameters...")
            print(f"   Synchronizing trail palettes...")

        elif event_type == QuantumEventType.ENVIRONMENTAL_EVENT:
            print(f"\n🌪️ [{timestamp}] ENVIRONMENTAL DISTURBANCE!")
            print(f"   Event data: {data}")

    def _spawn_memory_ghost(self, agent_id: str):
        """Unity-style memory ghost spawning"""
        path = self.daemon.get_agent_path(agent_id)
        if path:
            print(f"   🌟 Memory ghost spawned with {len(path)} quantum steps")
            print(f"   👻 Replaying path from {path[0].position} to {path[-1].position}")

            # Simulate ghost replay visualization
            for i, step in enumerate(path[-5:]):  # Show last 5 steps
                print(f"      Step {step.step_number}: pos=({step.position.x:.1f}, {step.position.y:.1f}, {step.position.z:.1f}) energy={step.energy_level:.2f}")

    def _trigger_global_effects(self):
        """Unity-style global effect triggering"""
        print("   🎭 Activating quantum lore archive...")
        print("   🎵 Playing ethereal collapse symphony...")
        print("   💫 Manifesting collective memory matrix...")

        # Get swarm analysis
        analysis = self.daemon.get_analysis()
        print(f"   📊 Convergence Metric: {analysis.convergence_metric:.3f}")
        print(f"   🎯 Dominant Theme: {analysis.dominant_theme}")
        print(f"   🔗 Total Nodes: {analysis.total_nodes}")

    async def run_demo_scenario(self, scenario_name: str):
        """Run a specific demonstration scenario"""
        print(f"\n🚀 Starting scenario: {scenario_name.upper()}")
        print("=" * 60)

        if scenario_name == "radial_convergence":
            await self._demo_radial_convergence()
        elif scenario_name == "sinusoidal_waves":
            await self._demo_sinusoidal_waves()
        elif scenario_name == "chaotic_dispersion":
            await self._demo_chaotic_dispersion()
        elif scenario_name == "recursive_nexus":
            await self._demo_recursive_nexus()
        else:
            print(f"❌ Unknown scenario: {scenario_name}")

    async def _demo_radial_convergence(self):
        """Demo: Agents converging in radial patterns"""
        print("📡 Spawning radial convergence agents...")

        # Create agents in radial formation
        agent_count = 6
        for i in range(agent_count):
            angle = (i / agent_count) * 2 * math.pi
            radius = 10.0
            agent_id = f"radial_agent_{i}"
            position = Vector3(
                math.cos(angle) * radius,
                math.sin(angle) * radius,
                0
            )

            self.daemon.register_agent(agent_id, position)
            self.agents.append(agent_id)
            print(f"   🎯 Agent {agent_id} at ({position.x:.1f}, {position.y:.1f})")

        # Simulate convergent movement
        steps = 50
        for step in range(steps):
            for i, agent_id in enumerate(self.agents):
                # Move toward center with some spiral motion
                convergence_factor = step / steps
                angle = (i / len(self.agents)) * 2 * math.pi + step * 0.1
                radius = 10.0 * (1 - convergence_factor)

                position = Vector3(
                    math.cos(angle) * radius,
                    math.sin(angle) * radius,
                    math.sin(step * 0.2) * 2.0  # Vertical oscillation
                )

                energy = 1.0 - (step / steps) * 0.5  # Gradual energy decrease
                self.daemon.register_step(position, agent_id, step, energy)

            # Spawn environmental events occasionally
            if step % 15 == 0:
                event_origin = Vector3(
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                    0
                )
                self.daemon.spawn_environmental_event(
                    random.choice(list(EnvironmentalEventType)),
                    event_origin,
                    radius=3.0,
                    duration=5.0
                )
                print(f"   🌪️ Environmental event spawned at step {step}")

            await asyncio.sleep(0.1)  # Simulation timestep

            # Check for convergence
            if step > 30:
                analysis = self.daemon.get_analysis()
                if analysis.convergence_metric > 0.8:
                    print(f"   ⚡ High convergence detected: {analysis.convergence_metric:.3f}")
                    break

    async def _demo_sinusoidal_waves(self):
        """Demo: Agents following sinusoidal wave patterns"""
        print("🌊 Generating sinusoidal wave agents...")

        # Create wave agents
        wave_count = 4
        for wave in range(wave_count):
            agent_id = f"wave_agent_{wave}"
            initial_position = Vector3(0, wave * 2, 0)
            self.daemon.register_agent(agent_id, initial_position)
            self.agents.append(agent_id)

        # Change to wave function
        self.daemon.ray_field_manager.set_function_type(MathFunctionType.WAVE)

        # Simulate wave motion
        steps = 60
        for step in range(steps):
            for wave, agent_id in enumerate(self.agents):
                # Different wave parameters per agent
                frequency = 0.1 + wave * 0.05
                amplitude = 3.0 + wave * 0.5
                phase = wave * math.pi / 2

                position = Vector3(
                    step * 0.3,
                    math.sin(step * frequency + phase) * amplitude,
                    math.cos(step * frequency * 0.5 + phase) * 1.0
                )

                energy = 0.8 + 0.2 * math.sin(step * 0.2 + phase)
                self.daemon.register_step(position, agent_id, step, energy)

            # Change function occasionally
            if step == 20:
                self.daemon.ray_field_manager.set_function_type(MathFunctionType.MULTIWAVE)
            elif step == 40:
                self.daemon.ray_field_manager.set_function_type(MathFunctionType.RIPPLE)

            await asyncio.sleep(0.08)

    async def _demo_chaotic_dispersion(self):
        """Demo: Chaotic agent behavior with environmental interference"""
        print("🌀 Initiating chaotic dispersion simulation...")

        # Create chaotic agents
        chaos_count = 8
        for i in range(chaos_count):
            agent_id = f"chaos_agent_{i}"
            position = Vector3(
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(-1, 1)
            )
            self.daemon.register_agent(agent_id, position)
            self.agents.append(agent_id)

        # Spawn multiple environmental disturbances
        disturbance_centers = [
            Vector3(5, 5, 0),
            Vector3(-5, 5, 0),
            Vector3(5, -5, 0),
            Vector3(-5, -5, 0)
        ]

        for center in disturbance_centers:
            self.daemon.spawn_environmental_event(
                EnvironmentalEventType.REALITY_DISTORTION,
                center,
                radius=4.0,
                duration=20.0
            )

        # Simulate chaotic motion
        steps = 80
        for step in range(steps):
            for agent_id in self.agents:
                # Chaotic movement with attractor influence
                chaos_factor = math.sin(step * 0.3) * math.cos(step * 0.17)

                # Random walk with bias toward nearest disturbance
                agent = self.daemon.ray_field_manager.get_agent(agent_id)
                if agent:
                    current_pos = agent.position

                    # Find nearest disturbance
                    nearest_dist = float('inf')
                    nearest_center = disturbance_centers[0]
                    for center in disturbance_centers:
                        dist = current_pos.distance(center)
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_center = center

                    # Move toward/away from disturbance chaotically
                    direction_x = (nearest_center.x - current_pos.x) / max(nearest_dist, 1.0)
                    direction_y = (nearest_center.y - current_pos.y) / max(nearest_dist, 1.0)

                    new_position = Vector3(
                        current_pos.x + direction_x * chaos_factor + random.uniform(-0.5, 0.5),
                        current_pos.y + direction_y * chaos_factor + random.uniform(-0.5, 0.5),
                        math.sin(step * 0.1) + random.uniform(-0.3, 0.3)
                    )

                    energy = 0.5 + 0.5 * abs(chaos_factor)
                    self.daemon.register_step(new_position, agent_id, step, energy)

            # Add more disturbances dynamically
            if step % 25 == 0:
                random_center = Vector3(
                    random.uniform(-8, 8),
                    random.uniform(-8, 8),
                    0
                )
                self.daemon.spawn_environmental_event(
                    EnvironmentalEventType.FLUX_SURGE,
                    random_center,
                    radius=2.0,
                    duration=8.0
                )

            await asyncio.sleep(0.06)

    async def _demo_recursive_nexus(self):
        """Demo: Recursive infrastructure analysis with quantum nexus activation"""
        print("🧠 Activating recursive quantum nexus...")

        # Create nexus agents
        nexus_count = 3
        nexus_topics = [
            "Quantum Consciousness Interface",
            "Recursive Memory Architecture",
            "Swarm Intelligence Convergence"
        ]

        for i, topic in enumerate(nexus_topics):
            agent_id = f"nexus_agent_{i}"
            position = Vector3(i * 4, 0, 0)
            self.daemon.register_agent(agent_id, position)
            self.agents.append(agent_id)

            # Start recursive analysis for each agent's topic
            print(f"   🔄 Initiating recursive analysis: '{topic}'")
            self.daemon.activate_nexus(topic)

        # Simulate nexus evolution
        steps = 40
        for step in range(steps):
            # Get current infrastructure memory
            memory = self.daemon.recursive_flow.get_memory()
            forward_memory = memory["forward_memory"]

            if forward_memory:
                latest_node = forward_memory[-1]
                print(f"   🌟 Step {step}: Latest topic: '{latest_node.topic}'")
                print(f"      Depth: {latest_node.iteration_depth}")
                print(f"      Derived: '{latest_node.derived_topic}'")

            # Move agents in patterns reflecting recursive depth
            for i, agent_id in enumerate(self.agents):
                depth_factor = len(forward_memory) / 20.0 if forward_memory else 0.1
                spiral_factor = step * 0.2 + i * math.pi * 2 / 3

                position = Vector3(
                    i * 4 + math.cos(spiral_factor) * depth_factor,
                    math.sin(spiral_factor) * depth_factor,
                    math.sin(step * 0.15 + i) * depth_factor
                )

                energy = 0.7 + 0.3 * depth_factor
                self.daemon.register_step(position, agent_id, step, energy)

            # Trigger more recursive analysis periodically
            if step % 12 == 0 and forward_memory:
                next_topic = forward_memory[-1].derived_topic
                print(f"   🔄 Continuing recursion: '{next_topic}'")
                self.daemon.recursive_flow.analyze_topic(next_topic, len(forward_memory))

            await asyncio.sleep(0.15)

        # Final swarm analysis
        print("\n📊 Final Nexus Analysis:")
        analysis = self.daemon.get_analysis()
        print(f"   Total Nodes: {analysis.total_nodes}")
        print(f"   Convergence: {analysis.convergence_metric:.3f}")
        print(f"   Dominant Theme: {analysis.dominant_theme}")
        print(f"   Unique Topics: {len(analysis.unique_topics)}")

        # Show infrastructure memory summary
        memory = self.daemon.recursive_flow.get_memory()
        print(f"   Infrastructure Memory: {len(memory['forward_memory'])} nodes")
        if memory['forward_memory']:
            print(f"   Last Topic: '{memory['forward_memory'][-1].topic}'")

    async def run_full_demo(self):
        """Run all demonstration scenarios"""
        print("🌌 QUANTUM PROTOCOL UNITY INTEGRATION DEMO")
        print("=" * 80)
        print("Demonstrating Unity-style quantum systems with:")
        print("• Multi-agent path simulation")
        print("• Environmental event propagation")
        print("• Real-time collapse detection")
        print("• Swarm mind coordination")
        print("• Recursive infrastructure analysis")
        print("• Memory ghost replay")
        print("=" * 80)

        # Start daemon
        self.daemon.start()
        await asyncio.sleep(1)  # Let daemon initialize

        try:
            # Run each scenario
            for scenario in self.demo_scenarios:
                await self.run_demo_scenario(scenario)

                # Show intermediate analysis
                analysis = self.daemon.get_analysis()
                print(f"\n📈 Scenario '{scenario}' Complete!")
                print(f"   Agents: {len(self.agents)}")
                print(f"   Total Steps: {len(self.daemon.ray_field_manager.get_all_steps())}")
                print(f"   Convergence: {analysis.convergence_metric:.3f}")

                # Resolve amplitudes
                result = self.daemon.amplitude_resolver.resolve_and_collapse()
                if result and result['winner_id']:
                    print(f"   🏆 Amplitude Winner: {result['winner_id']} ({result['max_amplitude']:.2f})")

                # Clear agents for next scenario
                self.agents.clear()
                self.daemon.ray_field_manager.agents.clear()

                print(f"\n⏳ Scenario transition pause...")
                await asyncio.sleep(2)

        finally:
            # Clean shutdown
            print("\n🛑 Stopping quantum protocol daemon...")
            self.daemon.stop()

            # Final statistics
            all_steps = self.daemon.ray_field_manager.get_all_steps()
            print(f"\n📊 FINAL DEMO STATISTICS:")
            print(f"   Total Quantum Steps Recorded: {len(all_steps)}")
            print(f"   Total Environmental Events: {len(self.daemon.environmental_system.active_events)}")
            print(f"   Infrastructure Nodes Generated: {len(self.daemon.recursive_flow.forward_memory)}")

            print("\n✨ Quantum protocol demonstration complete!")
            print("🌀 The nexus remembers all paths traveled...")

async def main():
    """Main demonstration entry point"""
    demo = UnityQuantumIntegrationDemo()
    await demo.run_full_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚡ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
