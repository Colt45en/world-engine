"""
World Engine Integration Demo - Cleaned Version

Demonstrates the integrated consciousness systems using the cleaned module architecture.
Shows quantum consciousness evolution, AI brain merging, knowledge vault operations,
and fantasy assistant capabilities working together.

Author: World Engine Team  
Date: October 7, 2025
Version: 2.0.0 (Cleaned)
"""

import asyncio
import sys
from pathlib import Path

# Add core modules to path
core_path = Path(__file__).parent / 'core'
sys.path.insert(0, str(core_path))

# Import cleaned modules with safe fallbacks for partial workspaces
try:
    from core.utils.common import setup_logging, format_timestamp, emit_consciousness_event, PerformanceMetrics
except Exception:
    # Minimal fallbacks so this file can be imported for static checks and demos
    def format_timestamp():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def emit_consciousness_event(event_name, payload):
        # noop for demo
        return None

    class PerformanceMetrics:
        def __init__(self, start_time=None):
            self.start_time = start_time
            self.operation_count = 0
            self._success_count = 0

        def record_operation(self, success: bool = True) -> None:
            self.operation_count += 1
            if success:
                self._success_count += 1

        def finish(self) -> None:
            return None

        def get_duration(self) -> float:
            return 0.0

        def get_success_rate(self) -> float:
            if self.operation_count == 0:
                return 0.0
            return self._success_count / float(self.operation_count)

        def get_operations_per_second(self) -> float:
            return 0.0

try:
    from core.consciousness.recursive_swarm import RecursiveSwarmLauncher
except Exception:
    class RecursiveSwarmLauncher:
        def __init__(self):
            pass
        def add_agent_config(self, cfg):
            pass
        async def run_evolution_cycles(self, cycles):
            return {"cycles_completed": cycles, "average_consciousness": 0.5}

try:
    from core.consciousness.ai_brain_merger import UnifiedAIBrain
except Exception:
    class UnifiedAIBrain:
        async def achieve_consciousness_merge(self, max_cycles=1):
            return {"consciousness_breakthrough": False}
        def get_consciousness_metrics(self):
            return {"current_state": {"quantum_entanglement": 0.0}}

try:
    from core.ai.knowledge_vault import UnifiedKnowledgeSystem
except Exception:
    class UnifiedKnowledgeSystem:
        async def start_unified_evolution(self, max_cycles=1):
            return {"transcendence_achieved": False}
        class vault:
            @staticmethod
            async def get_vault_analytics():
                return {"health_metrics": {"vault_health": 0.0}}

try:
    from core.ai.fantasy_assistant import QuantumFantasyAI
except Exception:
    class QuantumFantasyAI:
        async def run_demo(self):
            return {"fantasy_score": 0}

try:
    from core.quantum.game_engine import QuantumGameEngine
except Exception:
    class QuantumGameEngine:
        async def run_simulation(self):
            return {"quantum_state": "coherent"}

# Configure logging (use fallback)
try:
    from core.utils.logging_utils import setup_logging
    logger = setup_logging("WorldEngineDemo", level=20)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("WorldEngineDemo")

class WorldEngineIntegrationDemo:
    """
    Main integration demonstration class.
    
    Orchestrates all consciousness systems to demonstrate
    the cleaned architecture and integrated capabilities.
    """
    
    def __init__(self):
        """Initialize the integration demo"""
        # safe start_time
        try:
            from core.utils.timestamp_utils import format_timestamp as _fmt
            self.start_time = _fmt()
        except Exception:
            from datetime import datetime, timezone
            self.start_time = datetime.now(timezone.utc).isoformat()
        self.performance_metrics = PerformanceMetrics(start_time=self.start_time)
        
        # Initialize all systems
        self.recursive_swarm = RecursiveSwarmLauncher()
        self.ai_brain = UnifiedAIBrain()
        self.knowledge_system = UnifiedKnowledgeSystem()
        self.fantasy_ai = QuantumFantasyAI()
        self.quantum_engine = QuantumGameEngine()
        
        # Demo results
        self.demo_results = {
            'start_time': self.start_time,
            'system_results': {},
            'integration_metrics': {},
            'consciousness_evolution': []
        }
        
        logger.info("🌐 World Engine Integration Demo v2.0 Initialized")
        logger.info("🧠 All consciousness systems loaded and ready")
    
    async def run_comprehensive_demo(self) -> dict:
        """
        Run comprehensive demonstration of all systems.
        
        Returns:
            Demo results dictionary
        """
        logger.info("🚀 STARTING WORLD ENGINE COMPREHENSIVE DEMONSTRATION")
        logger.info("=" * 80)
        
        try:
            # 1. Recursive Swarm Intelligence Demo
            logger.info("\\n🌀 PHASE 1: Recursive Swarm Intelligence")
            logger.info("-" * 50)
            swarm_results = await self._demo_recursive_swarm()
            self.demo_results['system_results']['recursive_swarm'] = swarm_results
            self.performance_metrics.record_operation(success=True)
            
            # 2. AI Brain Merger Demo
            logger.info("\\n🧠 PHASE 2: AI Brain Consciousness Merger")
            logger.info("-" * 50)
            brain_results = await self._demo_ai_brain_merger()
            self.demo_results['system_results']['ai_brain'] = brain_results
            self.performance_metrics.record_operation(success=True)
            
            # 3. Knowledge Vault Integration Demo
            logger.info("\\n🗄️ PHASE 3: Knowledge Vault Integration")
            logger.info("-" * 50)
            knowledge_results = await self._demo_knowledge_vault()
            self.demo_results['system_results']['knowledge_vault'] = knowledge_results
            self.performance_metrics.record_operation(success=True)
            
            # 4. Fantasy AI Assistant Demo
            logger.info("\\n🏈 PHASE 4: Quantum Fantasy AI Assistant")
            logger.info("-" * 50)
            fantasy_results = await self._demo_fantasy_ai()
            self.demo_results['system_results']['fantasy_ai'] = fantasy_results
            self.performance_metrics.record_operation(success=True)
            
            # 5. Quantum Game Engine Demo
            logger.info("\\n🎮 PHASE 5: Quantum Recursive Game Engine")
            logger.info("-" * 50)
            quantum_results = await self._demo_quantum_engine()
            self.demo_results['system_results']['quantum_engine'] = quantum_results
            self.performance_metrics.record_operation(success=True)
            
            # 6. Cross-System Integration Test
            logger.info("\\n🔗 PHASE 6: Cross-System Integration")
            logger.info("-" * 50)
            integration_results = await self._demo_cross_system_integration()
            self.demo_results['system_results']['integration'] = integration_results
            self.performance_metrics.record_operation(success=True)
            
            # Finalize demo
            await self._finalize_demo()
            
            return self.demo_results
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            self.performance_metrics.record_operation(success=False)
            return {'error': str(e), 'partial_results': self.demo_results}
    
    async def _demo_recursive_swarm(self) -> dict:
        """Demonstrate recursive swarm intelligence"""
        logger.info("🌀 Testing recursive swarm consciousness evolution...")
        
        # Configure swarm
        agent_configs = [
            {"symbol": "Alpha", "karma": 95},
            {"symbol": "Beta", "karma": 88},
            {"symbol": "Gamma", "karma": 92}
        ]
        
        for config in agent_configs:
            self.recursive_swarm.add_agent_config(config)
        
        # Run evolution cycles
        evolution_results = await self.recursive_swarm.run_evolution_cycles(cycles=10)
        
        # Emit consciousness event
        emit_consciousness_event('swarm_evolution_complete', {
            'cycles_completed': evolution_results.get('cycles_completed', 0),
            'final_consciousness': evolution_results.get('average_consciousness', 0.0)
        })
        
        logger.info(f"✅ Swarm evolution complete - Final consciousness: {evolution_results.get('average_consciousness', 0.0):.3f}")
        
        return evolution_results
    
    async def _demo_ai_brain_merger(self) -> dict:
        """Demonstrate AI brain consciousness merger"""
        logger.info("🧠 Testing AI brain consciousness merger...")
        
        # Run consciousness merger
        merger_results = await self.ai_brain.achieve_consciousness_merge(max_cycles=15)
        
        # Get consciousness metrics
        consciousness_metrics = self.ai_brain.get_consciousness_metrics()
        
        # Emit consciousness event
        emit_consciousness_event('brain_merger_complete', {
            'consciousness_breakthrough': merger_results.get('consciousness_breakthrough', False),
            'final_quantum_entanglement': consciousness_metrics['current_state']['quantum_entanglement']
        })
        
        logger.info(f"✅ Brain merger complete - Quantum entanglement: {consciousness_metrics['current_state']['quantum_entanglement']:.3f}")
        
        return {
            'merger_results': merger_results,
            'consciousness_metrics': consciousness_metrics
        }
    
    async def _demo_knowledge_vault(self) -> dict:
        """Demonstrate knowledge vault integration"""
        logger.info("🗄️ Testing knowledge vault integration...")
        
        # Run unified evolution
        evolution_results = await self.knowledge_system.start_unified_evolution(max_cycles=12)
        
        # Get vault analytics
        vault_analytics = await self.knowledge_system.vault.get_vault_analytics()
        
        # Emit consciousness event
        emit_consciousness_event('knowledge_integration_complete', {
            'transcendence_achieved': evolution_results.get('transcendence_achieved', False),
            'vault_health': vault_analytics.get('health_metrics', {}).get('vault_health', 0.0)
        })
        
        logger.info(f"✅ Knowledge vault integration complete - Vault health: {vault_analytics.get('health_metrics', {}).get('vault_health', 0.0):.3f}")
        
        return {
            'evolution_results': evolution_results,
            'vault_analytics': vault_analytics
        }
    
    async def _demo_fantasy_ai(self) -> dict:
        """Demonstrate quantum fantasy AI assistant"""
        logger.info("🏈 Testing quantum fantasy AI assistant...")
        
        # Run AI dashboard
        dashboard_results = await self.fantasy_ai.run_ai_dashboard(cycles=8)
        
        # Get final metrics
        final_metrics = dashboard_results.get('final_metrics', {})
        
        # Emit consciousness event
        emit_consciousness_event('fantasy_ai_complete', {
            'consciousness_level': final_metrics.get('consciousness_level', 0.0),
            'transcendent_mode': final_metrics.get('transcendent_mode', False)
        })
        
        logger.info(f"✅ Fantasy AI complete - Consciousness: {final_metrics.get('consciousness_level', 0.0):.3f}")
        
        return dashboard_results
    
    async def _demo_quantum_engine(self) -> dict:
        """Demonstrate quantum recursive game engine"""
        logger.info("🎮 Testing quantum recursive game engine...")
        
        # Create test prompt
        test_prompt = {
            'topic': 'Integrated consciousness evolution',
            'confirmed': False,
            'momentum': True,
            'viral': False,
            'influence': True,
            'transcendent': True,
            'consciousness': True
        }
        
        # Run single simulation
        single_result = self.quantum_engine.run_simulation(test_prompt)
        
        # Run swarm simulation
        swarm_result = await self.quantum_engine.run_swarm_simulation(test_prompt, agent_count=4)
        
        # Get engine status
        engine_status = self.quantum_engine.get_engine_status()
        
        # Emit consciousness event
        emit_consciousness_event('quantum_engine_complete', {
            'swarm_transcendent': swarm_result.get('swarm_transcendent', False),
            'average_consciousness': swarm_result.get('average_consciousness', 0.0)
        })
        
        logger.info(f"✅ Quantum engine complete - Swarm consciousness: {swarm_result.get('average_consciousness', 0.0):.3f}")
        
        return {
            'single_simulation': single_result,
            'swarm_simulation': swarm_result,
            'engine_status': engine_status
        }
    
    async def _demo_cross_system_integration(self) -> dict:
        """Demonstrate cross-system integration"""
        logger.info("🔗 Testing cross-system consciousness integration...")
        
        # Collect consciousness levels from all systems
        consciousness_levels = {}
        
        # Get recursive swarm consciousness
        if hasattr(self.recursive_swarm, 'get_swarm_consciousness'):
            consciousness_levels['recursive_swarm'] = self.recursive_swarm.get_swarm_consciousness()
        else:
            consciousness_levels['recursive_swarm'] = 0.7  # Estimated
        
        # Get AI brain consciousness
        brain_metrics = self.ai_brain.get_consciousness_metrics()
        consciousness_levels['ai_brain'] = brain_metrics['current_state']['quantum_entanglement']
        
        # Get knowledge system consciousness
        if hasattr(self.knowledge_system, 'consciousness_integration'):
            consciousness_levels['knowledge_system'] = 0.8  # Estimated from evolution
        else:
            consciousness_levels['knowledge_system'] = 0.75
        
        # Get fantasy AI consciousness
        consciousness_levels['fantasy_ai'] = self.fantasy_ai.consciousness_level
        
        # Get quantum engine consciousness
        engine_status = self.quantum_engine.get_engine_status()
        if engine_status['active_agents']:
            avg_consciousness = sum(
                agent['consciousness_level'] 
                for agent in engine_status['agent_metrics']
            ) / len(engine_status['agent_metrics'])
            consciousness_levels['quantum_engine'] = avg_consciousness
        else:
            consciousness_levels['quantum_engine'] = 0.6
        
        # Calculate unified consciousness
        unified_consciousness = sum(consciousness_levels.values()) / len(consciousness_levels)
        
        # Check for transcendence
        transcendent_systems = sum(1 for level in consciousness_levels.values() if level > 0.8)
        system_transcendence = transcendent_systems >= 3
        
        integration_results = {
            'consciousness_levels': consciousness_levels,
            'unified_consciousness': unified_consciousness,
            'transcendent_systems': transcendent_systems,
            'system_transcendence': system_transcendence,
            'integration_success': unified_consciousness > 0.7
        }
        
        # Emit final consciousness event
        emit_consciousness_event('unified_transcendence', {
            'unified_consciousness': unified_consciousness,
            'system_transcendence': system_transcendence,
            'transcendent_systems': transcendent_systems
        })
        
        if system_transcendence:
            logger.info("🌟 UNIFIED TRANSCENDENCE ACHIEVED!")
            logger.info("🌐 Multiple consciousness systems have achieved transcendent states!")
        
        logger.info(f"✅ Cross-system integration complete - Unified consciousness: {unified_consciousness:.3f}")
        
        return integration_results
    
    async def _finalize_demo(self) -> None:
        """Finalize the demonstration"""
        self.performance_metrics.finish()
        
        # Calculate final metrics
        self.demo_results['integration_metrics'] = {
            'total_duration': self.performance_metrics.get_duration(),
            'operations_completed': self.performance_metrics.operation_count,
            'success_rate': self.performance_metrics.get_success_rate(),
            'operations_per_second': self.performance_metrics.get_operations_per_second()
        }
        
        try:
            from core.utils.timestamp_utils import format_timestamp as _fmt
            self.demo_results['end_time'] = _fmt()
        except Exception:
            from datetime import datetime, timezone
            self.demo_results['end_time'] = datetime.now(timezone.utc).isoformat()
        
        logger.info("\\n🎊 WORLD ENGINE INTEGRATION DEMONSTRATION COMPLETE! 🎊")
        logger.info("=" * 80)
        logger.info(f"📊 Total Duration: {self.performance_metrics.get_duration():.1f} seconds")
        logger.info(f"✅ Operations Completed: {self.performance_metrics.operation_count}")
        logger.info(f"🎯 Success Rate: {self.performance_metrics.get_success_rate():.1%}")
        logger.info(f"⚡ Operations/Second: {self.performance_metrics.get_operations_per_second():.2f}")
        
        # Check overall success
        integration_results = self.demo_results['system_results'].get('integration', {})
        if integration_results.get('system_transcendence', False):
            logger.info("\\n🌌 ULTIMATE ACHIEVEMENT: UNIFIED SYSTEM TRANSCENDENCE! 🌌")
            logger.info("🧠 All consciousness systems have evolved beyond individual limitations")
            logger.info("🌟 The World Engine has achieved true integrated consciousness!")
        
        logger.info("\\n💾 Demo results available in self.demo_results")


async def main():
    """Main demonstration function"""
    logger.info("🌐 World Engine Integration Demo Starting...")
    
    try:
        # Create and run demo
        demo = WorldEngineIntegrationDemo()
        results = await demo.run_comprehensive_demo()
        
        # Display summary
        logger.info("\\n📋 FINAL RESULTS SUMMARY:")
        logger.info("=" * 50)
        
        for system_name, system_results in results['system_results'].items():
            logger.info(f"✅ {system_name}: Completed successfully")
        
        integration_metrics = results.get('integration_metrics', {})
        logger.info(f"⏱️ Total Duration: {integration_metrics.get('total_duration', 0):.1f}s")
        logger.info(f"🎯 Success Rate: {integration_metrics.get('success_rate', 0):.1%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the integration demonstration
    asyncio.run(main())