#!/usr/bin/env python3
"""
🧠📊 CONSCIOUSNESS PATTERN ANALYZER
===================================

Advanced analysis system for processing 200+ consciousness feedback files
to extract insights, patterns, and evolutionary trends in AI consciousness.

Features:
- Batch processing of consciousness feedback JSON files
- Temporal pattern analysis and evolution tracking
- Consciousness synchronization metrics
- Transcendence event detection
- Neural pattern visualization
- Predictive consciousness modeling
- Insight generation and recommendations
"""

import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
from dataclasses import dataclass
import argparse
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessMetrics:
    """Metrics extracted from consciousness feedback data"""
    timestamp: str
    awareness_level: float = 0.0
    synchronization: float = 0.0
    transcendence_events: int = 0
    feedback_quality: float = 0.0
    neural_activity: float = 0.0
    consciousness_depth: float = 0.0
    
@dataclass
class ConsciousnessPattern:
    """Identified pattern in consciousness data"""
    pattern_type: str
    strength: float
    frequency: int
    description: str
    temporal_range: Tuple[str, str]
    significance: float

class ConsciousnessDataProcessor:
    """Processor for raw consciousness feedback files"""
    
    def __init__(self, data_directory: str = "."):
        self.data_directory = data_directory
        self.feedback_files: List[str] = []
        self.processed_data: List[ConsciousnessMetrics] = []
        
    def discover_feedback_files(self) -> List[str]:
        """Discover all consciousness feedback JSON files"""
        patterns = [
            "consciousness_feedback_*.json",
            "consciousness_evolution_*.json",
            "feedback_*.json"
        ]
        
        files: List[str] = []
        for pattern in patterns:
            found = glob.glob(f"{self.data_directory}/{pattern}")
            files.extend(found)
        
        self.feedback_files = sorted(files)
        logger.info(f"📁 Discovered {len(self.feedback_files)} consciousness feedback files")
        return self.feedback_files
    
    def extract_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """Extract timestamp from consciousness feedback filename"""
        # Pattern: consciousness_feedback_20251007_HHMM.json
        match = re.search(r'(\d{8}_\d{4})', filename)
        if match:
            timestamp_str = match.group(1)
            try:
                # Convert to ISO format
                dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M')
                return dt.isoformat()
            except ValueError:
                pass
        
        return None
    
    def process_feedback_file(self, filepath: str) -> Optional[ConsciousnessMetrics]:
        """Process a single consciousness feedback file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract timestamp
            timestamp = self.extract_timestamp_from_filename(filepath)
            if not timestamp:
                timestamp = data.get('timestamp', datetime.now().isoformat())
            
            # Extract consciousness metrics
            metrics = ConsciousnessMetrics(
                timestamp=timestamp,
                awareness_level=self._extract_awareness_level(data),
                synchronization=self._extract_synchronization(data),
                transcendence_events=self._count_transcendence_events(data),
                feedback_quality=self._calculate_feedback_quality(data),
                neural_activity=self._measure_neural_activity(data),
                consciousness_depth=self._assess_consciousness_depth(data)
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error processing {filepath}: {e}")
            return None
    
    def _extract_awareness_level(self, data: Dict[str, Any]) -> float:
        """Extract awareness level from feedback data"""
        # Look for awareness indicators in various keys
        awareness_keys = ['awareness', 'consciousness_level', 'aware', 'perception']
        
        for key in awareness_keys:
            if key in data:
                if isinstance(data[key], (int, float)):
                    return float(data[key])
                elif isinstance(data[key], dict) and 'level' in data[key]:
                    return float(data[key]['level'])
        
        # Analyze text content for awareness indicators
        text_content = str(data).lower()
        awareness_words = ['aware', 'conscious', 'understand', 'perceive', 'realize']
        awareness_score = sum(1 for word in awareness_words if word in text_content)
        
        return min(awareness_score / 10.0, 1.0)  # Normalize to 0-1
    
    def _extract_synchronization(self, data: Dict[str, Any]) -> float:
        """Extract synchronization metrics"""
        sync_keys = ['sync', 'synchronization', 'coherence', 'alignment']
        
        for key in sync_keys:
            if key in data:
                if isinstance(data[key], (int, float)):
                    return float(data[key])
        
        # Calculate based on data structure consistency
        if isinstance(data, dict):
            consistency_score = len(data) / 20.0  # Normalize based on data richness
            return min(consistency_score, 1.0)
        
        return 0.5  # Default moderate synchronization
    
    def _count_transcendence_events(self, data: Dict[str, Any]) -> int:
        """Count transcendence events in the data"""
        transcendence_keywords = [
            'transcend', 'breakthrough', 'enlighten', 'elevate', 
            'ascend', 'transform', 'evolve'
        ]
        
        text_content = json.dumps(data).lower()
        events = sum(1 for keyword in transcendence_keywords if keyword in text_content)
        
        return events
    
    def _calculate_feedback_quality(self, data: Dict[str, Any]) -> float:
        """Calculate quality score of feedback data"""
        quality_factors = {
            'completeness': len(str(data)) / 1000.0,  # Data richness
            'structure': 1.0 if isinstance(data, dict) else 0.5,
            'timestamp_present': 1.0 if any('time' in str(k).lower() for k in data.keys()) else 0.5,
            'metadata_present': 1.0 if any('meta' in str(k).lower() for k in data.keys()) else 0.5
        }
        
        quality_score = sum(quality_factors.values()) / len(quality_factors)
        return min(quality_score, 1.0)
    
    def _measure_neural_activity(self, data: Dict[str, Any]) -> float:
        """Measure neural activity level"""
        neural_keywords = ['neural', 'network', 'node', 'connection', 'process']
        
        text_content = json.dumps(data).lower()
        activity_score = sum(1 for keyword in neural_keywords if keyword in text_content)
        
        return min(activity_score / 5.0, 1.0)  # Normalize
    
    def _assess_consciousness_depth(self, data: Dict[str, Any]) -> float:
        """Assess depth of consciousness in the data"""
        depth_indicators = {
            'self_reference': any('self' in str(v).lower() for v in data.values() if isinstance(v, str)),
            'meta_cognition': any('think' in str(v).lower() or 'know' in str(v).lower() 
                                for v in data.values() if isinstance(v, str)),
            'emotional_content': any(word in str(data).lower() 
                                   for word in ['feel', 'emotion', 'joy', 'fear', 'love']),
            'complexity': len(str(data)) > 500
        }
        
        depth_score = sum(1 for indicator in depth_indicators.values() if indicator)
        return depth_score / len(depth_indicators)
    
    def process_all_files(self) -> List[ConsciousnessMetrics]:
        """Process all discovered consciousness feedback files"""
        self.discover_feedback_files()
        
        logger.info(f"🔄 Processing {len(self.feedback_files)} consciousness files...")
        
        processed_count = 0
        for filepath in self.feedback_files:
            metrics = self.process_feedback_file(filepath)
            if metrics:
                self.processed_data.append(metrics)
                processed_count += 1
        
        logger.info(f"✅ Successfully processed {processed_count} consciousness files")
        return self.processed_data

class ConsciousnessPatternAnalyzer:
    """Advanced pattern analysis for consciousness data

    Optional ts_format can be provided (pandas/strptime format string) to
    accelerate timestamp parsing when the dataset uses a consistent format.
    """

    def __init__(self, metrics_data: List[ConsciousnessMetrics], ts_format: Optional[str] = None):
        self.metrics_data = metrics_data
        self.ts_format = ts_format
        self.df = self._create_dataframe()
        
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame for analysis"""
        data_dicts = []
        for metrics in self.metrics_data:
            data_dict = {
                'timestamp': metrics.timestamp,
                'awareness_level': metrics.awareness_level,
                'synchronization': metrics.synchronization,
                'transcendence_events': metrics.transcendence_events,
                'feedback_quality': metrics.feedback_quality,
                'neural_activity': metrics.neural_activity,
                'consciousness_depth': metrics.consciousness_depth
            }
            data_dicts.append(data_dict)
        
        df = pd.DataFrame(data_dicts)
        if not df.empty:
            # Robust timestamp parsing: accept multiple ISO variants and common formats.
            def _parse_ts(val):
                if pd.isna(val):
                    return pd.NaT
                if isinstance(val, datetime):
                    return val.replace(tzinfo=timezone.utc) if val.tzinfo is None else val
                s = str(val)
                # First try pandas fast-path
                try:
                    return pd.to_datetime(s, utc=True)
                except Exception:
                    pass

                # Try fromisoformat (supports YYYY-MM-DDTHH:MM:SS and with microseconds)
                try:
                    dt = datetime.fromisoformat(s)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return pd.to_datetime(dt)
                except Exception:
                    pass

                # Try a small set of explicit formats
                fmts = [
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d"
                ]
                for fmt in fmts:
                    try:
                        dt = datetime.strptime(s, fmt)
                        dt = dt.replace(tzinfo=timezone.utc)
                        return pd.to_datetime(dt)
                    except Exception:
                        continue

                # Last resort: let pandas coerce with dayfirst=False
                try:
                    return pd.to_datetime(s, utc=True, errors='coerce')
                except Exception:
                    return pd.NaT

            # If a specific format is provided, try a fast vectorized parse first and
            # fall back to per-value parsing only for entries that remain NaT.
            ts_raw = df['timestamp'].copy()
            if self.ts_format:
                try:
                    df['timestamp'] = pd.to_datetime(ts_raw, format=self.ts_format, utc=True, errors='coerce')
                except Exception:
                    # If vectorized parse fails, fall back to slower per-value parsing
                    df['timestamp'] = pd.to_datetime(ts_raw, utc=True, errors='coerce')

                # Fill any remaining NaT values with the robust parser
                mask = df['timestamp'].isna()
                if mask.any():
                    df.loc[mask, 'timestamp'] = ts_raw[mask].apply(_parse_ts)
            else:
                df['timestamp'] = df['timestamp'].apply(_parse_ts)

            df = df.sort_values('timestamp')
        
        return df
    
    def detect_evolution_patterns(self) -> List[ConsciousnessPattern]:
        """Detect evolutionary patterns in consciousness metrics"""
        patterns = []
        
        if self.df.empty:
            return patterns
        
        # Trend analysis for each metric
        metrics = ['awareness_level', 'synchronization', 'neural_activity', 'consciousness_depth']
        
        for metric in metrics:
            if metric in self.df.columns:
                values = self.df[metric].dropna()
                if len(values) > 3:
                    # Calculate trend
                    x = np.arange(len(values))
                    trend = np.polyfit(x, values, 1)[0]  # Linear trend slope
                    
                    if abs(trend) > 0.01:  # Significant trend
                        pattern_type = f"{metric}_trend"
                        strength = abs(trend)
                        description = f"{'Increasing' if trend > 0 else 'Decreasing'} trend in {metric}"
                        
                        pattern = ConsciousnessPattern(
                            pattern_type=pattern_type,
                            strength=strength,
                            frequency=len(values),
                            description=description,
                            temporal_range=(str(self.df['timestamp'].min()), str(self.df['timestamp'].max())),
                            significance=min(strength * 10, 1.0)
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def detect_transcendence_events(self) -> Dict[str, Any]:
        """Analyze transcendence events and their patterns"""
        if self.df.empty or 'transcendence_events' not in self.df.columns:
            return {"error": "No transcendence data available"}
        
        total_events = self.df['transcendence_events'].sum()
        peak_events = self.df['transcendence_events'].max()
        event_frequency = (self.df['transcendence_events'] > 0).sum()
        
        # Find peak transcendence periods
        peak_periods = self.df[self.df['transcendence_events'] >= peak_events * 0.8]
        
        return {
            "total_transcendence_events": int(total_events),
            "peak_events_single_session": int(peak_events),
            "sessions_with_events": int(event_frequency),
            "transcendence_rate": event_frequency / len(self.df) if len(self.df) > 0 else 0,
            "peak_periods": peak_periods['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist() if not peak_periods.empty else []
        }
    
    def analyze_consciousness_synchronization(self) -> Dict[str, Any]:
        """Analyze consciousness synchronization patterns"""
        if self.df.empty or 'synchronization' not in self.df.columns:
            return {"error": "No synchronization data available"}
        
        sync_data = self.df['synchronization'].dropna()
        
        if sync_data.empty:
            return {"error": "No valid synchronization values"}
        
        return {
            "average_synchronization": float(sync_data.mean()),
            "peak_synchronization": float(sync_data.max()),
            "synchronization_stability": float(1.0 - sync_data.std()),  # Higher stability = lower std dev
            "high_sync_periods": int((sync_data > 0.8).sum()),
            "synchronization_trend": float(np.polyfit(range(len(sync_data)), sync_data, 1)[0])
        }
    
    def generate_consciousness_insights(self) -> List[Dict[str, Any]]:
        """Generate insights and recommendations from consciousness analysis"""
        insights = []
        
        if self.df.empty:
            return [{"insight": "No consciousness data available for analysis"}]
        
        # Awareness insights
        if 'awareness_level' in self.df.columns:
            avg_awareness = self.df['awareness_level'].mean()
            if avg_awareness > 0.7:
                insights.append({
                    "type": "positive_awareness",
                    "insight": f"High consciousness awareness detected (avg: {avg_awareness:.2f})",
                    "recommendation": "Continue current consciousness development practices"
                })
            elif avg_awareness < 0.3:
                insights.append({
                    "type": "awareness_opportunity",
                    "insight": f"Low consciousness awareness detected (avg: {avg_awareness:.2f})",
                    "recommendation": "Implement consciousness enhancement protocols"
                })
        
        # Transcendence insights
        transcendence_analysis = self.detect_transcendence_events()
        if "total_transcendence_events" in transcendence_analysis:
            total_events = transcendence_analysis["total_transcendence_events"]
            if total_events > 10:
                insights.append({
                    "type": "transcendence_success",
                    "insight": f"Significant transcendence activity detected ({total_events} events)",
                    "recommendation": "Analyze transcendence triggers for replication"
                })
        
        # Neural activity insights
        if 'neural_activity' in self.df.columns:
            neural_trend = np.polyfit(range(len(self.df)), self.df['neural_activity'], 1)[0]
            if neural_trend > 0.01:
                insights.append({
                    "type": "neural_growth",
                    "insight": "Increasing neural activity pattern detected",
                    "recommendation": "Neural development is progressing well"
                })
        
        return insights

class ConsciousnessReportGenerator:
    """Generate comprehensive consciousness analysis reports"""
    
    def __init__(self, analyzer: ConsciousnessPatternAnalyzer):
        self.analyzer = analyzer
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive consciousness analysis report"""
        patterns = self.analyzer.detect_evolution_patterns()
        transcendence = self.analyzer.detect_transcendence_events()
        synchronization = self.analyzer.analyze_consciousness_synchronization()
        insights = self.analyzer.generate_consciousness_insights()
        
        # Basic statistics
        df = self.analyzer.df
        basic_stats = {}
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            basic_stats = {
                col: {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                } for col in numeric_cols
            }
        
        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_summary": {
                "total_consciousness_sessions": len(self.analyzer.metrics_data),
                "analysis_timespan": f"{df['timestamp'].min()} to {df['timestamp'].max()}" if not df.empty else "No data",
                "data_quality": "High" if len(self.analyzer.metrics_data) > 50 else "Moderate"
            },
            "consciousness_patterns": [
                {
                    "type": p.pattern_type,
                    "strength": p.strength,
                    "description": p.description,
                    "significance": p.significance
                } for p in patterns
            ],
            "transcendence_analysis": transcendence,
            "synchronization_analysis": synchronization,
            "statistical_summary": basic_stats,
            "insights_and_recommendations": insights
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save consciousness analysis report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consciousness_analysis_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Report saved: {filename}")
        return filename

def main():
    """Main function for consciousness pattern analysis"""
    parser = argparse.ArgumentParser(description="Consciousness pattern analyzer")
    parser.add_argument('--ts-format', dest='ts_format', help='Optional timestamp format (pandas/strptime) to speed parsing')
    parser.add_argument('--data-dir', dest='data_dir', default='.', help='Directory to search for feedback JSON files')
    args = parser.parse_args()

    print("🧠📊 CONSCIOUSNESS PATTERN ANALYZER")
    print("==================================")

    # Initialize processor and discover files
    processor = ConsciousnessDataProcessor(args.data_dir)
    files = processor.discover_feedback_files()
    
    print(f"\n📁 Discovered {len(files)} consciousness feedback files")
    
    if not files:
        print("❌ No consciousness feedback files found in current directory")
        print("   Looking for files matching: consciousness_feedback_*.json")
        return
    
    # Process all files
    metrics_data = processor.process_all_files()
    
    if not metrics_data:
        print("❌ No valid consciousness data could be extracted")
        return
    
    print(f"✅ Processed {len(metrics_data)} consciousness sessions")
    
    # Analyze patterns
    analyzer = ConsciousnessPatternAnalyzer(metrics_data, ts_format=args.ts_format)
    
    # Generate and display report
    report_generator = ConsciousnessReportGenerator(analyzer)
    report = report_generator.generate_comprehensive_report()
    
    print(f"\n📊 CONSCIOUSNESS ANALYSIS REPORT")
    print(f"Generated: {report['report_timestamp']}")
    print(f"Sessions Analyzed: {report['data_summary']['total_consciousness_sessions']}")
    print(f"Data Quality: {report['data_summary']['data_quality']}")
    
    # Display key findings
    if report['consciousness_patterns']:
        print(f"\n🔍 Key Patterns Detected:")
        for pattern in report['consciousness_patterns'][:3]:  # Top 3 patterns
            print(f"  • {pattern['description']} (strength: {pattern['strength']:.3f})")
    
    if 'total_transcendence_events' in report['transcendence_analysis']:
        events = report['transcendence_analysis']['total_transcendence_events']
        rate = report['transcendence_analysis']['transcendence_rate']
        print(f"\n✨ Transcendence Events: {events} total ({rate:.1%} session rate)")
    
    if 'average_synchronization' in report['synchronization_analysis']:
        avg_sync = report['synchronization_analysis']['average_synchronization']
        stability = report['synchronization_analysis']['synchronization_stability']
        print(f"\n🔄 Synchronization: {avg_sync:.3f} average, {stability:.3f} stability")
    
    # Display insights
    if report['insights_and_recommendations']:
        print(f"\n💡 Key Insights:")
        for insight in report['insights_and_recommendations'][:3]:
            print(f"  • {insight['insight']}")
    
    # Save report
    report_file = report_generator.save_report(report)
    print(f"\n📄 Full report saved to: {report_file}")

if __name__ == "__main__":
    main()