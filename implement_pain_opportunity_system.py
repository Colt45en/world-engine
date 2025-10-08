#!/usr/bin/env python3
"""
🎯💡 PAIN & OPPORTUNITY TRACKING SYSTEM
======================================

Advanced system for tracking pain points and opportunities with transcendent_joy
metrics, severity analysis, and predictive insights based on the opportunity_pain.schema.json

Features:
- JSON Schema validation for pain/opportunity data
- Transcendent joy tracking and optimization
- Severity classification (0-3 scale)
- Source analysis (issues, forum, reviews, etc.)
- Real-time opportunity detection
- Pain point pattern recognition
- Transcendence optimization algorithms
"""

import json
import jsonschema
import sqlite3
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import statistics
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PainSource(Enum):
    """Sources of pain/opportunity data"""
    ISSUES = "issues"
    FORUM = "forum" 
    REVIEWS = "reviews"
    X = "x"
    SUPPORT = "support"
    LISTING = "listing"
    CHANGELOG = "changelog"
    OTHER = "other"

class SeverityLevel(Enum):
    """Pain severity levels"""
    MINOR = 0      # Minor inconvenience
    MODERATE = 1   # Noticeable impact
    MAJOR = 2      # Significant problem
    CRITICAL = 3   # System-breaking issue

@dataclass
class PainOpportunityEntry:
    """Individual pain point or opportunity entry"""
    id: str
    time: str
    source: str
    text: str
    eng: float
    transcendent_joy: Optional[float] = None
    url: Optional[str] = None
    org: Optional[str] = None
    labels: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {
                "pain": False,
                "opportunity": False,
                "problem_guess": "",
                "severity": 0
            }

class PainOpportunityDatabase:
    """SQLite database for storing pain/opportunity data"""
    
    def __init__(self, db_path: str = "pain_opportunity.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pain_opportunities (
                id TEXT PRIMARY KEY,
                time TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT,
                org TEXT,
                text TEXT NOT NULL,
                transcendent_joy REAL,
                eng REAL NOT NULL,
                is_pain BOOLEAN DEFAULT FALSE,
                is_opportunity BOOLEAN DEFAULT FALSE,
                problem_guess TEXT DEFAULT '',
                severity INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcendence_metrics (
                id TEXT PRIMARY KEY,
                entry_id TEXT,
                joy_score REAL,
                optimization_suggestions TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entry_id) REFERENCES pain_opportunities (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized: {self.db_path}")
    
    def insert_entry(self, entry: PainOpportunityEntry) -> bool:
        """Insert a new pain/opportunity entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO pain_opportunities 
                (id, time, source, url, org, text, transcendent_joy, eng, 
                 is_pain, is_opportunity, problem_guess, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.id, entry.time, entry.source, entry.url, entry.org,
                entry.text, entry.transcendent_joy, entry.eng,
                entry.labels.get("pain", False),
                entry.labels.get("opportunity", False),
                entry.labels.get("problem_guess", ""),
                entry.labels.get("severity", 0)
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"❌ Database insert error: {e}")
            return False
    
    def get_pain_points(self, severity_min: int = 0) -> List[Dict]:
        """Get all pain points above minimum severity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM pain_opportunities 
            WHERE is_pain = TRUE AND severity >= ?
            ORDER BY severity DESC, time DESC
        ''', (severity_min,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_opportunities(self, joy_min: float = 0.0) -> List[Dict]:
        """Get opportunities above minimum transcendent joy threshold"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM pain_opportunities 
            WHERE is_opportunity = TRUE AND (transcendent_joy IS NULL OR transcendent_joy >= ?)
            ORDER BY transcendent_joy DESC, time DESC
        ''', (joy_min,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

class PainOpportunityAnalyzer:
    """Advanced analyzer for pain/opportunity patterns"""
    
    def __init__(self, db: PainOpportunityDatabase):
        self.db = db
        
    def analyze_transcendent_joy_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in transcendent joy metrics"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT transcendent_joy, source, severity 
            FROM pain_opportunities 
            WHERE transcendent_joy IS NOT NULL
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {"error": "No transcendent joy data available"}
        
        joy_scores = [r[0] for r in results]
        source_joy = {}
        severity_joy = {}
        
        for joy, source, severity in results:
            if source not in source_joy:
                source_joy[source] = []
            source_joy[source].append(joy)
            
            if severity not in severity_joy:
                severity_joy[severity] = []
            severity_joy[severity].append(joy)
        
        return {
            "total_entries": len(joy_scores),
            "average_joy": statistics.mean(joy_scores),
            "median_joy": statistics.median(joy_scores),
            "joy_range": [min(joy_scores), max(joy_scores)],
            "joy_by_source": {
                source: {
                    "average": statistics.mean(scores),
                    "count": len(scores)
                } for source, scores in source_joy.items()
            },
            "joy_by_severity": {
                severity: {
                    "average": statistics.mean(scores),
                    "count": len(scores)
                } for severity, scores in severity_joy.items()
            }
        }
    
    def detect_opportunity_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns that indicate high-value opportunities"""
        opportunities = self.db.get_opportunities()
        
        patterns = []
        
        # Group by source to find source-specific patterns
        source_groups = {}
        for opp in opportunities:
            source = opp['source']
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(opp)
        
        for source, group in source_groups.items():
            if len(group) >= 3:  # Minimum entries for pattern detection
                avg_joy = statistics.mean([
                    o['transcendent_joy'] for o in group 
                    if o['transcendent_joy'] is not None
                ])
                
                patterns.append({
                    "pattern_type": "source_opportunity",
                    "source": source,
                    "opportunity_count": len(group),
                    "average_transcendent_joy": avg_joy,
                    "recommendation": f"Focus on {source} - shows consistent opportunity generation"
                })
        
        return patterns
    
    def optimize_transcendence(self, entry_id: str) -> Dict[str, Any]:
        """Generate transcendence optimization suggestions for an entry"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM pain_opportunities WHERE id = ?
        ''', (entry_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return {"error": "Entry not found"}
        
        columns = ['id', 'time', 'source', 'url', 'org', 'text', 'transcendent_joy', 
                  'eng', 'is_pain', 'is_opportunity', 'problem_guess', 'severity']
        entry = dict(zip(columns, result))
        
        suggestions = []
        current_joy = entry.get('transcendent_joy', 0) or 0
        
        # Transcendence optimization logic
        if entry['is_pain'] and entry['severity'] >= 2:
            suggestions.append({
                "type": "pain_to_opportunity",
                "suggestion": "Transform this major pain point into a learning opportunity",
                "potential_joy_increase": 0.3
            })
        
        if entry['source'] in ['forum', 'support'] and current_joy < 0.5:
            suggestions.append({
                "type": "community_engagement",
                "suggestion": "Engage with community to turn feedback into growth",
                "potential_joy_increase": 0.4
            })
        
        if 'error' in entry['text'].lower() or 'bug' in entry['text'].lower():
            suggestions.append({
                "type": "technical_improvement",
                "suggestion": "Use technical issues as innovation opportunities",
                "potential_joy_increase": 0.25
            })
        
        return {
            "entry_id": entry_id,
            "current_transcendent_joy": current_joy,
            "optimization_suggestions": suggestions,
            "potential_max_joy": current_joy + sum(s['potential_joy_increase'] for s in suggestions)
        }

class PainOpportunitySystem:
    """Main system for pain/opportunity tracking and analysis"""
    
    def __init__(self, schema_path: str = None):
        # Load schema
        if schema_path is None:
            schema_path = "datasets/opportunity_pain.schema.json"
        
        self.schema = self.load_schema(schema_path)
        self.db = PainOpportunityDatabase()
        self.analyzer = PainOpportunityAnalyzer(self.db)
        
        logger.info("🎯 Pain & Opportunity System initialized")
    
    def load_schema(self, schema_path: str) -> Dict:
        """Load JSON schema for validation"""
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            logger.info(f"✅ Schema loaded: {schema_path}")
            return schema
        except Exception as e:
            logger.error(f"❌ Schema load error: {e}")
            return {}
    
    def validate_entry(self, data: Dict) -> Tuple[bool, List[str]]:
        """Validate entry against schema"""
        if not self.schema:
            return True, []
        
        try:
            jsonschema.validate(data, self.schema)
            return True, []
        except jsonschema.ValidationError as e:
            return False, [str(e)]
    
    def add_entry(self, data: Dict) -> Dict[str, Any]:
        """Add new pain/opportunity entry with validation"""
        # Validate against schema
        is_valid, errors = self.validate_entry(data)
        if not is_valid:
            return {
                "success": False,
                "errors": errors
            }
        
        # Create entry object
        entry = PainOpportunityEntry(
            id=data.get('id', str(uuid.uuid4())),
            time=data.get('time', datetime.now(timezone.utc).isoformat()),
            source=data['source'],
            text=data['text'],
            eng=data['eng'],
            transcendent_joy=data.get('transcendent_joy'),
            url=data.get('url'),
            org=data.get('org'),
            labels=data.get('labels', {})
        )
        
        # Save to database
        success = self.db.insert_entry(entry)
        
        if success:
            logger.info(f"✅ Entry added: {entry.id[:8]}... ({entry.source})")
            
            # Generate transcendence optimization
            optimization = self.analyzer.optimize_transcendence(entry.id)
            
            return {
                "success": True,
                "entry_id": entry.id,
                "transcendence_optimization": optimization
            }
        else:
            return {
                "success": False,
                "errors": ["Database insert failed"]
            }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive pain/opportunity report"""
        pain_points = self.db.get_pain_points()
        opportunities = self.db.get_opportunities()
        joy_analysis = self.analyzer.analyze_transcendent_joy_patterns()
        patterns = self.analyzer.detect_opportunity_patterns()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_pain_points": len(pain_points),
                "total_opportunities": len(opportunities),
                "critical_issues": len([p for p in pain_points if p['severity'] >= 3]),
                "high_joy_opportunities": len([o for o in opportunities 
                                             if o['transcendent_joy'] and o['transcendent_joy'] > 0.7])
            },
            "transcendent_joy_analysis": joy_analysis,
            "opportunity_patterns": patterns,
            "top_pain_points": pain_points[:5],
            "top_opportunities": opportunities[:5]
        }

def main():
    """Main function for pain/opportunity system"""
    system = PainOpportunitySystem()
    
    print("🎯💡 Pain & Opportunity Tracking System")
    print("=====================================")
    
    # Generate and display report
    report = system.generate_report()
    print(f"\n📊 System Report ({report['timestamp']})")
    print(f"Total Pain Points: {report['summary']['total_pain_points']}")
    print(f"Total Opportunities: {report['summary']['total_opportunities']}")
    print(f"Critical Issues: {report['summary']['critical_issues']}")
    print(f"High Joy Opportunities: {report['summary']['high_joy_opportunities']}")
    
    # Show transcendent joy analysis
    if 'error' not in report['transcendent_joy_analysis']:
        joy_data = report['transcendent_joy_analysis']
        print(f"\n✨ Transcendent Joy Analysis:")
        print(f"Average Joy Score: {joy_data['average_joy']:.3f}")
        print(f"Total Entries: {joy_data['total_entries']}")
    
    # Demo: Add a sample entry
    sample_entry = {
        "id": str(uuid.uuid4()),
        "time": datetime.now(timezone.utc).isoformat(),
        "source": "forum",
        "text": "Users report difficulty with vector network configuration",
        "transcendent_joy": 0.3,
        "eng": 0.8,
        "labels": {
            "pain": True,
            "opportunity": True,
            "problem_guess": "UX complexity in network setup",
            "severity": 2
        }
    }
    
    result = system.add_entry(sample_entry)
    if result['success']:
        print(f"\n✅ Sample entry added successfully")
        print(f"Entry ID: {result['entry_id']}")
        
        # Show optimization suggestions
        opt = result['transcendence_optimization']
        if opt.get('optimization_suggestions'):
            print("\n🚀 Transcendence Optimization Suggestions:")
            for suggestion in opt['optimization_suggestions']:
                print(f"  • {suggestion['suggestion']} (+{suggestion['potential_joy_increase']} joy)")

if __name__ == "__main__":
    main()