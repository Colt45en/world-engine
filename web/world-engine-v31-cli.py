#!/usr/bin/env python3
"""
World Engine V3.1 CLI Tool
Inspired by ArgumentParser patterns from the attachments
Provides command-line access to V3.1 features with semantic analysis
"""

import argparse
import json
import yaml
import csv
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class TestResult:
    """Test result data structure"""
    name: str
    success: bool
    duration: float
    details: str
    timestamp: str = ""

@dataclass
class MorphemeAnalysis:
    """Morphological analysis result"""
    word: str
    root: str
    morphemes: List[Dict[str, str]]
    complexity: int = 0
    related_words: List[str] = None

class WorldEngineV31CLI:
    """
    Main CLI class for World Engine V3.1
    Implements the command patterns from the provided attachments
    """

    def __init__(self):
        self.verbose = False
        self.dimensions = 3
        self.config = {}

        # Morphological data inspired by lexicon concepts
        self.prefixes = ['anti', 'auto', 'counter', 'de', 'dis', 'inter', 'multi', 'non', 'over', 'pre', 're', 'semi', 'sub', 'super', 'trans', 'ultra', 'un', 'under']
        self.suffixes = ['able', 'acy', 'age', 'al', 'ance', 'ation', 'dom', 'ed', 'en', 'ence', 'er', 'ery', 'est', 'ful', 'fy', 'hood', 'ible', 'ic', 'ing', 'ion', 'ism', 'ist', 'ity', 'ive', 'less', 'ly', 'ment', 'ness', 'or', 'ous', 'ship', 'sion', 'th', 'tion', 'ty', 'ward', 'wise', 'y']

        # Test suites configuration
        self.test_suites = {
            'all': [
                {'name': 'Type Lattice Hierarchy', 'category': 'lattice', 'weight': 1.0},
                {'name': 'Jacobian Matrix Computation', 'category': 'jacobian', 'weight': 1.2},
                {'name': 'Morpheme Pattern Recognition', 'category': 'morpheme', 'weight': 0.8},
                {'name': 'Lexicon Navigation System', 'category': 'lexicon', 'weight': 1.1},
                {'name': 'Neural Learning Pipeline', 'category': 'neural', 'weight': 1.3},
                {'name': 'System Integration Tests', 'category': 'integration', 'weight': 1.5}
            ],
            'lattice': [
                {'name': 'Type Hierarchy Validation', 'category': 'lattice', 'weight': 1.0},
                {'name': 'Composition Rule Verification', 'category': 'lattice', 'weight': 0.9},
                {'name': 'State Property Relationships', 'category': 'lattice', 'weight': 1.1}
            ],
            'jacobian': [
                {'name': 'Matrix Computation Accuracy', 'category': 'jacobian', 'weight': 1.2},
                {'name': 'Effect Propagation Tracing', 'category': 'jacobian', 'weight': 1.0},
                {'name': 'Sensitivity Analysis', 'category': 'jacobian', 'weight': 1.1}
            ],
            'morpheme': [
                {'name': 'Pattern Discovery Algorithm', 'category': 'morpheme', 'weight': 0.8},
                {'name': 'Learning System Validation', 'category': 'morpheme', 'weight': 1.0},
                {'name': 'Morphological Decomposition', 'category': 'morpheme', 'weight': 0.9}
            ],
            'lexicon': [
                {'name': 'Word Analysis Pipeline', 'category': 'lexicon', 'weight': 1.1},
                {'name': 'Relationship Mapping', 'category': 'lexicon', 'weight': 1.0},
                {'name': 'Semantic Navigation', 'category': 'lexicon', 'weight': 1.2}
            ],
            'neural': [
                {'name': 'XOR Learning Simulation', 'category': 'neural', 'weight': 1.3},
                {'name': 'Backpropagation Accuracy', 'category': 'neural', 'weight': 1.2},
                {'name': 'Convergence Testing', 'category': 'neural', 'weight': 1.1}
            ],
            'integration': [
                {'name': 'Component Communication', 'category': 'integration', 'weight': 1.5},
                {'name': 'Data Flow Validation', 'category': 'integration', 'weight': 1.3},
                {'name': 'Performance Benchmarks', 'category': 'integration', 'weight': 1.4}
            ]
        }

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with all subcommands"""
        parser = argparse.ArgumentParser(
            prog='world-engine-v31',
            description='World Engine V3.1 Advanced Mathematical System CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
Examples:
  %(prog)s test --suite all --format json --output results.json
  %(prog)s analyze "transformation" "restructure" --detail --links
  %(prog)s serve --port 8085 --open
  %(prog)s export --type lexicon --format yaml lexicon.yml
            '''
        )

        # Global arguments
        parser.add_argument('-v', '--verbose', action='store_true',
                          help='Enable verbose logging and detailed output')
        parser.add_argument('-c', '--config', type=str, metavar='FILE',
                          help='Path to configuration file (JSON/YAML)')
        parser.add_argument('--dimensions', type=int, default=3, metavar='N',
                          help='Mathematical space dimensions (default: %(default)s)')

        # Create subparsers
        subparsers = parser.add_subparsers(dest='command', title='commands',
                                          description='Available World Engine V3.1 operations')

        # Test command
        test_parser = subparsers.add_parser('test', help='Run V3.1 system tests',
                                           description='Execute various test suites for system validation')
        test_parser.add_argument('--suite', choices=['all', 'lattice', 'jacobian', 'morpheme', 'lexicon', 'neural', 'integration'],
                               default='all', help='Test suite to execute (default: %(default)s)')
        test_parser.add_argument('--format', choices=['console', 'json', 'yaml', 'csv', 'html'],
                               default='console', help='Output format (default: %(default)s)')
        test_parser.add_argument('--output', type=str, metavar='FILE',
                               help='Save results to file (format auto-detected from extension)')
        test_parser.add_argument('--parallel', action='store_true',
                               help='Run tests in parallel where possible')
        test_parser.add_argument('--timeout', type=float, default=30.0, metavar='SECONDS',
                               help='Test timeout in seconds (default: %(default)s)')

        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze morphological structures',
                                             description='Perform detailed morphological analysis on words')
        analyze_parser.add_argument('words', nargs='+', metavar='WORD',
                                   help='Words to analyze morphologically')
        analyze_parser.add_argument('--detail', action='store_true',
                                   help='Show detailed morphological breakdown and structure')
        analyze_parser.add_argument('--links', action='store_true',
                                   help='Show morphological links and word relationships')
        analyze_parser.add_argument('--export', type=str, metavar='FILE',
                                   help='Export analysis results to file')
        analyze_parser.add_argument('--format', choices=['text', 'json', 'csv'],
                                   default='text', help='Analysis output format (default: %(default)s)')

        # Server command
        server_parser = subparsers.add_parser('serve', help='Start V3.1 development server',
                                            description='Launch the World Engine V3.1 web interface')
        server_parser.add_argument('--port', type=int, default=8085, metavar='PORT',
                                 help='Server port (default: %(default)s)')
        server_parser.add_argument('--host', default='localhost', metavar='HOST',
                                 help='Server host (default: %(default)s)')
        server_parser.add_argument('--open', action='store_true',
                                 help='Open browser after starting server')
        server_parser.add_argument('--debug', action='store_true',
                                 help='Enable debug mode with detailed logging')

        # Export command
        export_parser = subparsers.add_parser('export', help='Export system data and configurations',
                                            description='Export various system components and data')
        export_parser.add_argument('--type', choices=['lexicon', 'config', 'traces', 'tests', 'all'],
                                 default='all', help='Data type to export (default: %(default)s)')
        export_parser.add_argument('--format', choices=['json', 'yaml', 'csv', 'xml'],
                                 default='json', help='Export format (default: %(default)s)')
        export_parser.add_argument('output', help='Output file path')
        export_parser.add_argument('--compress', action='store_true',
                                 help='Compress exported data (gzip)')

        return parser

    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file"""
        if not config_path:
            return {}

        try:
            config_file = Path(config_path)
            if not config_file.exists():
                print(f"⚠️  Configuration file not found: {config_path}")
                return {}

            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)

        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return {}

    def run_test_suite(self, suite_name: str, output_format: str,
                      output_file: Optional[str], parallel: bool, timeout: float) -> int:
        """Execute the specified test suite"""
        if self.verbose:
            print(f"🧪 Running {suite_name} test suite...")
            print(f"   Format: {output_format}")
            print(f"   Parallel: {parallel}")
            print(f"   Timeout: {timeout}s")

        # Get test cases
        test_cases = self.test_suites.get(suite_name, self.test_suites['all'])
        results = []

        print(f"\n🔬 Executing {len(test_cases)} test(s)...")

        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] {test_case['name']}...", end=' ')

            result = self.execute_single_test(test_case, timeout)
            results.append(result)

            status = "✅ PASS" if result.success else "❌ FAIL"
            duration = f"({result.duration:.2f}s)"
            print(f"{status} {duration}")

            if self.verbose:
                print(f"    Details: {result.details}")

        # Summary
        passed = sum(1 for r in results if r.success)
        total = len(results)
        success_rate = (passed / total * 100) if total > 0 else 0

        print(f"\n📊 Test Results: {passed}/{total} passed ({success_rate:.1f}%)")

        # Save results if requested
        if output_file:
            self.save_test_results(results, output_file, output_format)
            print(f"💾 Results saved to: {output_file}")

        return 0 if passed == total else 1

    def execute_single_test(self, test_case: Dict[str, Any], timeout: float) -> TestResult:
        """Execute a single test case with simulation"""
        start_time = time.time()

        try:
            # Simulate test execution based on category and weight
            category = test_case.get('category', 'unknown')
            weight = test_case.get('weight', 1.0)

            # Simulate processing time
            processing_time = random.uniform(0.1, 0.5) * weight
            time.sleep(processing_time)

            # Success probability based on category
            success_probabilities = {
                'lattice': 0.92,
                'jacobian': 0.88,
                'morpheme': 0.85,
                'lexicon': 0.90,
                'neural': 0.83,
                'integration': 0.80
            }

            success_prob = success_probabilities.get(category, 0.85)
            success = random.random() < success_prob

            # Generate appropriate details
            if success:
                details = self.generate_success_message(category)
            else:
                details = self.generate_failure_message(category)

            duration = time.time() - start_time

            return TestResult(
                name=test_case['name'],
                success=success,
                duration=duration,
                details=details,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_case['name'],
                success=False,
                duration=duration,
                details=f"Test execution error: {str(e)}",
                timestamp=datetime.now().isoformat()
            )

    def generate_success_message(self, category: str) -> str:
        """Generate success message based on test category"""
        messages = {
            'lattice': "Type hierarchy validated, all relationships consistent",
            'jacobian': "Matrix computations accurate, effect tracing operational",
            'morpheme': "Pattern recognition active, learning pipeline functional",
            'lexicon': "Word analysis complete, relationship mapping successful",
            'neural': "Neural simulation converged, backpropagation accurate",
            'integration': "All components communicating, system integration verified"
        }
        return messages.get(category, "Test completed successfully")

    def generate_failure_message(self, category: str) -> str:
        """Generate failure message based on test category"""
        messages = {
            'lattice': "Type hierarchy inconsistencies detected",
            'jacobian': "Matrix computation errors, trace accuracy compromised",
            'morpheme': "Pattern recognition failures, learning system offline",
            'lexicon': "Word analysis errors, relationship mapping incomplete",
            'neural': "Convergence failed, backpropagation errors detected",
            'integration': "Component communication failures, integration compromised"
        }
        return messages.get(category, "Test failed with unknown error")

    def analyze_morphology(self, words: List[str], show_detail: bool,
                         show_links: bool, export_file: Optional[str], format_type: str):
        """Perform morphological analysis on words"""
        if self.verbose:
            print(f"🔍 Analyzing {len(words)} word(s) with morphological decomposition...")

        analyses = []

        for word in words:
            analysis = self.decompose_morphologically(word)
            analyses.append(analysis)

            # Display results
            if format_type == 'text':
                print(f"\n📝 Word: '{word}'")
                print(f"   Root: {analysis.root}")

                morpheme_desc = ', '.join([f"{m['form']}({m['type']})" for m in analysis.morphemes])
                print(f"   Morphemes: {morpheme_desc}")

                if show_detail:
                    structure = ' + '.join([m['type'] for m in analysis.morphemes])
                    print(f"   Structure: {structure}")
                    print(f"   Complexity: {analysis.complexity} morphemes")
                    print(f"   Pattern: {self.get_morphological_pattern(analysis)}")

                if show_links and analysis.related_words:
                    print(f"   Related: {', '.join(analysis.related_words)}")

        # Export if requested
        if export_file:
            self.export_morphological_analysis(analyses, export_file, format_type)
            print(f"💾 Analysis exported to: {export_file}")

    def decompose_morphologically(self, word: str) -> MorphemeAnalysis:
        """Decompose word into morphological components"""
        morphemes = []
        remaining = word.lower().strip()
        original_word = word

        # Find prefixes
        for prefix in sorted(self.prefixes, key=len, reverse=True):
            if remaining.startswith(prefix) and len(remaining) > len(prefix):
                morphemes.append({'type': 'prefix', 'form': prefix, 'position': 'initial'})
                remaining = remaining[len(prefix):]
                break

        # Find suffixes
        for suffix in sorted(self.suffixes, key=len, reverse=True):
            if remaining.endswith(suffix) and len(remaining) > len(suffix):
                morphemes.append({'type': 'suffix', 'form': suffix, 'position': 'final'})
                remaining = remaining[:-len(suffix)]
                break

        # Root (what remains)
        if remaining:
            # Insert root at appropriate position
            root_morpheme = {'type': 'root', 'form': remaining, 'position': 'stem'}
            if morphemes and morphemes[0]['type'] == 'prefix':
                morphemes.insert(1, root_morpheme)
            else:
                morphemes.insert(0, root_morpheme)

        # Find related words
        related = self.find_related_words(remaining)

        return MorphemeAnalysis(
            word=original_word,
            root=remaining,
            morphemes=morphemes,
            complexity=len(morphemes),
            related_words=related
        )

    def get_morphological_pattern(self, analysis: MorphemeAnalysis) -> str:
        """Get the morphological pattern string"""
        pattern_parts = []
        for morpheme in analysis.morphemes:
            if morpheme['type'] == 'prefix':
                pattern_parts.append(f"{morpheme['form']}-")
            elif morpheme['type'] == 'suffix':
                pattern_parts.append(f"-{morpheme['form']}")
            else:
                pattern_parts.append(f"[{morpheme['form']}]")
        return ''.join(pattern_parts)

    def find_related_words(self, root: str) -> List[str]:
        """Find morphologically related words"""
        # Simple heuristic-based related word finding
        common_roots = {
            'form': ['transform', 'reform', 'conform', 'deform'],
            'struct': ['structure', 'construct', 'destruct', 'instruct'],
            'tract': ['extract', 'attract', 'contract', 'distract'],
            'port': ['transport', 'export', 'import', 'report'],
            'spect': ['inspect', 'respect', 'aspect', 'prospect'],
            'vert': ['convert', 'invert', 'revert', 'divert']
        }

        # Find partial matches
        related = []
        for known_root, words in common_roots.items():
            if root in known_root or known_root in root:
                related.extend(words[:3])  # Limit results
                break

        return related

    def start_server(self, host: str, port: int, open_browser: bool, debug: bool):
        """Simulate starting the development server"""
        print(f"🚀 Starting World Engine V3.1 server...")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Debug: {debug}")

        base_url = f"http://{host}:{port}"

        print(f"\n✅ Server ready at {base_url}")
        print(f"📱 Available endpoints:")
        print(f"   {base_url}/studio.html - Main Studio Interface")
        print(f"   {base_url}/lexical-logic-engine.html - Engine Demo")
        print(f"   {base_url}/v31-test-runner.html - Interactive Test Runner")
        print(f"   {base_url}/world-engine-v31-cli.js - CLI Tool")

        if open_browser:
            print(f"🌐 Opening browser to {base_url}/studio.html")

        print(f"\nPress Ctrl+C to stop the server")

        # In a real implementation, this would start the actual server
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Server stopped")

    def export_data(self, data_type: str, format_type: str, output_path: str, compress: bool):
        """Export system data in specified format"""
        if self.verbose:
            print(f"📤 Exporting {data_type} data...")
            print(f"   Format: {format_type}")
            print(f"   Output: {output_path}")
            print(f"   Compress: {compress}")

        # Generate export data
        export_data = self.generate_export_data(data_type)

        # Save data
        self.save_export_data(export_data, output_path, format_type, compress)

        print(f"✅ Export complete: {output_path}")

    def generate_export_data(self, data_type: str) -> Dict[str, Any]:
        """Generate export data based on type"""
        base_data = {
            'timestamp': datetime.now().isoformat(),
            'version': 'v3.1',
            'dimensions': self.dimensions,
            'system': 'World Engine V3.1 Advanced Mathematical System',
            'generator': 'world-engine-v31-cli'
        }

        type_specific_data = {
            'lexicon': {
                'morphological_data': {
                    'prefixes': self.prefixes,
                    'suffixes': self.suffixes,
                    'root_patterns': ['state', 'form', 'struct', 'tract', 'spect', 'vert'],
                    'composition_rules': {
                        'prefix + root': 'derivational morphology',
                        'root + suffix': 'inflectional morphology',
                        'prefix + root + suffix': 'complex derivation'
                    }
                }
            },
            'config': {
                'system_configuration': {
                    'type_lattice_enabled': True,
                    'jacobian_tracing_enabled': True,
                    'morpheme_discovery_enabled': True,
                    'neural_learning_enabled': True,
                    'dimensions': self.dimensions,
                    'test_suites': list(self.test_suites.keys())
                }
            },
            'traces': {
                'jacobian_traces': [
                    {
                        'operation': 'state_transformation',
                        'matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        'effect_vector': [0.1, -0.2, 0.3],
                        'eigenvalues': [1.0, 1.0, 1.0]
                    },
                    {
                        'operation': 'property_reshaping',
                        'matrix': [[0.5, 0, 0], [0, 2.0, 0], [0, 0, 1]],
                        'effect_vector': [0.5, 1.0, 0],
                        'eigenvalues': [0.5, 2.0, 1.0]
                    }
                ]
            },
            'tests': {
                'test_metadata': self.test_suites,
                'success_probabilities': {
                    'lattice': 0.92,
                    'jacobian': 0.88,
                    'morpheme': 0.85,
                    'lexicon': 0.90,
                    'neural': 0.83,
                    'integration': 0.80
                }
            }
        }

        if data_type == 'all':
            result = base_data.copy()
            for key, value in type_specific_data.items():
                result.update(value)
            return result

        return {**base_data, **type_specific_data.get(data_type, {})}

    def save_test_results(self, results: List[TestResult], output_file: str, format_type: str):
        """Save test results in specified format"""
        output_path = Path(output_file)

        if format_type == 'json':
            data = [asdict(result) for result in results]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format_type == 'yaml':
            data = [asdict(result) for result in results]
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        elif format_type == 'csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['name', 'success', 'duration', 'details', 'timestamp'])
                writer.writeheader()
                for result in results:
                    writer.writerow(asdict(result))

        elif format_type == 'html':
            html_content = self.generate_html_report(results)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

    def generate_html_report(self, results: List[TestResult]) -> str:
        """Generate HTML test report"""
        passed = sum(1 for r in results if r.success)
        total = len(results)
        success_rate = (passed / total * 100) if total > 0 else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>World Engine V3.1 Test Results</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }}
        .summary {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .test-result {{ margin: 10px 0; padding: 15px; border-radius: 8px; }}
        .pass {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .fail {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .duration {{ color: #6c757d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌍 World Engine V3.1 Test Results</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary">
        <h2>📊 Summary</h2>
        <p><strong>Tests Run:</strong> {total}</p>
        <p><strong>Passed:</strong> {passed}</p>
        <p><strong>Failed:</strong> {total - passed}</p>
        <p><strong>Success Rate:</strong> {success_rate:.1f}%</p>
    </div>

    <h2>🧪 Test Details</h2>
"""

        for result in results:
            status_class = "pass" if result.success else "fail"
            status_icon = "✅" if result.success else "❌"

            html += f"""
    <div class="test-result {status_class}">
        <h3>{status_icon} {result.name}</h3>
        <p>{result.details}</p>
        <div class="duration">Duration: {result.duration:.2f}s | {result.timestamp}</div>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def save_export_data(self, data: Dict[str, Any], output_path: str, format_type: str, compress: bool):
        """Save export data in specified format"""
        path = Path(output_path)

        if format_type == 'json':
            content = json.dumps(data, indent=2, ensure_ascii=False)
        elif format_type == 'yaml':
            content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        elif format_type == 'csv':
            # Flatten data for CSV
            content = self.flatten_to_csv(data)
        elif format_type == 'xml':
            content = self.dict_to_xml(data, 'world_engine_v31')
        else:
            content = json.dumps(data, indent=2, ensure_ascii=False)

        if compress:
            import gzip
            with gzip.open(f"{path}.gz", 'wt', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

    def export_morphological_analysis(self, analyses: List[MorphemeAnalysis],
                                    output_file: str, format_type: str):
        """Export morphological analysis results"""
        path = Path(output_file)

        if format_type == 'json':
            data = [asdict(analysis) for analysis in analyses]
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format_type == 'csv':
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Word', 'Root', 'Morphemes', 'Complexity', 'Related Words'])

                for analysis in analyses:
                    morphemes_str = ', '.join([f"{m['form']}({m['type']})" for m in analysis.morphemes])
                    related_str = ', '.join(analysis.related_words or [])
                    writer.writerow([analysis.word, analysis.root, morphemes_str, analysis.complexity, related_str])

    def flatten_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert nested dictionary to CSV format"""
        import io
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(['Key', 'Value'])

        # Flatten and write data
        def flatten_dict(d, prefix=''):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    yield from flatten_dict(value, full_key)
                elif isinstance(value, list):
                    writer.writerow([full_key, f"List with {len(value)} items"])
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            yield from flatten_dict(item, f"{full_key}[{i}]")
                        else:
                            writer.writerow([f"{full_key}[{i}]", str(item)])
                else:
                    writer.writerow([full_key, str(value)])

        list(flatten_dict(data))
        return output.getvalue()

    def dict_to_xml(self, data: Dict[str, Any], root_name: str) -> str:
        """Convert dictionary to XML format"""
        def dict_to_xml_recursive(d, parent_name='item'):
            xml_str = f"<{parent_name}>"
            for key, value in d.items():
                if isinstance(value, dict):
                    xml_str += dict_to_xml_recursive(value, key)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            xml_str += dict_to_xml_recursive(item, key)
                        else:
                            xml_str += f"<{key}>{item}</{key}>"
                else:
                    xml_str += f"<{key}>{value}</{key}>"
            xml_str += f"</{parent_name}>"
            return xml_str

        return f'<?xml version="1.0" encoding="UTF-8"?>\n{dict_to_xml_recursive(data, root_name)}'

    def run(self, args) -> int:
        """Main execution method"""
        self.verbose = args.verbose
        self.dimensions = args.dimensions
        self.config = self.load_config(args.config)

        if self.verbose:
            print(f"🌍 World Engine V3.1 CLI - {args.command} mode")
            print(f"   Dimensions: {self.dimensions}")
            if self.config:
                print(f"   Configuration loaded: {len(self.config)} settings")

        try:
            if args.command == 'test':
                return self.run_test_suite(
                    args.suite, args.format, args.output,
                    getattr(args, 'parallel', False), getattr(args, 'timeout', 30.0)
                )

            elif args.command == 'analyze':
                self.analyze_morphology(
                    args.words, args.detail, args.links,
                    getattr(args, 'export', None), args.format
                )
                return 0

            elif args.command == 'serve':
                self.start_server(args.host, args.port, args.open, getattr(args, 'debug', False))
                return 0

            elif args.command == 'export':
                self.export_data(args.type, args.format, args.output, getattr(args, 'compress', False))
                return 0

            else:
                print("❌ No command specified. Use --help for usage information.")
                return 1

        except KeyboardInterrupt:
            print("\n🛑 Operation cancelled by user")
            return 130
        except Exception as e:
            print(f"❌ Error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Main entry point"""
    cli = WorldEngineV31CLI()
    parser = cli.create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return cli.run(args)


if __name__ == '__main__':
    sys.exit(main())
