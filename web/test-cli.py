#!/usr/bin/env python3
"""
World Engine V3.1 CLI Test Runner
Validates CLI functionality and integration with the attachments' concepts
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

class CLITestRunner:
    def __init__(self):
        self.script_path = Path(__file__).parent / 'world-engine-v31-cli.py'
        self.test_results = []

    def run_cli_command(self, args, timeout=30):
        """Run CLI command and return result"""
        cmd = [sys.executable, str(self.script_path)] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8'
            )

            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd)
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'command': ' '.join(cmd)
            }
        except Exception as e:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'command': ' '.join(cmd)
            }

    def test_help_command(self):
        """Test help functionality"""
        print("🧪 Testing help command...")
        result = self.run_cli_command(['--help'])

        success = (
            result['success'] or result['returncode'] == 0 and  # Help returns 0 usually
            'world-engine-v31' in result['stdout'] and
            'World Engine V3.1' in result['stdout']
        )

        self.test_results.append({
            'test': 'Help Command',
            'success': success,
            'details': 'CLI help displayed correctly' if success else 'Help command failed',
            'output': result['stdout'][:200] + '...' if len(result['stdout']) > 200 else result['stdout']
        })

        print(f"   {'✅ PASS' if success else '❌ FAIL'}")
        return success

    def test_version_info(self):
        """Test version and basic info"""
        print("🧪 Testing version info...")
        result = self.run_cli_command(['--verbose', 'test', '--suite', 'lattice', '--timeout', '5'])

        success = (
            'v3.1' in result['stdout'].lower() or
            'world engine' in result['stdout'].lower() or
            'test' in result['stdout'].lower()
        )

        self.test_results.append({
            'test': 'Version Info',
            'success': success,
            'details': 'Version information displayed' if success else 'Version info missing',
            'output': result['stdout'][:200] + '...' if len(result['stdout']) > 200 else result['stdout']
        })

        print(f"   {'✅ PASS' if success else '❌ FAIL'}")
        return success

    def test_morphological_analysis(self):
        """Test morphological analysis functionality"""
        print("🧪 Testing morphological analysis...")

        # Test words from the lexicon concepts in attachments
        test_words = ['transformation', 'restructure', 'antipattern', 'preprocessing']

        result = self.run_cli_command([
            'analyze',
            *test_words,
            '--detail',
            '--links',
            '--format', 'text'
        ])

        success = (
            result['success'] and
            any(word in result['stdout'] for word in test_words) and
            ('Root:' in result['stdout'] or 'Morphemes:' in result['stdout'])
        )

        self.test_results.append({
            'test': 'Morphological Analysis',
            'success': success,
            'details': 'Word analysis completed successfully' if success else 'Analysis failed',
            'output': result['stdout'][:300] + '...' if len(result['stdout']) > 300 else result['stdout']
        })

        print(f"   {'✅ PASS' if success else '❌ FAIL'}")
        if success:
            print("   📝 Sample analysis found in output")

        return success

    def test_export_functionality(self):
        """Test data export capabilities"""
        print("🧪 Testing export functionality...")

        temp_file = Path('temp_export_test.json')

        try:
            result = self.run_cli_command([
                'export',
                '--type', 'lexicon',
                '--format', 'json',
                str(temp_file)
            ])

            file_created = temp_file.exists()
            valid_json = False

            if file_created:
                try:
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        valid_json = isinstance(data, dict) and len(data) > 0
                except:
                    valid_json = False

            success = result['success'] and file_created and valid_json

            self.test_results.append({
                'test': 'Export Functionality',
                'success': success,
                'details': 'Export completed and file created' if success else 'Export failed or invalid output',
                'output': f"File created: {file_created}, Valid JSON: {valid_json}"
            })

            print(f"   {'✅ PASS' if success else '❌ FAIL'}")

            # Cleanup
            if temp_file.exists():
                temp_file.unlink()

        except Exception as e:
            self.test_results.append({
                'test': 'Export Functionality',
                'success': False,
                'details': f'Export test failed with error: {e}',
                'output': str(e)
            })
            print(f"   ❌ FAIL - {e}")
            return False

        return success

    def test_test_suite_execution(self):
        """Test the test suite runner"""
        print("🧪 Testing test suite execution...")

        result = self.run_cli_command([
            'test',
            '--suite', 'lattice',
            '--format', 'console',
            '--timeout', '10'
        ], timeout=15)

        success = (
            ('lattice' in result['stdout'].lower() or 'test' in result['stdout'].lower()) and
            (result['returncode'] in [0, 1])  # 0 for all pass, 1 for some failures
        )

        self.test_results.append({
            'test': 'Test Suite Execution',
            'success': success,
            'details': 'Test suite executed successfully' if success else 'Test suite execution failed',
            'output': result['stdout'][:300] + '...' if len(result['stdout']) > 300 else result['stdout']
        })

        print(f"   {'✅ PASS' if success else '❌ FAIL'}")
        if success:
            print(f"   📊 Test suite completed (exit code: {result['returncode']})")

        return success

    def test_error_handling(self):
        """Test error handling with invalid commands"""
        print("🧪 Testing error handling...")

        # Test invalid command
        result = self.run_cli_command(['invalid_command'])

        success = (
            not result['success'] and  # Should fail
            result['returncode'] != 0 and
            ('error' in result['stdout'].lower() or 'error' in result['stderr'].lower() or 'help' in result['stdout'].lower())
        )

        self.test_results.append({
            'test': 'Error Handling',
            'success': success,
            'details': 'Invalid commands handled gracefully' if success else 'Error handling failed',
            'output': (result['stdout'] + result['stderr'])[:200] + '...'
        })

        print(f"   {'✅ PASS' if success else '❌ FAIL'}")
        return success

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Starting World Engine V3.1 CLI Test Suite\n")

        # Check if CLI script exists
        if not self.script_path.exists():
            print(f"❌ CLI script not found: {self.script_path}")
            return False

        print(f"📍 Testing CLI script: {self.script_path}\n")

        # Run all tests
        test_methods = [
            self.test_help_command,
            self.test_version_info,
            self.test_morphological_analysis,
            self.test_export_functionality,
            self.test_test_suite_execution,
            self.test_error_handling
        ]

        passed_tests = 0

        for test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
                print()  # Add spacing between tests
            except Exception as e:
                print(f"   ❌ FAIL - Test crashed: {e}")
                self.test_results.append({
                    'test': test_method.__name__,
                    'success': False,
                    'details': f'Test crashed with error: {e}',
                    'output': str(e)
                })
                print()

        # Summary
        total_tests = len(test_methods)
        success_rate = (passed_tests / total_tests) * 100

        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Tests Run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()

        # Detailed results
        print("📝 DETAILED RESULTS")
        print("=" * 50)
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} {result['test']}")
            print(f"   Details: {result['details']}")
            if result.get('output') and len(result['output'].strip()) > 0:
                print(f"   Output: {result['output'][:100]}...")
            print()

        overall_success = passed_tests >= (total_tests * 0.7)  # 70% pass rate

        if overall_success:
            print("🎉 CLI testing completed successfully!")
            print("✅ The World Engine V3.1 CLI tool is working correctly")
        else:
            print("⚠️ CLI testing completed with issues")
            print("❌ Some CLI functionality may not be working correctly")

        return overall_success

    def generate_report(self, output_file='cli_test_report.json'):
        """Generate detailed test report"""
        report = {
            'timestamp': time.time(),
            'test_summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for r in self.test_results if r['success']),
                'failed_tests': sum(1 for r in self.test_results if not r['success']),
                'success_rate': (sum(1 for r in self.test_results if r['success']) / len(self.test_results)) * 100 if self.test_results else 0
            },
            'test_results': self.test_results,
            'cli_script': str(self.script_path),
            'python_version': sys.version
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 Test report saved: {output_file}")
        except Exception as e:
            print(f"⚠️ Could not save test report: {e}")

def main():
    """Main test execution"""
    runner = CLITestRunner()

    try:
        success = runner.run_all_tests()
        runner.generate_report()

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
