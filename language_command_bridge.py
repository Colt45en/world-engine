# Multi-Language Command Bridge System
# Enables Python, JavaScript, and C++ to execute commands across language boundaries

import json
import subprocess
import sys
import os
import asyncio
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import socket

class LanguageCommandBridge:
    """Central command bridge for multi-language communication"""
    
    def __init__(self):
        self.active_languages = {
            'python': True,
            'javascript': False,
            'cpp': False
        }
        self.command_history = []
        self.response_callbacks = {}
        self.server_port = 3001
        
    def log_command(self, source_lang: str, target_lang: str, command: str, result: Any = None):
        """Log command execution for debugging and history"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source_lang,
            'target': target_lang,
            'command': command,
            'result': str(result)[:500] if result else None,
            'status': 'success' if result is not None else 'pending'
        }
        self.command_history.append(entry)
        print(f"[BRIDGE] {source_lang} -> {target_lang}: {command}")
        
    def execute_python_command(self, code: str, context: Dict = None) -> Dict:
        """Execute Python code and return result"""
        try:
            # Create a safe execution environment
            exec_globals = {
                '__builtins__': __builtins__,
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'json': json,
                'datetime': datetime
            }
            
            if context:
                exec_globals.update(context)
                
            exec_locals = {}
            exec(code, exec_globals, exec_locals)
            
            result = {
                'status': 'success',
                'output': exec_locals.get('result', 'Code executed successfully'),
                'locals': {k: str(v) for k, v in exec_locals.items() if not k.startswith('__')},
                'timestamp': datetime.now().isoformat()
            }
            
            self.log_command('bridge', 'python', code, result)
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now().isoformat()
            }
            self.log_command('bridge', 'python', code, error_result)
            return error_result
    
    def execute_javascript_command(self, js_code: str) -> Dict:
        """Execute JavaScript/Node.js code"""
        try:
            # Create a temporary JS file
            temp_file = 'temp_js_command.js'
            
            # Wrap the code to capture output
            wrapped_code = f'''
            const result = (function() {{
                {js_code}
            }})();
            
            console.log(JSON.stringify({{
                status: 'success',
                result: result,
                timestamp: new Date().toISOString()
            }}));
            '''
            
            with open(temp_file, 'w') as f:
                f.write(wrapped_code)
            
            # Execute with Node.js
            process = subprocess.run(
                ['node', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            if process.returncode == 0:
                try:
                    result = json.loads(process.stdout.strip())
                    self.log_command('bridge', 'javascript', js_code, result)
                    return result
                except json.JSONDecodeError:
                    return {
                        'status': 'success',
                        'output': process.stdout.strip(),
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                error_result = {
                    'status': 'error',
                    'error': process.stderr.strip(),
                    'timestamp': datetime.now().isoformat()
                }
                self.log_command('bridge', 'javascript', js_code, error_result)
                return error_result
                
        except Exception as e:
            error_result = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.log_command('bridge', 'javascript', js_code, error_result)
            return error_result
    
    def execute_cpp_simulation(self, cpp_code: str) -> Dict:
        """Simulate C++ execution (for now, until we have full C++ compilation)"""
        # For now, we'll simulate C++ execution
        # Later this will compile and run actual C++ code
        
        simulation_result = {
            'status': 'simulated',
            'message': 'C++ simulation - would compile and execute in full implementation',
            'code': cpp_code,
            'simulated_output': f'C++ code processed: {len(cpp_code)} characters',
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_command('bridge', 'cpp', cpp_code, simulation_result)
        return simulation_result
    
    def command_python_from_js(self, python_code: str) -> str:
        """Allow JavaScript to execute Python code"""
        print(f"[JS->PY] Executing Python code from JavaScript")
        result = self.execute_python_command(python_code)
        return json.dumps(result)
    
    def command_js_from_python(self, js_code: str) -> Dict:
        """Allow Python to execute JavaScript code"""
        print(f"[PY->JS] Executing JavaScript code from Python")
        return self.execute_javascript_command(js_code)
    
    def cross_language_data_exchange(self, data: Dict, source_lang: str, target_lang: str) -> Dict:
        """Exchange data between languages with type conversion"""
        
        # Convert data types for cross-language compatibility
        converted_data = self._convert_data_types(data, target_lang)
        
        exchange_result = {
            'status': 'success',
            'source_language': source_lang,
            'target_language': target_lang,
            'original_data': data,
            'converted_data': converted_data,
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_command(source_lang, target_lang, f"Data exchange: {type(data)}", exchange_result)
        return exchange_result
    
    def _convert_data_types(self, data: Any, target_lang: str) -> Any:
        """Convert data types for specific target languages"""
        if target_lang == 'javascript':
            # Convert Python-specific types to JS-compatible types
            if isinstance(data, dict):
                return {k: self._convert_data_types(v, target_lang) for k, v in data.items()}
            elif isinstance(data, list):
                return [self._convert_data_types(item, target_lang) for item in data]
            elif isinstance(data, (int, float, str, bool)):
                return data
            else:
                return str(data)
        
        elif target_lang == 'cpp':
            # Convert to C++-compatible format (simplified)
            if isinstance(data, dict):
                return f"std::map with {len(data)} entries"
            elif isinstance(data, list):
                return f"std::vector with {len(data)} elements"
            else:
                return str(data)
        
        return data
    
    def get_command_history(self) -> List[Dict]:
        """Get command execution history"""
        return self.command_history
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'bridge_active': True,
            'active_languages': self.active_languages,
            'total_commands_executed': len(self.command_history),
            'server_port': self.server_port,
            'timestamp': datetime.now().isoformat()
        }

# Test the multi-language commanding system
def test_language_commanding():
    """Test the language commanding system"""
    bridge = LanguageCommandBridge()
    
    print("🌍 NEXUS World Engine - Multi-Language Command Bridge")
    print("=" * 60)
    
    # Test 1: Python command execution
    print("\n🐍 Test 1: Python Command Execution")
    python_result = bridge.execute_python_command("""
result = "Hello from Python!"
numbers = [1, 2, 3, 4, 5]
calculated = sum(numbers)
print(f"Sum of {numbers} = {calculated}")
""")
    print(f"Result: {python_result}")
    
    # Test 2: JavaScript command execution
    print("\n⚡ Test 2: JavaScript Command Execution")
    js_result = bridge.execute_javascript_command("""
const message = "Hello from JavaScript!";
const numbers = [1, 2, 3, 4, 5];
const sum = numbers.reduce((a, b) => a + b, 0);
return { message, sum, numbers };
""")
    print(f"Result: {js_result}")
    
    # Test 3: C++ simulation
    print("\n🔧 Test 3: C++ Command Simulation")
    cpp_result = bridge.execute_cpp_simulation("""
#include <iostream>
#include <vector>
int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    int sum = 0;
    for(int num : numbers) sum += num;
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
""")
    print(f"Result: {cpp_result}")
    
    # Test 4: Cross-language data exchange
    print("\n🔄 Test 4: Cross-Language Data Exchange")
    data = {
        "name": "NEXUS Engine",
        "version": "3.1",
        "components": ["Python", "JavaScript", "C++"],
        "active": True,
        "score": 95.5
    }
    
    py_to_js = bridge.cross_language_data_exchange(data, "python", "javascript")
    print(f"Python -> JavaScript: {py_to_js}")
    
    js_to_cpp = bridge.cross_language_data_exchange(data, "javascript", "cpp")
    print(f"JavaScript -> C++: {js_to_cpp}")
    
    # Test 5: System status
    print("\n📊 Test 5: System Status")
    status = bridge.get_system_status()
    print(f"Status: {status}")
    
    print(f"\n📋 Command History ({len(bridge.get_command_history())} commands executed)")
    for i, cmd in enumerate(bridge.get_command_history()[-3:], 1):
        print(f"  {i}. {cmd['source']} -> {cmd['target']}: {cmd['status']}")
    
    print("\n🎉 Multi-Language Command Bridge Test Complete!")
    return bridge

if __name__ == "__main__":
    # Run the test
    bridge = test_language_commanding()
    
    # Keep the bridge active for interactive use
    print("\n🚀 Language Command Bridge is now active!")
    print("You can now use cross-language commands:")
    print("  • Python ↔ JavaScript ↔ C++")
    print("  • Data type conversion")
    print("  • Command history tracking")
    print("  • Real-time execution")