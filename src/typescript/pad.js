(function () {
    const vscode = acquireVsCodeApi();

    require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.51.0/min/vs' } });
    require(['vs/editor/editor.main'], function () {
        let editor;

        function initEditor(text, language, theme) {
            if (!editor) {
                editor = monaco.editor.create(document.getElementById('editor'), {
                    value: text || '',
                    language: language || 'javascript',
                    fontSize: 14,
                    lineNumbers: 'on',
                    automaticLayout: true,
                    renderLineHighlight: 'all',
                    renderIndentGuides: true,
                    tabSize: 4,
                    insertSpaces: true,
                    rulers: [80, 120],
                    minimap: { enabled: false },
                    theme: theme || 'vs'
                });

                let t;
                editor.onDidChangeModelContent(() => {
                    clearTimeout(t);
                    t = setTimeout(() => {
                        const text = editor.getValue();
                        const padName = document.getElementById('padSwitcher').value || 'default';
                        vscode.postMessage({ type: 'save', padName, text });
                        vscode.setState({ padName, text });
                    }, 250);
                });
            } else {
                editor.setValue(text || '');
                if (language) monaco.editor.setModelLanguage(editor.getModel(), language);
                if (theme) monaco.editor.setTheme(theme);
            }
        }

        // Toolbar wiring
        function wireToolbar() {
            document.querySelectorAll('#toolbar [data-cmd]').forEach(btn => {
                btn.addEventListener('click', () => {
                    const command = btn.getAttribute('data-cmd');
                    vscode.postMessage({ type: 'command', command });
                });
            });

            const switcher = document.getElementById('padSwitcher');
            if (switcher) {
                switcher.addEventListener('change', () => {
                    const padName = switcher.value;
                    vscode.postMessage({ type: 'switchPad', padName });
                });
            }
        }

        // Handle messages from extension
        window.addEventListener('message', (e) => {
            const msg = e.data;
            if (msg.type === 'state') {
                // pads + currentPad + language + theme + text
                const switcher = document.getElementById('padSwitcher');
                if (switcher) {
                    switcher.innerHTML = '';
                    (msg.pads || ['default']).forEach((name) => {
                        const opt = document.createElement('option');
                        opt.value = name;
                        opt.textContent = name;
                        if (name === msg.currentPad) opt.selected = true;
                        switcher.appendChild(opt);
                    });
                }
                initEditor(msg.text || '', msg.language, msg.theme);
            }
            if (msg.type === 'setLanguage') {
                if (editor) monaco.editor.setModelLanguage(editor.getModel(), msg.language);
            }
            if (msg.type === 'setTheme') {
                if (editor) monaco.editor.setTheme(msg.theme);
            }
            if (msg.type === 'loadPad') {
                if (editor) editor.setValue(msg.text || '');
            }
        });

        wireToolbar();
    });
})();
