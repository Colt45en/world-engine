# 🌟 R3F Editor Framework - Accessibility Edition

A professional-grade React Three.js editor with comprehensive accessibility features designed for comfort, inclusion, and flow.

## 🎯 Accessibility Features

### Reader Mode (Dyslexia-Friendly)
- **Atkinson Hyperlegible Font**: Specially designed font for enhanced readability
- **Optimized Spacing**: Improved letter spacing (0.02em) and word spacing (0.06em)
- **Enhanced Line Height**: 1.6 for better text flow
- **Strong Focus Indicators**: 3px blue outlines with 2px offset for clear navigation
- **Code Font Enhancement**: Enhanced monospace fonts with improved spacing

**Activation**: Press `R` or use Command Palette → "Reader Mode: Enable"

### Reduced Motion
- **UI Animations**: Shortened transitions (80ms) and minimal animations
- **3D Scene Damping**: Reduced camera damping (0.02 vs 0.08)
- **Entity Animations**: Gentler bobbing and rotation speeds
- **Visual Effects**: Minimal pulse effects and softer transitions
- **Performance Optimization**: Lower DPR and reduced GPU usage

**Activation**: Press `M` or use Command Palette → "Reduce Motion: Enable"

### UI Font Scaling
- **Dynamic Scaling**: 80% to 160% in 5% increments
- **CSS Variable Based**: Single `--ui-scale` variable controls all text
- **Real-time Updates**: Instant visual feedback
- **Responsive**: Adapts across different screen sizes

**Controls**:
- Increase: `+` key or Command Palette → "UI Font: Increase"
- Decrease: `-` key or Command Palette → "UI Font: Decrease"
- Reset: `0` key or Command Palette → "UI Font: Reset"

## 🎮 Usage

### Command Palette
Press `Ctrl+K` (Windows/Linux) or `Cmd+K` (Mac) to open the command palette:

```
⌘ Command Palette
├── Accessibility
│   ├── Reader Mode: Enable/Disable [R]
│   ├── Reduce Motion: Enable/Disable [M]
│   ├── UI Font: Increase [+]
│   ├── UI Font: Decrease [-]
│   └── UI Font: Reset [0]
├── Entities
│   ├── Add Cube [C]
│   ├── Add Sphere [S]
│   └── Delete Selected [Delete]
└── Edit
    ├── Undo [Ctrl+Z]
    └── Redo [Ctrl+Y]
```

### Accessibility Panel
The dedicated accessibility tool panel provides visual controls:

- **Toggle Buttons**: One-click enable/disable for Reader Mode and Reduced Motion
- **Font Scale Slider**: Visual slider with percentage display
- **Quick Actions**: "Max Comfort" and "Reset All" presets
- **Live Status**: Real-time feedback on active features
- **Keyboard Shortcuts**: Reference guide built-in

### Quick Preset Actions

**Max Comfort Mode**:
```typescript
// Enables all accessibility features
- Reader Mode: ✓ Enabled
- Reduce Motion: ✓ Enabled
- UI Scale: 120%
```

**Reset All**:
```typescript
// Returns to default state
- Reader Mode: ✗ Disabled
- Reduce Motion: ✗ Disabled
- UI Scale: 100%
```

## 🔧 Technical Implementation

### State Management
```typescript
// UI State Hook
const {
  dyslexiaMode,
  reducedMotion,
  uiScale,
  toggleReaderMode,
  toggleReducedMotion,
  setUiScale,
  bumpUiScale
} = useUI()
```

### CSS Integration
The accessibility features are applied through CSS classes and variables:

```css
/* Applied to root element */
.dyslexia-mode { /* Reader mode styles */ }
.reduced-motion { /* Motion reduction styles */ }

/* Dynamic UI scaling */
:root { --ui-scale: 1; }
html, body { font-size: calc(16px * var(--ui-scale)); }
```

### R3F Scene Integration
The 3D scene respects accessibility preferences:

```typescript
// Camera controls adapt to motion preferences
const damping = reducedMotion ? 0.02 : 0.08
const rotateSpeed = reducedMotion ? 0.2 : 0.8

// Entity animations scale with motion settings
const motionMultiplier = reducedMotion ? 0.3 : 1.0
```

## 🎨 Design Principles

### Universal Design
- **Progressive Enhancement**: Features enhance rather than replace core functionality
- **No Cognitive Load**: Zero-thinking activation and usage
- **Preserved Functionality**: All editor features remain fully accessible

### Inclusive Patterns
- **Multiple Access Methods**: Keyboard shortcuts, command palette, and visual controls
- **Clear Feedback**: Visual and semantic indicators for all states
- **Graceful Degradation**: Smooth fallbacks when features are disabled

### Performance Conscious
- **Efficient Rendering**: Reduced GPU load for motion-sensitive users
- **Optimized Animations**: Shorter durations and simpler easing
- **Resource Management**: Lower DPR and power-friendly settings

## 📋 Development Setup

### VS Code Configuration
The framework includes comprehensive VS Code settings:

- **Spell Checking**: Custom dictionary with R3F/WebGL terms
- **Code Formatting**: Prettier/ESLint integration
- **Accessibility Extensions**: Enhanced development experience
- **Font Recommendations**: Cascadia Code with ligatures

### Project Structure
```
src/
├── state/
│   ├── ui.ts              # Accessibility state management
│   └── editor.ts          # Core editor state
├── styles/
│   └── globals.css        # Accessibility CSS
├── tools/
│   ├── AccessibilityTool.tsx    # Visual accessibility panel
│   ├── CommandPalette.tsx       # Command interface
│   └── ToolManager.tsx          # Tool orchestration
└── scene/
    ├── SceneRoot.tsx            # Motion-aware scene root
    └── layers/
        └── EntityLayer.tsx      # Motion-responsive entities
```

## 🚀 Extension Points

### Custom Accessibility Features
```typescript
// Extend the UI store with custom features
interface ExtendedUIState extends UIState {
  highContrast: boolean
  largePointers: boolean
  audioFeedback: boolean
}
```

### Scene Adaptations
```typescript
// Add your own motion-sensitive components
function CustomEntity() {
  const { reducedMotion } = useUI()
  const animationSpeed = reducedMotion ? 0.1 : 1.0

  // Your motion-aware logic
}
```

### Command Extensions
```typescript
// Add accessibility commands to the palette
const customCommands = [
  {
    id: 'accessibility.highContrast.toggle',
    title: 'Toggle High Contrast',
    run: toggleHighContrast,
    group: 'Accessibility'
  }
]
```

## 💡 Best Practices

### Implementation Guidelines
- **Test with Real Users**: Validate with users who have accessibility needs
- **Keyboard Navigation**: Ensure all features are keyboard accessible
- **Screen Reader Support**: Provide semantic HTML and ARIA attributes
- **Color Independence**: Don't rely solely on color for information

### Performance Optimization
- **Conditional Rendering**: Only apply expensive effects when needed
- **Efficient Updates**: Use React.memo and useMemo for accessibility components
- **Resource Scaling**: Automatically adjust quality based on accessibility preferences

## 🔍 Testing

### Manual Testing Checklist
- [ ] Reader Mode enables and applies font changes
- [ ] Reduced Motion affects both UI and 3D animations
- [ ] UI scaling works across all components
- [ ] Keyboard shortcuts function correctly
- [ ] Command palette is accessible via keyboard
- [ ] Focus indicators are visible and consistent

### Accessibility Audit
- [ ] Color contrast meets WCAG AA standards
- [ ] All interactive elements are focusable
- [ ] Screen reader announces state changes
- [ ] No motion without user consent
- [ ] Text remains readable at 200% zoom

## 🎯 Future Enhancements

### Planned Features
- **High Contrast Mode**: System-level contrast preference detection
- **Audio Feedback**: Optional sound cues for interactions
- **Voice Controls**: Speech recognition for command input
- **Gesture Support**: Touch-friendly accessibility gestures
- **Custom Themes**: User-defined accessibility color schemes

### Integration Opportunities
- **System Preferences**: Automatically detect OS accessibility settings
- **Save Preferences**: Persist user accessibility choices
- **Workspace Sync**: Share settings across development environments
- **Analytics**: Anonymous usage data to improve accessibility features

---

## 🤝 Contributing

We welcome contributions that improve accessibility! Please ensure:

1. **Test with Assistive Technology**: Validate changes with screen readers and other tools
2. **Follow WCAG Guidelines**: Adhere to Web Content Accessibility Guidelines
3. **Document Changes**: Update this README with new accessibility features
4. **Performance Impact**: Consider the performance implications of accessibility features

---

*Built with ❤️ for everyone. Accessibility is not a feature—it's a foundation.*
