# WoW Lite Tier-4 Bridge - React Three Fiber Integration

A complete React Three Fiber (R3F) implementation of the Tier-4 Room Bridge system, integrated into a WoW Lite game environment.

## 🎮 Overview

This component converts the original TypeScript Tier4RoomBridge class into a modern React Three Fiber component that provides:

- **3D Visual Bridge**: Interactive 3D representation of the Tier-4 system
- **Real-time WebSocket**: Automatic connection to nucleus events
- **WoW-Themed Operators**: Game-specific operator mappings (Warrior, Mage, Paladin, etc.)
- **Macro Sequences**: Class-specific ability sequences
- **State Visualization**: Real-time 3D visualization of system state

## 🚀 Features

### 3D Visualization
- **Bridge Structure**: Central 3D bridge that changes color based on connection status
- **State Spheres**: Four colored spheres representing the state vector `x`
- **Kappa Indicator**: Dynamic cylinder showing kappa values
- **Connection Indicator**: Pulsing sphere for WebSocket status

### WoW Lite Integration
- **Class Operators**: ST (Strength), SL (Spell Power), CP (Critical), etc.
- **Class Macros**: WARRIOR_RAGE, MAGE_ARCANE, PALADIN_HOLY, DRUID_BALANCE
- **Game Theming**: WoW-inspired visual design and terminology

### Real-time Systems
- **WebSocket Bridge**: Automatic reconnection and message handling
- **Nucleus Integration**: Maps nucleus events to WoW operators
- **State Management**: React hooks for state tracking
- **Event System**: Custom events for operator/macro application

## 📦 Installation

```bash
npm install @react-three/fiber @react-three/drei three
```

## 🛠️ Usage

### Basic Integration

```tsx
import React from 'react';
import { Canvas } from '@react-three/fiber';
import { WoWLiteTier4Bridge } from './wow_lite_tier4_bridge';

function App() {
  return (
    <Canvas>
      <WoWLiteTier4Bridge
        websocketUrl="ws://localhost:8080/ws"
        position={[0, 0, 0]}
      />
    </Canvas>
  );
}
```

### Advanced Integration with State Tracking

```tsx
import React from 'react';
import { Canvas } from '@react-three/fiber';
import { WoWLiteTier4Bridge, useWoWLiteTier4Bridge } from './wow_lite_tier4_bridge';

function WoWGame() {
  const { bridgeState, lastOperator } = useWoWLiteTier4Bridge();

  return (
    <div>
      {/* Game UI */}
      <div>Current Kappa: {bridgeState?.kappa.toFixed(3)}</div>
      <div>Last Operator: {lastOperator}</div>

      {/* 3D Scene */}
      <Canvas>
        <WoWLiteTier4Bridge
          onStateChange={(state) => console.log('State changed:', state)}
          onOperatorApplied={(op, state) => console.log('Applied:', op)}
        />
      </Canvas>
    </div>
  );
}
```

## 🎯 Component Props

```tsx
interface WoWLiteTier4BridgeProps {
  websocketUrl?: string;        // WebSocket URL (default: ws://localhost:8080/ws)
  onStateChange?: (state: Tier4State) => void;     // State change callback
  onOperatorApplied?: (operator: string, state: Tier4State) => void; // Operator applied callback
  position?: [number, number, number];             // 3D position (default: [0, 0, 0])
}
```

## 🎮 WoW Lite Operators

| Operator | Name | Effect | Theme |
|----------|------|--------|-------|
| ST | Strength | +10% to first state component | Warrior power |
| SL | Spell Power | +10% to second state component | Mage intellect |
| CP | Critical Power | +15% to third state component | Rogue precision |
| CV | Convert | Swap first/second components | Druid shapeshifting |
| PR | Protect | +20% to fourth state component | Paladin defense |
| RC | Recover | -20% to fourth component | Priest healing |
| TL | Teleport | -10% to all components | Mage teleportation |
| RB | Rebirth | Reset to 50% + 50% bonus | Death Knight rebirth |
| MD | Modify | Small adjustments to all | Warlock corruption |

## 🎪 WoW Lite Macros

| Macro | Sequence | Class Theme |
|-------|----------|-------------|
| WARRIOR_RAGE | ST → CP → PR | Berserker rage build |
| MAGE_ARCANE | SL → CV → MD | Arcane power manipulation |
| PALADIN_HOLY | PR → RB → SL | Holy protection and healing |
| DRUID_BALANCE | CV → TL → RC | Nature balance and restoration |

## 🔧 WebSocket Integration

The bridge automatically connects to WebSocket and handles:

- **Nucleus Events**: Maps VIBRATE→ST, OPTIMIZATION→SL, STATE→CP, SEED→RB
- **Auto-Operators**: Automatically applies operators based on nucleus events
- **Reconnection**: Automatic reconnection with exponential backoff
- **Event Forwarding**: Forwards all events to the 3D visualization

## 🎨 Customization

### Styling
The component uses external CSS (`wow_lite_tier4_bridge.css`) for all UI elements. Modify colors, sizes, and animations there.

### Operators
Extend the `operators` object to add new WoW classes or abilities:

```tsx
const customOperators = {
  ...operators,
  'DK_FROST': {
    name: 'Frost Strike',
    matrix: [[1.05, 0, 0, 0], [0, 1.05, 0, 0], [0, 0, 1.05, 0], [0, 0, 0, 0.9]],
    bias: [0.05, 0.05, 0.05, -0.1],
    kappaDelta: 0.02
  }
};
```

### Visual Effects
Modify the 3D elements in the component to add particle effects, animations, or additional visual elements.

## 🔌 API Reference

### useWoWLiteTier4Bridge Hook

```tsx
const {
  bridgeState,        // Current Tier4State
  lastOperator,       // Last applied operator
  handleStateChange,  // State change handler
  handleOperatorApplied // Operator applied handler
} = useWoWLiteTier4Bridge();
```

### Manual Operator Application

```tsx
// Apply operator programmatically
const event = new CustomEvent('apply-operator', {
  detail: { operator: 'ST' }
});
window.dispatchEvent(event);

// Apply macro programmatically
const macroEvent = new CustomEvent('apply-macro', {
  detail: { macro: 'WARRIOR_RAGE' }
});
window.dispatchEvent(macroEvent);
```

## 🏗️ Architecture

```
WoWLiteTier4Bridge (R3F Component)
├── 3D Bridge Structure (connection status)
├── State Spheres (x vector visualization)
├── Kappa Indicator (dynamic cylinder)
├── Connection Indicator (pulsing sphere)
├── Operator Panel (HTML overlay)
├── Macro Panel (HTML overlay)
├── WebSocket Manager (auto-reconnect)
├── State Transformer (matrix operations)
└── Event Handlers (nucleus integration)
```

## 🎯 Integration with WoW Lite Game

1. **Import the component** into your main game scene
2. **Position it** in your 3D world where players can interact
3. **Connect callbacks** to update game state/UI
4. **Use operators** to modify game mechanics (damage, healing, etc.)
5. **Trigger macros** for complex ability sequences

## 🚀 Performance

- **Efficient Rendering**: Uses React Three Fiber's optimized rendering
- **Minimal Re-renders**: State updates only trigger necessary re-renders
- **WebSocket Optimization**: Connection pooling and message batching
- **Memory Management**: Automatic cleanup of event listeners

## 🐛 Troubleshooting

### WebSocket Connection Issues
- Check that your WebSocket server is running on the specified URL
- Verify CORS settings allow connections from your domain
- Check browser console for connection errors

### 3D Rendering Issues
- Ensure you have proper lighting in your scene
- Check that @react-three/fiber and @react-three/drei are installed
- Verify Three.js version compatibility

### State Not Updating
- Check that event listeners are properly attached
- Verify operator names match the defined operators
- Check browser console for state transformation errors

## 📝 License

This component is part of the World Engine project. See main project license for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add WoW-themed operators or macros
4. Test with WebSocket integration
5. Submit a pull request

---

**Ready to build your WoW Lite game with advanced Tier-4 bridge technology!** ⚔️✨
