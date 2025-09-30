//Worldengine-tier4-bundle
// javascript
/*!
 * WorldEngine Tier-4 Bundle
 * Combines: EngineRoom + RoomLayout + Tier4Room (browser)
 *          + StudioBridge wiring
 *          + createTier4RoomBridge(iframe, wsUrl)
 *          + Tier4EnhancedRelay (Node-only)
 * UMD export: window.WorldEngineTier4 or module.exports
 */

import { StudioBridge } from './src/nucleus/worldengine-tier4-bundle';

((global, factory) => {
    if (typeof module === 'object' && typeof module.exports === 'object') {
        module.exports = factory(global);
    } else {
        (global as any).WorldEngineTier4 = factory(global as any);
    }
})((typeof window !== 'undefined' ? window : globalThis) as any, function (root: any) {
  'use strict';


  // --------------------------------------
  // StudioBridge (link or shim)
  // - If a full StudioBridge already exists, we reuse it.
  // - Otherwise we create a minimal bus-compatible shim so this file works standalone.
  // --------------------------------------
  var StudioBridge = (function ensureBridge() {
    if (root.StudioBridge) return root.StudioBridge;


    var busName = 'studio-bus';
    var bc = (typeof BroadcastChannel !== 'undefined') ? new BroadcastChannel(busName) : null;


    var listeners: { type: any; fn: any; }[] = [];
    function callListeners(msg: { type: any; }) {
      // 1) fn(msg)
      listeners.forEach(function (entry) {
        try {
          if (!entry.type || entry.type === msg.type) entry.fn(msg);
        } catch (e) {
          console.error('[StudioBridge shim] listener error', e);
        }
      });
    }


    if (bc) {
      bc.onmessage = function (e) { callListeners(e.data); };
    } else if (root && root.addEventListener) {
      root.addEventListener('studio:msg', function (e: { detail: any; }) { callListeners(e.detail); });
    }


    function onBus(typeOrFn: any, maybeFn: any) {
      if (typeof typeOrFn === 'function') {
        listeners.push({ type: null, fn: typeOrFn });
      } else if (typeof maybeFn === 'function') {
        listeners.push({ type: typeOrFn, fn: maybeFn });
      }
    }


    function sendBus(msg: any) {
      if (bc) bc.postMessage(msg);
      else if (root && root.dispatchEvent)
        root.dispatchEvent(new CustomEvent('studio:msg', { detail: msg }));
    }


    var shim = {
      onBus: onBus,
      sendBus: sendBus,
      RecorderAPI: root.RecorderAPI || {},
      EngineAPI: root.EngineAPI || {},
      ChatAPI: root.ChatAPI || {},
      Store: {
        save: async function (key: any, value: any) {
          if (root.localStorage) root.localStorage.setItem(key, JSON.stringify(value));
          return value;
        },

/*
Done! I bundled everything into a single UMD script and wired it to the EngineRoom and StudioBridge layers.

What you’ve got:

EngineRoom / RoomLayout / Tier4Room (browser) with panels and snapshot/event plumbing.

StudioBridge link/shim so it auto-hooks into your existing bus (or falls back if missing).

createTier4RoomBridge(iframe, wsUrl) — the glue your EngineRoom React component expects. It connects the iframe room, adds operator/macro handlers, syncs snapshots, and talks to your WebSocket relay. It also forwards messages over the Studio bus (tier4.applyOperator, tier4.applyMacro, tier4.setState) and emits tier4.operatorApplied.

Tier4EnhancedRelay (Node-only) included in the same file; it’s exported when required from Node.

Open the canvas to see/edit the full file:
worldengine-tier4-bundle.js (on the right)

How to use it

In the browser (with your React EngineRoom):

<script src="/js/worldengine-tier4-bundle.js"></script>
<script>
  // If you’re not using the React wrapper, you can still call:
  // WorldEngineTier4.createTier4RoomBridge(iframeEl, 'ws://localhost:9000')
</script>


Your existing React EngineRoom that calls createTier4RoomBridge(iframe, websocketUrl) will just work.

StudioBridge bus hooks:

Send: { type: 'tier4.applyOperator', operator: 'UP' }

Send: { type: 'tier4.applyMacro', macro: 'MERGE_ABC' }

Send: { type: 'tier4.setState', state: { x:[…], kappa:…, level:… } }

Receive: { type: 'tier4.operatorApplied', operator, previousState, newState }

WebSocket relay (Node):

// node relay.js
const { Tier4EnhancedRelay } = require('./worldengine-tier4-bundle.js');
new Tier4EnhancedRelay(process.env.PORT || 9000);

Notes

No duplicated typedefs; everything is consolidated.

Operator panel buttons inside the room post apply-operator / apply-macro — the bridge listens and applies them.

A small default transform is included so state changes are visible even without your custom kernel logic; swap it by passing transform to createTier4RoomBridge(…, { transform }).

If you want me to also fold in your earlier TypeLattice and Studio Bridge Event Bus code into this same bundle, say the word and I’ll slot them under the same UMD wrapper with namespaced exports.
*/

    function javascript(arg0: (global: { WorldEngineTier4: any; }, factory: (arg0: any) => any) => void) {
        throw new Error('Function not implemented.');
    }
        throw new Error('Function not implemented.');
    }

    function javascript(arg0: (global: { WorldEngineTier4: any; }, factory: (arg0: any) => any) => void) {
        throw new Error('Function not implemented.');
    }

    function javascript(arg0: (global: { WorldEngineTier4: any; }, factory: (arg0: any) => any) => void) {
    // Placeholder for the Tier4Room implementation; to be implemented in the future.
    function Tier4Room(browser: any) {
        throw new Error('Function not implemented.');
    }
    function Tier4Room(browser: any) {
        throw new Error('Function not implemented.');
    }

    function bus(or: typeof or, falls: any, back: any) {
        throw new Error('Function not implemented.');
    }

    function browser() {
        throw new Error('Function not implemented.');
    }

    function createTier4RoomBridge(iframe: any, websocketUrl: any) {
        throw new Error('Function not implemented.');
    }

    function relay(Node: { new(): Node; prototype: Node; readonly ELEMENT_NODE: 1; readonly ATTRIBUTE_NODE: 2; readonly TEXT_NODE: 3; readonly CDATA_SECTION_NODE: 4; readonly ENTITY_REFERENCE_NODE: 5; readonly ENTITY_NODE: 6; readonly PROCESSING_INSTRUCTION_NODE: 7; readonly COMMENT_NODE: 8; readonly DOCUMENT_NODE: 9; readonly DOCUMENT_TYPE_NODE: 10; readonly DOCUMENT_FRAGMENT_NODE: 11; readonly NOTATION_NODE: 12; readonly DOCUMENT_POSITION_DISCONNECTED: 1; readonly DOCUMENT_POSITION_PRECEDING: 2; readonly DOCUMENT_POSITION_FOLLOWING: 4; readonly DOCUMENT_POSITION_CONTAINS: 8; readonly DOCUMENT_POSITION_CONTAINED_BY: 16; readonly DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC: 32; }) {
        throw new Error('Function not implemented.');
}
    function relay(Node: { new(): Node; prototype: Node; readonly ELEMENT_NODE: 1; readonly ATTRIBUTE_NODE: 2; readonly TEXT_NODE: 3; readonly CDATA_SECTION_NODE: 4; readonly ENTITY_REFERENCE_NODE: 5; readonly ENTITY_NODE: 6; readonly PROCESSING_INSTRUCTION_NODE: 7; readonly COMMENT_NODE: 8; readonly DOCUMENT_NODE: 9; readonly DOCUMENT_TYPE_NODE: 10; readonly DOCUMENT_FRAGMENT_NODE: 11; readonly NOTATION_NODE: 12; readonly DOCUMENT_POSITION_DISCONNECTED: 1; readonly DOCUMENT_POSITION_PRECEDING: 2; readonly DOCUMENT_POSITION_FOLLOWING: 4; readonly DOCUMENT_POSITION_CONTAINS: 8; readonly DOCUMENT_POSITION_CONTAINED_BY: 16; readonly DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC: 32; }) {
        throw new Error('Function not implemented.');
    }

    }
