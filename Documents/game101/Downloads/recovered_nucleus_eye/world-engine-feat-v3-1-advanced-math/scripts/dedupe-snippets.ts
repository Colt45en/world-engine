import { SnipStore } from "../src/snips/store.js";

const store = new SnipStore(process.cwd());
const r = await store.dedupe();
console.log(`dedupe: ${r.before} → ${r.after}`);
