// Runs lofi_bench's wasm32-unknown-unknown build under Node, lending it the
// clock and the console that target's std lacks. Usage: lofi_bench.rs's header.

import { readFileSync } from 'node:fs';
import { arch, cpus, platform } from 'node:os';

const [wasm] = process.argv.slice(2);
if (!wasm) {
  console.error('usage: node examples/lofi_bench.mjs <lofi_bench.wasm>');
  process.exit(2);
}
const decoder = new TextDecoder();
const { exports } = new WebAssembly.Instance(new WebAssembly.Module(readFileSync(wasm)), {
  bench: {
    now_ms: () => performance.now(),
    print: (ptr, len) =>
      console.log(decoder.decode(new Uint8Array(exports.memory.buffer, ptr, len))),
  },
});
console.log(`host: ${cpus()[0].model}, ${platform()} ${arch()}, Node ${process.version}`);
// a Rust `main`'s export takes (argc, argv) and returns its exit code
process.exitCode = exports.main(0, 0);
