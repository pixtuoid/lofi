// Runs lofi_bench's wasm32-unknown-unknown build under Node, lending it what
// its wasm32 `host` imports, whose doc says why. Usage: lofi_bench.rs's header.

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
// `main` is LLVM's (argc, argv) wrapper of a zero-argument C main
// (https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/llvm/lib/Target/WebAssembly/WebAssemblyFixFunctionBitcasts.cpp#L238-L295),
// here rustc's on this target
// (https://github.com/rust-lang/rust/blob/1.98.1/compiler/rustc_target/src/spec/base/wasm.rs#L126),
// which returns the program's exit code
// (https://github.com/rust-lang/rust/blob/1.98.1/compiler/rustc_codegen_ssa/src/base.rs#L571-L576)
process.exitCode = exports.main(0, 0);
