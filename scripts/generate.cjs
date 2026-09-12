'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const v8 = require('node:v8');
const Module = require('node:module');
const mode = process.argv[2];
if (mode === 'eager') v8.setFlagsFromString('--no-lazy --no-flush-bytecode');
if (process.versions.electron !== '42.2.0' || process.versions.v8 !== '14.8.178.14-electron.0') {
  throw new Error(`Unexpected runtime: ${JSON.stringify(process.versions)}`);
}
const corpus = {
  minimal: '1 + 2;',
  nested: 'function outer(a) { const local = "hello"; return function inner(b) { return a + b + local; }; } outer(1)(2);',
  objects: 'class Base { m(x) { return x + 1; } } class Child extends Base { #v = 42; m(x) { return super.m(x) + this.#v; } } const data = { list: [1, 2, 3], fn: x => /a+b/gi.test(x), value: 12345678901234567890n }; async function af(x) { return await Promise.resolve(x); } function* gen() { yield* [1, 2]; }',
  wide: `const values = [${Array.from({length: 1500}, (_, i) => JSON.stringify(`constant-${i}`)).join(',')}]; function pick(i) { return values[i]; }`,
  functions: Array.from({length: 1200}, (_, i) => `function f${i}(x) { const s = "constant-${i}"; return function g${i}(y) { return x + y + s; }; }`).join('\n'),
  bundle: `(function(modules) { globalThis.syntheticModules = modules; })({${Array.from({length: 450}, (_, i) => `${i}: function(module, exports, require) { class C${i} { #n = ${i}; run(x) { try { return x?.value ?? this.#n; } catch (e) { return e.message; } } } exports.create = x => new C${i}(x); exports.strings = ["module-${i}", "name-${i}"]; exports.task = async x => await Promise.resolve(x); }`).join(',\n')}});`,
  unicode: `const message = ${JSON.stringify('你好 🌍\n'.repeat(12000))}; function read() { return message; }`,
};
const output = path.resolve('evidence', 'fixtures');
fs.mkdirSync(output, {recursive: true});
const records = [];
for (const [name, source] of Object.entries(corpus)) {
  for (const wrapped of [false, true]) {
    const code = wrapped ? Module.wrap(source) : source;
    const label = `${mode}-${name}-${wrapped ? 'commonjs' : 'script'}`;
    const filename = `${label}.js`;
    const compiled = new vm.Script(code, {filename, produceCachedData: true});
    const cache = compiled.createCachedData();
    const checked = new vm.Script(code, {filename, cachedData: cache});
    if (checked.cachedDataRejected) throw new Error(`${label}: own cache rejected`);
    fs.writeFileSync(path.join(output, `${label}.jsc`), cache);
    fs.writeFileSync(path.join(output, filename), code);
    records.push({label, bytes: cache.length, sourceLength: code.length, acceptedByElectron: true, header: cache.subarray(0, 32).toString('hex')});
  }
}
fs.writeFileSync(path.resolve('evidence', `generator-${mode}.json`), JSON.stringify({versions: process.versions, flags: mode, records}, null, 2));
console.log(`Generated ${records.length} ${mode} caches using Electron ${process.versions.electron} / V8 ${process.versions.v8}`);
