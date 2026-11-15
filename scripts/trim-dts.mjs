// tsc's declaration printer folds a long parameter list and leaves a space after the last
// comma it keeps on the line. A `.d.ts` carries no string-literal semantics, so stripping
// trailing blanks there is pure formatting; `dist/index.js` is deliberately NOT touched (its
// template literals hold model-visible copy, where a trailing space can be load-bearing).
import { readFileSync, writeFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

let trimmed = 0
for (const name of readdirSync('dist')) {
  if (!name.endsWith('.d.ts')) continue
  const path = join('dist', name)
  const text = readFileSync(path, 'utf-8')
  const next = text.replace(/[ \t]+$/gm, '')
  if (next !== text) {
    writeFileSync(path, next)
    trimmed += text.split('\n').length - next.split('\n').length
  }
}
console.log(`trim-dts: stripped trailing blanks in dist/*.d.ts (files touched: ${trimmed > 0 ? 'yes' : 'none'})`)
