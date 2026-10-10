// Runs under plain Node against the built package: proves the npm artifact works without Bun.
import assert from 'node:assert/strict'
import { Cassini, mmrRerank } from '../dist/index.js'

const idx = new Cassini({ space: 'euclidean', numDimensions: 2 })
idx.addItems([[0, 0], [1, 0], [5, 5]])
assert.deepEqual(idx.query([0.9, 0], 2).ids, [1, 0])
const copy = Cassini.fromSnapshot(idx.toSnapshot())
assert.deepEqual(copy.query([5, 5], 1).ids, [2])
assert.deepEqual(mmrRerank([{ item: 'a', relevance: 1, vector: [1, 0] }], 1), ['a'])
console.log('smoke ok')
