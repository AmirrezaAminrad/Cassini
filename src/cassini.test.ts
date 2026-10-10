/**
 * cassini.test.ts — engine contracts (bun test src): recall vs brute force on
 * all three spaces, deterministic builds under seed, snapshot round-trip,
 * queryById self-exclusion and clamping, degenerate inputs, and the MMR
 * diversity re-ranker.
 */
import { describe, test, expect } from 'bun:test'
import { Cassini, mmrRerank, type MmrEntry } from './index.js'
import { distance, normalizeInPlace, type Space } from './space.js'

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function randomVectors(seed: number, n: number, dim: number): Float32Array[] {
  const rng = mulberry32(seed)
  return Array.from({ length: n }, () => {
    const v = new Float32Array(dim)
    for (let i = 0; i < dim; i++) v[i] = rng() * 2 - 1
    return v
  })
}

/** The index normalizes in cosine space — brute force must see the same data. */
function asIndexSees(space: Space, vectors: Float32Array[]): Float32Array[] {
  if (space !== 'cosine') return vectors
  return vectors.map((v) => {
    const copy = Float32Array.from(v)
    normalizeInPlace(copy)
    return copy
  })
}

function bruteForce(space: Space, vectors: Float32Array[], q: Float32Array, k: number): number[] {
  return vectors
    .map((v, id) => ({ id, d: distance(space, Float32Array.from(q), v) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, k)
    .map((c) => c.id)
}

const N = 1200
const DIM = 24
const K = 10
const QUERIES = 30

describe('recall vs brute force', () => {
  for (const space of ['cosine', 'euclidean', 'dot'] as Space[]) {
    test(`${space}: avg recall@${K} >= 0.9 over ${N} vectors`, () => {
      const vectors = asIndexSees(space, randomVectors(42, N, DIM))
      const idx = new Cassini({ space, numDimensions: DIM })
      idx.addItems(vectors)
      expect(idx.size).toBe(N)
      const queries = asIndexSees(space, randomVectors(7, QUERIES, DIM))
      let hits = 0
      for (const q of queries) {
        const truth = new Set(bruteForce(space, vectors, q, K))
        const got = new Set(idx.query(q, K).ids)
        for (const id of got) if (truth.has(id)) hits++
      }
      const recall = hits / (QUERIES * K)
      expect(recall).toBeGreaterThanOrEqual(0.9)
    })
  }
})

describe('determinism', () => {
  test('same seed builds answer identically', () => {
    const vectors = randomVectors(3, 400, 16)
    const a = new Cassini({ space: 'cosine', numDimensions: 16, randomSeed: 99 })
    const b = new Cassini({ space: 'cosine', numDimensions: 16, randomSeed: 99 })
    a.addItems(vectors)
    b.addItems(vectors)
    const q = randomVectors(5, 1, 16)[0]
    expect(a.query(q, 10)).toEqual(b.query(q, 10))
  })
})

describe('snapshot round-trip', () => {
  test('restored index answers identically', () => {
    const vectors = randomVectors(11, 500, 20)
    const original = new Cassini({ space: 'euclidean', numDimensions: 20, randomSeed: 2 })
    original.addItems(vectors)
    const restored = Cassini.fromSnapshot(original.toSnapshot())
    expect(restored.size).toBe(original.size)
    for (const q of randomVectors(13, 5, 20)) {
      expect(restored.query(q, 10)).toEqual(original.query(q, 10))
    }
  })
  test('rejects foreign bytes', () => {
    expect(() => Cassini.fromSnapshot(new Uint8Array(64))).toThrow()
  })
})

describe('query semantics', () => {
  test('addItem returns sequential ids; queryById excludes itself', () => {
    const idx = new Cassini({ numDimensions: 3 })
    for (const v of [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0], [0, 0, 1]]) idx.addItem(v)
    expect(idx.size).toBe(4)
    const res = idx.queryById(0, 3)
    expect(res.ids).not.toContain(0)
    expect(res.ids.length).toBe(3)
    expect(res.ids[0]).toBe(2) // [0.9, 0.1, 0] is id 0's nearest
  })

  test('cosine normalizes: scale-invariant', () => {
    const idx = new Cassini({ space: 'cosine', numDimensions: 2 })
    const id = idx.addItem([3, 4])
    const res = idx.query([30, 40], 1)
    expect(res.ids[0]).toBe(id)
    expect(res.distances[0]).toBeCloseTo(0, 5)
  })

  test('k above size clamps; empty index answers empty', () => {
    const idx = new Cassini({ numDimensions: 4 })
    expect(idx.query([1, 0, 0, 0], 5).ids).toEqual([])
    idx.addItem([1, 0, 0, 0])
    expect(idx.query([1, 0, 0, 0], 5).ids.length).toBe(1)
  })

  test('duplicate vectors are handled without NaN', () => {
    const idx = new Cassini({ numDimensions: 2 })
    for (let i = 0; i < 50; i++) idx.addItem([1, 1])
    const res = idx.query([1, 1], 10)
    for (const d of res.distances) expect(Number.isFinite(d)).toBe(true)
    expect(res.ids.length).toBe(10)
  })

  test('non-finite input components are zeroed, wrong dims rejected', () => {
    const idx = new Cassini({ numDimensions: 2 })
    expect(() => idx.addItem([1])).toThrow()
    idx.addItem([NaN, 1])
    expect(idx.query([0, 1], 1).distances[0]).toBeCloseTo(0, 5)
  })

  test('queryById range is validated', () => {
    const idx = new Cassini({ numDimensions: 2 })
    expect(() => idx.queryById(0, 1)).toThrow()
  })

  test('invalid constructor options are rejected', () => {
    expect(() => new Cassini({ numDimensions: 0 })).toThrow()
    expect(() => new Cassini({ numDimensions: 8, M: 1 })).toThrow()
  })
})

describe('mmrRerank', () => {
  const entry = (item: string, vector: number[]): MmrEntry<string> => ({
    item,
    relevance: 1,
    vector,
  })
  test('diversifies away from a near-duplicate clump', () => {
    const entries = [
      entry('a1', [1, 0]),
      entry('a2', [0.99, 0.01]),
      entry('a3', [0.98, 0.02]),
      entry('b1', [0, 1]),
    ]
    const top = mmrRerank(entries, 2, 0.5)
    expect(top).toContain('a1')
    expect(top).toContain('b1') // diversity term wins over redundant a2/a3
  })
  test('lambda 1 degrades to pure relevance order', () => {
    const entries: MmrEntry<string>[] = [
      { item: 'x', relevance: 0.9, vector: [1, 0] },
      { item: 'y', relevance: 0.8, vector: [1, 0] },
      { item: 'z', relevance: 0.7, vector: [1, 0] },
    ]
    expect(mmrRerank(entries, 3, 1)).toEqual(['x', 'y', 'z'])
  })
  test('k above pool size returns everything', () => {
    expect(mmrRerank([entry('a', [1])], 5, 0.7)).toEqual(['a'])
  })
})

describe('snapshot seed and validation', () => {
  const build = () => {
    const idx = new Cassini({ space: 'euclidean', numDimensions: 6, randomSeed: 77 })
    idx.addItems(randomVectors(21, 150, 6))
    return idx
  }

  test('inserts after a restore match a never-saved build', () => {
    const extra = randomVectors(22, 40, 6)
    const direct = build()
    const restored = Cassini.fromSnapshot(build().toSnapshot())
    direct.addItems(extra)
    restored.addItems(extra)
    expect(restored.toSnapshot()).toEqual(direct.toSnapshot())
  })

  test('v1 snapshots (no seed) still load', () => {
    const v2 = build().toSnapshot()
    const v1 = new Uint8Array(v2.length - 4)
    v1.set(v2.subarray(0, 40))
    v1.set(v2.subarray(44), 40)
    new DataView(v1.buffer).setUint32(4, 1, true)
    const restored = Cassini.fromSnapshot(v1)
    expect(restored.size).toBe(150)
    const q = randomVectors(23, 1, 6)[0]
    expect(restored.query(q, 5)).toEqual(build().query(q, 5))
  })

  test('truncated, padded, and corrupted snapshots throw cleanly', () => {
    const good = build().toSnapshot()
    expect(() => Cassini.fromSnapshot(good.subarray(0, 20))).toThrow(/invalid snapshot/)
    expect(() => Cassini.fromSnapshot(good.subarray(0, good.length - 7))).toThrow(/invalid snapshot/)
    expect(() => Cassini.fromSnapshot(new Uint8Array([...good, 0, 0, 0, 0]))).toThrow(/trailing/)
    const view = (b: Uint8Array) => new DataView(b.buffer)
    const hugeCount = Uint8Array.from(good)
    view(hugeCount).setUint32(24, 0xffffffff, true)
    expect(() => Cassini.fromSnapshot(hugeCount)).toThrow(/invalid snapshot/)
    const badVersion = Uint8Array.from(good)
    view(badVersion).setUint32(4, 9, true)
    expect(() => Cassini.fromSnapshot(badVersion)).toThrow(/version/)
    const badEntry = Uint8Array.from(good)
    view(badEntry).setInt32(28, 5000, true)
    expect(() => Cassini.fromSnapshot(badEntry)).toThrow(/entry point/)
  })
})
