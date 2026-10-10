/**
 * hnsw.test.ts — the raw graph (bun test src): constructor validation, graph
 * invariants (link caps, no self/duplicate/out-of-range links, entry point at
 * the top level), search ordering and ef clamping, growth past maxElements,
 * seeded determinism, and seedFromString.
 */
import { describe, test, expect } from 'bun:test'
import { Hnsw, seedFromString, type HnswOptions } from './hnsw.js'
import { distance } from './space.js'

const DIM = 8
const M = 6

function build(n: number, over: Partial<HnswOptions> = {}): { h: Hnsw; vecs: Float32Array[] } {
  const h = new Hnsw({
    space: 'euclidean',
    numDimensions: DIM,
    M,
    efConstruction: 40,
    randomSeed: 5,
    maxElements: 4, // tiny on purpose: forces the growth path
    ...over,
  })
  let a = 12345
  const rng = () => ((a = (Math.imul(a, 1103515245) + 12345) >>> 0) / 4294967296)
  const vecs = Array.from({ length: n }, () => Float32Array.from({ length: DIM }, () => rng() * 2 - 1))
  for (const v of vecs) h.insert(v)
  return { h, vecs }
}

describe('constructor validation', () => {
  const base: HnswOptions = { space: 'cosine', numDimensions: 4, M: 4, efConstruction: 8, randomSeed: 1, maxElements: 1 }
  test('rejects bad dims, M and efConstruction', () => {
    expect(() => new Hnsw({ ...base, numDimensions: 0 })).toThrow()
    expect(() => new Hnsw({ ...base, numDimensions: 1.5 })).toThrow()
    expect(() => new Hnsw({ ...base, M: 1 })).toThrow()
    expect(() => new Hnsw({ ...base, efConstruction: 3 })).toThrow()
  })
  test('empty graph: size 0, no entry point, search is empty', () => {
    const h = new Hnsw(base)
    expect(h.size).toBe(0)
    expect(h.top).toEqual({ entryPoint: -1, level: -1 })
    expect(h.search(new Float32Array(4), 3, 10)).toEqual([])
  })
})

describe('graph invariants', () => {
  const { h } = build(600)

  test('grows past maxElements and keeps stored vectors intact', () => {
    expect(h.size).toBe(600)
    const { h: g, vecs } = build(50)
    for (let i = 0; i < 50; i++) expect(Array.from(g.vectorOf(i))).toEqual(Array.from(vecs[i]))
  })

  test('links are in range, unique, non-self, and within caps', () => {
    for (let id = 0; id < h.size; id++) {
      for (let lc = 0; lc <= h.topLevel(id); lc++) {
        const ids = Array.from(h.linkIds(id, lc))
        expect(ids.length).toBeLessThanOrEqual(lc === 0 ? 2 * M : M)
        expect(new Set(ids).size).toBe(ids.length)
        for (const n of ids) {
          expect(n).not.toBe(id)
          expect(n).toBeGreaterThanOrEqual(0)
          expect(n).toBeLessThan(h.size)
          // a neighbour at layer lc must itself exist on that layer
          expect(h.topLevel(n)).toBeGreaterThanOrEqual(lc)
        }
      }
    }
  })

  test('entry point sits on the highest level', () => {
    const { entryPoint, level } = h.top
    expect(h.topLevel(entryPoint)).toBe(level)
    for (let id = 0; id < h.size; id++) expect(h.topLevel(id)).toBeLessThanOrEqual(level)
  })

  test('level assignment is geometric: most nodes are layer-0 only', () => {
    let upper = 0
    for (let id = 0; id < h.size; id++) if (h.topLevel(id) > 0) upper++
    // P(level > 0) = 1/M ≈ 17%; allow a wide band
    expect(upper / h.size).toBeGreaterThan(0.05)
    expect(upper / h.size).toBeLessThan(0.35)
  })

  test('every node is reachable from the entry point on layer 0', () => {
    const seen = new Set<number>([h.top.entryPoint])
    const stack = [h.top.entryPoint]
    while (stack.length) {
      for (const n of h.linkIds(stack.pop()!, 0)) {
        if (!seen.has(n)) {
          seen.add(n)
          stack.push(n)
        }
      }
    }
    expect(seen.size).toBe(h.size)
  })
})

describe('search', () => {
  test('results are ascending by distance and distances are correct', () => {
    const { h, vecs } = build(300)
    const q = vecs[17]
    const res = h.search(q, 10, 50)
    expect(res.length).toBe(10)
    for (let i = 1; i < res.length; i++) expect(res[i].dist).toBeGreaterThanOrEqual(res[i - 1].dist)
    for (const r of res) expect(r.dist).toBeCloseTo(distance('euclidean', q, vecs[r.id]), 5)
    expect(res[0].id).toBe(17) // an indexed point finds itself
    expect(res[0].dist).toBe(0)
  })

  test('ef below k is clamped up to k', () => {
    const { h, vecs } = build(200)
    expect(h.search(vecs[0], 15, 1).length).toBe(15)
  })

  test('k larger than size returns every element once; k <= 0 is empty', () => {
    const { h, vecs } = build(20)
    const res = h.search(vecs[0], 100, 100)
    expect(new Set(res.map((r) => r.id)).size).toBe(20)
    expect(h.search(vecs[0], 0, 10)).toEqual([])
    expect(h.search(vecs[0], -3, 10)).toEqual([])
  })

  test('single element is found', () => {
    const { h, vecs } = build(1)
    expect(h.search(vecs[0], 5, 10).map((r) => r.id)).toEqual([0])
  })
})

describe('determinism', () => {
  const links = (h: Hnsw) =>
    Array.from({ length: h.size }, (_, id) =>
      Array.from({ length: h.topLevel(id) + 1 }, (_, lc) => Array.from(h.linkIds(id, lc))),
    )

  test('same seed ⇒ identical graph; different seed ⇒ different levels', () => {
    const a = build(200, { randomSeed: 9 }).h
    const b = build(200, { randomSeed: 9 }).h
    const c = build(200, { randomSeed: 10 }).h
    expect(links(a)).toEqual(links(b))
    expect(links(a)).not.toEqual(links(c))
  })
})

describe('seedFromString', () => {
  test('stable, unsigned 32-bit, input-sensitive', () => {
    expect(seedFromString('my-catalog')).toBe(seedFromString('my-catalog'))
    expect(seedFromString('a')).not.toBe(seedFromString('b'))
    const s = seedFromString('my-catalog')
    expect(Number.isInteger(s) && s >= 0 && s <= 0xffffffff).toBe(true)
  })
})
