/**
 * hnsw.ts — hierarchical navigable small world graph, the core of the engine.
 * TypeScript port of the algorithm in upstream voyager's `hnswalg.h` (itself
 * following Malkov & Yashunin 2016 + hnswlib):
 *
 *  - randomized level assignment: `floor(-ln(uniform) * mL)`, `mL = 1/ln(M)`
 *  - insert: greedy descent on upper layers, `ef_construction` beam search on
 *    each layer from the new element's level down to 0
 *  - neighbor selection: the heuristic (keep a candidate only when it is
 *    closer to the query than to anything already selected), with
 *    keep-pruned fill-up to M; links are bidirectional with caps of M per
 *    layer and 2*M at layer 0, pruned by the same heuristic
 *  - search: greedy descent from the entry point, then search-layer with
 *    an `ef` beam at the bottom layer
 *
 * Fully deterministic: the level RNG is seeded (mulberry32) and every heap
 * tie-breaks by element id, so identical vectors + seed ⇒ identical graph.
 */

import type { Space } from './space.js'

const MULT_32 = 0x9e3779b9

/** Deterministic seeded PRNG — mulberry32. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Binary min-heap of (dist, id) — ties break on id for full determinism. */
class MinHeap {
  private dist: Float64Array
  private ids: Int32Array
  private n = 0

  constructor(capacity: number) {
    const cap = Math.max(4, capacity)
    this.dist = new Float64Array(cap)
    this.ids = new Int32Array(cap)
  }

  get length(): number {
    return this.n
  }

  push(d: number, id: number): void {
    if (this.n === this.dist.length) this.grow()
    let i = this.n++
    this.dist[i] = d
    this.ids[i] = id
    while (i > 0) {
      const parent = (i - 1) >> 1
      if (this.less(parent, i)) break
      this.swap(parent, i)
      i = parent
    }
  }

  /** Smallest element without removing it (NaN when empty). */
  peek(): { dist: number; id: number } | undefined {
    return this.n === 0 ? undefined : { dist: this.dist[0], id: this.ids[0] }
  }

  pop(): { dist: number; id: number } | undefined {
    if (this.n === 0) return undefined
    const top = { dist: this.dist[0], id: this.ids[0] }
    this.n--
    if (this.n > 0) {
      this.dist[0] = this.dist[this.n]
      this.ids[0] = this.ids[this.n]
      let i = 0
      for (;;) {
        const l = 2 * i + 1
        const r = l + 1
        let smallest = i
        if (l < this.n && this.less(l, smallest)) smallest = l
        if (r < this.n && this.less(r, smallest)) smallest = r
        if (smallest === i) break
        this.swap(i, smallest)
        i = smallest
      }
    }
    return top
  }

  private less(a: number, b: number): boolean {
    const da = this.dist[a]
    const db = this.dist[b]
    return da < db || (da === db && this.ids[a] < this.ids[b])
  }

  private swap(a: number, b: number): void {
    const d = this.dist[a]
    this.dist[a] = this.dist[b]
    this.dist[b] = d
    const i = this.ids[a]
    this.ids[a] = this.ids[b]
    this.ids[b] = i
  }

  private grow(): void {
    const dist = new Float64Array(this.dist.length * 2)
    dist.set(this.dist)
    this.dist = dist
    const ids = new Int32Array(this.ids.length * 2)
    ids.set(this.ids)
    this.ids = ids
  }
}

export interface HnswOptions {
  space: Space
  numDimensions: number
  /** max links per node per layer (2× at layer 0) — upstream voyager's `M` */
  M: number
  /** beam width during construction — upstream voyager's `ef_construction` */
  efConstruction: number
  randomSeed: number
  /** pre-allocation hint; the graph grows by doubling regardless */
  maxElements: number
}

interface Candidate {
  dist: number
  id: number
}

export class Hnsw {
  readonly space: Space
  readonly numDimensions: number
  readonly M: number
  readonly efConstruction: number
  readonly randomSeed: number

  private vecs: Float32Array
  private levels: Int32Array = new Int32Array(0)
  /** links[node][layer] → neighbor ids, unsorted insertion order */
  private links: Array<Array<Int32Array>> = []
  private count = 0
  private entryPoint = -1
  private maxLevel = -1
  private pickLevel: () => number
  private visitMark = new Uint32Array(0)
  private visitEpoch = 0

  constructor(opts: HnswOptions) {
    if (!Number.isInteger(opts.numDimensions) || opts.numDimensions <= 0) {
      throw new Error(`Cassini: num_dimensions must be a positive integer, got ${opts.numDimensions}`)
    }
    if (opts.M < 2) throw new Error('Cassini: M must be >= 2')
    if (opts.efConstruction < opts.M) throw new Error('Cassini: ef_construction must be >= M')
    this.space = opts.space
    this.numDimensions = opts.numDimensions
    this.M = opts.M
    this.efConstruction = opts.efConstruction
    this.randomSeed = opts.randomSeed
    this.vecs = new Float32Array(Math.max(1, opts.maxElements) * opts.numDimensions)
    const mult = 1 / Math.log(opts.M)
    const rng = mulberry32(opts.randomSeed)
    this.pickLevel = () => Math.floor(-Math.log(1 - rng()) * mult)
  }

  get size(): number {
    return this.count
  }

  /** Distance to an inserted element's own vector (used by queryById paths). */
  distToElement(q: Float32Array, id: number): number {
    const dim = this.numDimensions
    const off = id * dim
    const v = this.vecs
    if (this.space === 'euclidean') {
      let s = 0
      for (let i = 0; i < dim; i++) {
        const d = q[i] - v[off + i]
        s += d * d
      }
      return s
    }
    let s = 0
    for (let i = 0; i < dim; i++) s += q[i] * v[off + i]
    return 1 - s
  }

  /** Distance between two stored elements — no view allocation (hot path). */
  private distBetween(a: number, b: number): number {
    const dim = this.numDimensions
    const oa = a * dim
    const ob = b * dim
    const v = this.vecs
    if (this.space === 'euclidean') {
      let s = 0
      for (let i = 0; i < dim; i++) {
        const d = v[oa + i] - v[ob + i]
        s += d * d
      }
      return s
    }
    let s = 0
    for (let i = 0; i < dim; i++) s += v[oa + i] * v[ob + i]
    return 1 - s
  }

  vectorOf(id: number): Float32Array {
    return this.vecs.subarray(id * this.numDimensions, (id + 1) * this.numDimensions)
  }

  linkIds(id: number, layer: number): Int32Array {
    return this.links[id]?.[layer] ?? new Int32Array(0)
  }

  topLevel(id: number): number {
    return this.levels[id] ?? 0
  }

  get top(): { entryPoint: number; level: number } {
    return { entryPoint: this.entryPoint, level: this.maxLevel }
  }

  /** Backing vector storage — only the first `size * numDimensions` are valid. */
  get rawVectors(): Float32Array {
    return this.vecs
  }

  /** Snapshot restore: register an element with prebuilt links (no re-insert). */
  restore(vector: Float32Array, layers: Int32Array[], id: number, entryPoint: number, maxLevel: number): void {
    this.ensureCapacity(id + 1)
    this.vecs.set(vector, id * this.numDimensions)
    this.levels[id] = layers.length - 1
    this.links.push(layers)
    this.count++
    this.pickLevel() // one RNG draw per element, as insert() does, so later inserts continue the original sequence
    this.entryPoint = entryPoint
    this.maxLevel = maxLevel
  }

  /** Insert one vector (already stored by the caller via `store` helpers). */
  insert(vector: ArrayLike<number>): number {
    const id = this.count
    this.ensureCapacity(id + 1)
    const offset = id * this.numDimensions
    for (let i = 0; i < this.numDimensions; i++) this.vecs[offset + i] = vector[i] ?? 0
    const level = this.pickLevel()
    this.levels[id] = level
    this.links.push(new Array(level + 1))
    this.count++

    if (this.entryPoint === -1) {
      this.entryPoint = id
      this.maxLevel = level
      return id
    }

    const q = this.vectorOf(id)
    let cur = this.entryPoint
    let curDist = this.distToElement(q, cur)

    // greedy descent through layers above the new node's level
    for (let lc = this.maxLevel; lc > level; lc--) {
      ;[cur, curDist] = this.greedyDescent(q, cur, curDist, lc)
    }

    // beam-search insert from min(level, maxLevel) down to 0
    let eps: Candidate[] = [{ dist: curDist, id: cur }]
    for (let lc = Math.min(level, this.maxLevel); lc >= 0; lc--) {
      const found = this.searchLayer(q, eps, this.efConstruction, lc)
      const m = lc === 0 ? this.M * 2 : this.M
      const neighbors = this.selectNeighbors(found, this.M)
      for (const { id: n } of neighbors) {
        this.link(n, id, lc, m)
        this.link(id, n, lc, m)
      }
      eps = found
    }

    if (level > this.maxLevel) {
      this.maxLevel = level
      this.entryPoint = id
    }
    return id
  }

  /**
   * k nearest elements to `q` — ids ascending by distance (best first).
   * Correctness over recall shortcuts: `ef` is clamped to at least `k`.
   */
  search(q: Float32Array, k: number, ef: number): Candidate[] {
    if (this.count === 0 || k <= 0) return []
    let cur = this.entryPoint
    let curDist = this.distToElement(q, cur)
    for (let lc = this.maxLevel; lc >= 1; lc--) {
      ;[cur, curDist] = this.greedyDescent(q, cur, curDist, lc)
    }
    const beam = Math.max(ef, k)
    const found = this.searchLayer(q, [{ dist: curDist, id: cur }], beam, 0)
    return found.slice(0, k)
  }

  /* ------------------------------ internals ------------------------------ */

  private ensureCapacity(n: number): void {
    // levels grow on their own schedule — gating them on vector capacity made
    // writes fall off the end of the typed array (silently) for big indexes
    if (n > this.levels.length) {
      const levels = new Int32Array(Math.max(n, this.levels.length * 2))
      levels.set(this.levels)
      this.levels = levels
    }
    const need = n * this.numDimensions
    if (need <= this.vecs.length) return
    let cap = this.vecs.length
    while (cap < need) cap *= 2
    const grown = new Float32Array(cap)
    grown.set(this.vecs)
    this.vecs = grown
  }

  private greedyDescent(q: Float32Array, start: number, startDist: number, layer: number): [number, number] {
    let cur = start
    let curDist = startDist
    for (;;) {
      let improved = false
      const ids = this.linkIds(cur, layer)
      for (let i = 0; i < ids.length; i++) {
        const d = this.distToElement(q, ids[i])
        if (d < curDist) {
          curDist = d
          cur = ids[i]
          improved = true
        }
      }
      if (!improved) return [cur, curDist]
    }
  }

  /**
   * search-layer (Algorithm 2): beam of width `ef` over `layer`,
   * returned ascending by distance.
   */
  private searchLayer(q: Float32Array, eps: Candidate[], ef: number, layer: number): Candidate[] {
    // epoch-stamped visited marks: no per-search Set allocation
    if (this.visitMark.length < this.count) this.visitMark = new Uint32Array(Math.max(this.count, this.visitMark.length * 2))
    if (++this.visitEpoch === 0xffffffff) {
      this.visitMark.fill(0)
      this.visitEpoch = 1
    }
    const mark = this.visitMark
    const epoch = this.visitEpoch
    const candidates = new MinHeap(ef + eps.length)
    const results = new MinHeap(ef + 1) // negated distances → max-heap
    for (const ep of eps) {
      if (mark[ep.id] === epoch) continue
      mark[ep.id] = epoch
      candidates.push(ep.dist, ep.id)
      results.push(-ep.dist, ep.id)
    }

    while (candidates.length > 0) {
      const c = candidates.pop()!
      const worst = results.peek()
      if (worst && -worst.dist < c.dist && results.length >= ef) break
      const ids = this.linkIds(c.id, layer)
      for (let i = 0; i < ids.length; i++) {
        const e = ids[i]
        if (mark[e] === epoch) continue
        mark[e] = epoch
        const d = this.distToElement(q, e)
        const worst2 = results.peek()
        if (!worst2 || results.length < ef || d < -worst2.dist) {
          candidates.push(d, e)
          results.push(-d, e)
          if (results.length > ef) results.pop()
        }
      }
    }

    const out: Candidate[] = []
    while (results.length > 0) {
      const r = results.pop()!
      out.push({ dist: -r.dist, id: r.id })
    }
    out.reverse() // heap pops worst-first; flip to best-first
    return out
  }

  /**
   * Neighbor-selection heuristic (Algorithm 4 + keep-pruned fill): keep a
   * candidate only when it is closer to the base point than to anything
   * already kept; top up from the pruned list when underfull.
   */
  private selectNeighbors(candidates: Candidate[], m: number): Candidate[] {
    const selected: Candidate[] = []
    const pruned: Candidate[] = []
    for (const c of candidates) {
      if (selected.length >= m) break
      let good = true
      for (const s of selected) {
        if (this.distBetween(c.id, s.id) < c.dist) {
          good = false
          break
        }
      }
      if (good) selected.push(c)
      else pruned.push(c)
    }
    for (const c of pruned) {
      if (selected.length >= m) break
      selected.push(c)
    }
    return selected
  }

  /** Add a directed link `from → to`, pruning when over capacity. */
  private link(from: number, to: number, layer: number, cap: number): void {
    const layers = this.links[from]
    const current = layers[layer]
    if (current && current.includes(to)) return
    const grown = new Int32Array((current?.length ?? 0) + 1)
    if (current) grown.set(current)
    grown[grown.length - 1] = to
    layers[layer] = grown
    if (grown.length <= cap) return

    // over capacity — re-select from the full candidate set around `from`
    const fq = this.vectorOf(from)
    const all: Candidate[] = []
    for (let i = 0; i < grown.length; i++) {
      all.push({ dist: this.distToElement(fq, grown[i]), id: grown[i] })
    }
    all.sort((a, b) => a.dist - b.dist || a.id - b.id)
    const kept = this.selectNeighbors(all, cap)
    const next = new Int32Array(kept.length)
    for (let i = 0; i < kept.length; i++) next[i] = kept[i].id
    layers[layer] = next
  }
}

/** FNV-1a seed helper so callers can derive stable seeds from strings. */
export function seedFromString(str: string): number {
  let h = 2166136261
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, MULT_32)
  }
  return h >>> 0
}
