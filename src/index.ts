/**
 * index.ts — the Cassini public API, modeled on upstream voyager's
 * `Voyager` class (Python binding):
 *
 *   new Cassini({ space, numDimensions, M, ef_construction, random_seed,
 *                 max_elements, query_ef })
 *   add_item(v, id?) / add_items(vs, ids?) / query(v, k) / query_on_id(id, k)
 *   save/load  →  toSnapshot(bytes) / Cassini.fromSnapshot(bytes)
 *
 * Differences from Voyager (see README.md): pure TypeScript, seeded
 * deterministic builds, and an MMR diversity re-ranker the original lacks.
 */

import { Hnsw, seedFromString } from './hnsw.js'
import { distance, isSpace, normalizeInPlace, type Space } from './space.js'

export { SPACES, type Space } from './space.js'
export { seedFromString } from './hnsw.js'

export interface CassiniOptions {
  space?: Space
  numDimensions: number
  /** max links per node per layer (2× at layer 0) — default 12, as upstream voyager */
  M?: number
  /** construction beam width — default 200, as upstream voyager */
  efConstruction?: number
  /** default 1, as upstream voyager */
  randomSeed?: number
  /** allocation hint only — the index grows as needed */
  maxElements?: number
  /** beam width at query time — default -1 ⇒ reuse efConstruction, as upstream voyager */
  queryEf?: number
}

export interface QueryOptions {
  /** override the instance queryEf for this query */
  queryEf?: number
}

export interface QueryResult {
  /** internal element ids (positions in insertion order) */
  ids: number[]
  /** distances, smaller = closer (squared L2 for 'euclidean') */
  distances: number[]
}

const SNAPSHOT_MAGIC = 0x314e_5343 // "CSN1" little-endian
const SNAPSHOT_VERSION = 2 // v2 adds the random seed; v1 (40-byte header, no seed) still loads
const SNAPSHOT_HEADER = 44
const SNAPSHOT_HEADER_V1 = 40
const MAX_LEVELS = 64 // a real graph has ~log_M(n) layers; anything near this is corrupt
const M_DEFAULT = 12
const EF_DEFAULT = 200

export class Cassini {
  private h: Hnsw
  private queryEfDefault: number

  constructor(opts: CassiniOptions) {
    const m = opts.M ?? M_DEFAULT
    const ef = opts.efConstruction ?? EF_DEFAULT
    this.h = new Hnsw({
      space: opts.space ?? 'cosine',
      numDimensions: opts.numDimensions,
      M: m,
      efConstruction: ef,
      randomSeed: opts.randomSeed ?? 1,
      maxElements: opts.maxElements ?? 128,
    })
    this.queryEfDefault = opts.queryEf ?? -1
  }

  get space(): Space {
    return this.h.space
  }

  get numDimensions(): number {
    return this.h.numDimensions
  }

  /** Number of elements (`len(index)`; upstream voyager uses `len(voyager)`). */
  get size(): number {
    return this.h.size
  }

  /**
   * Add a single vector; returns its element id (sequential from 0 when no
   * explicit ids are used across the index). Cosine space normalizes the copy.
   */
  addItem(vector: ArrayLike<number>): number {
    return this.h.insert(this.prepare(vector))
  }

  /** Add a batch; returns the first id added. */
  addItems(vectors: ArrayLike<ArrayLike<number>>): number {
    const first = this.size
    for (let i = 0; i < vectors.length; i++) this.addItem(vectors[i])
    return first
  }

  /** k nearest neighbours of a query vector, best first. */
  query(vector: ArrayLike<number>, k = 10, opts?: QueryOptions): QueryResult {
    const q = this.prepare(vector)
    return this.runQuery(q, k, opts)
  }

  /** k nearest neighbours of an indexed element (self excluded). */
  queryById(elementId: number, k = 10, opts?: QueryOptions): QueryResult {
    if (elementId < 0 || elementId >= this.size) {
      throw new Error(`Cassini: element id ${elementId} out of range (size ${this.size})`)
    }
    return this.runQuery(this.h.vectorOf(elementId), k, opts, elementId)
  }

  /** The stored (normalized in cosine space) vector of an element. */
  vectorOf(elementId: number): Float32Array {
    return this.h.vectorOf(elementId)
  }

  /**
   * Serialize the whole index (graph + vectors + params) — the counterpart of
   * upstream voyager's `get_serialized_index` / `save`. Round-trips through
   * `Cassini.fromSnapshot`.
   *
   * Layout (little-endian): header (44 bytes: magic "CSN1", format version,
   * space, dims, M, ef_construction, count, entry point, max level,
   * query_ef, random seed; 3 pad), then `count × dims` float32 vectors, then per element
   * [level, [len, ids…] per layer] as uint32 words.
   */
  toSnapshot(): Uint8Array {
    const h = this.h
    const n = h.size
    const dim = this.numDimensions
    const SPACE_IDS: readonly Space[] = ['cosine', 'euclidean', 'dot']

    // encode the graph as uint32 words, one Uint32Array per element
    const perNode = new Array<Uint32Array>(n)
    let graphWords = 0
    for (let id = 0; id < n; id++) {
      const levels = h.topLevel(id) + 1
      const words: number[] = [levels]
      for (let lc = 0; lc < levels; lc++) {
        const ids = h.linkIds(id, lc)
        words.push(ids.length)
        for (let i = 0; i < ids.length; i++) words.push(ids[i])
      }
      perNode[id] = Uint32Array.from(words)
      graphWords += words.length
    }

    const HEADER = SNAPSHOT_HEADER
    const bytes = new Uint8Array(HEADER + n * dim * 4 + graphWords * 4)
    const view = new DataView(bytes.buffer)
    view.setUint32(0, SNAPSHOT_MAGIC, true)
    view.setUint32(4, SNAPSHOT_VERSION, true)
    view.setUint8(8, SPACE_IDS.indexOf(h.space))
    view.setUint32(12, dim, true)
    view.setUint32(16, h.M, true)
    view.setUint32(20, h.efConstruction, true)
    view.setUint32(24, n, true)
    view.setInt32(28, h.top.entryPoint, true)
    view.setInt32(32, h.top.level, true)
    view.setInt32(36, this.queryEfDefault, true)
    view.setUint32(40, h.randomSeed, true)

    bytes.set(new Uint8Array(h.rawVectors.buffer, 0, n * dim * 4), HEADER)
    let off = HEADER + n * dim * 4
    for (let id = 0; id < n; id++) {
      const node = perNode[id]
      bytes.set(new Uint8Array(node.buffer, node.byteOffset, node.byteLength), off)
      off += node.byteLength
    }
    return bytes
  }

  /** Throws on malformed or truncated bytes rather than reading past the end. */
  static fromSnapshot(bytes: Uint8Array, spaceOverride?: Space): Cassini {
    const bad = (why: string) => new Error(`Cassini: invalid snapshot (${why})`)
    if (bytes.byteLength < SNAPSHOT_HEADER_V1) throw bad('too short')
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    if (view.getUint32(0, true) !== SNAPSHOT_MAGIC) throw new Error('Cassini: not a Cassini snapshot')
    const version = view.getUint32(4, true)
    if (version !== 1 && version !== SNAPSHOT_VERSION) throw bad(`unsupported version ${version}`)
    const headerSize = version === 1 ? SNAPSHOT_HEADER_V1 : SNAPSHOT_HEADER
    if (bytes.byteLength < headerSize) throw bad('too short')
    const SPACE_IDS: readonly Space[] = ['cosine', 'euclidean', 'dot']
    const space = spaceOverride ?? SPACE_IDS[view.getUint8(8)]
    if (!isSpace(space)) throw new Error('Cassini: unknown space in snapshot')
    const numDimensions = view.getUint32(12, true)
    const m = view.getUint32(16, true)
    const efConstruction = view.getUint32(20, true)
    const n = view.getUint32(24, true)
    const entryPoint = view.getInt32(28, true)
    const maxLevel = view.getInt32(32, true)
    const queryEf = view.getInt32(36, true)
    const randomSeed = version === 1 ? 1 : view.getUint32(40, true)

    // check sizes before allocating anything the header controls
    const vecBytes = n * numDimensions * 4
    if (!Number.isSafeInteger(vecBytes) || headerSize + vecBytes > bytes.byteLength) throw bad('truncated vectors')
    if (n === 0 ? entryPoint !== -1 : entryPoint < 0 || entryPoint >= n) throw bad('entry point out of range')

    const idx = new Cassini({ space, numDimensions, M: m, efConstruction, queryEf, randomSeed })
    let off = headerSize
    const vecs = new Float32Array(bytes.buffer.slice(bytes.byteOffset + off, bytes.byteOffset + off + vecBytes))
    off += vecBytes

    const readWord = () => {
      if (off + 4 > bytes.byteLength) throw bad('truncated graph')
      const w = view.getUint32(off, true)
      off += 4
      return w
    }
    for (let id = 0; id < n; id++) {
      const levels = readWord()
      if (levels < 1 || levels > MAX_LEVELS) throw bad(`element ${id} has ${levels} layers`)
      const layers: Int32Array[] = new Array(levels)
      for (let lc = 0; lc < levels; lc++) {
        const len = readWord()
        if (off + len * 4 > bytes.byteLength) throw bad('truncated graph')
        const ids = new Int32Array(len)
        for (let i = 0; i < len; i++) {
          const nb = readWord()
          if (nb >= n || nb === id) throw bad(`element ${id} links to ${nb}`)
          ids[i] = nb
        }
        layers[lc] = ids
      }
      idx.restoreElement(vecs.subarray(id * numDimensions, (id + 1) * numDimensions), layers, entryPoint, maxLevel)
    }
    if (off !== bytes.byteLength) throw bad('trailing bytes')
    if (n > 0 && idx.h.topLevel(entryPoint) !== maxLevel) throw bad('entry point is not on the top level')
    return idx
  }

  /* ------------------------------ internals ------------------------------ */

  private prepare(vector: ArrayLike<number>): Float32Array {
    if (vector.length !== this.numDimensions) {
      throw new Error(`Cassini: expected ${this.numDimensions} dimensions, got ${vector.length}`)
    }
    const v = new Float32Array(this.numDimensions)
    for (let i = 0; i < v.length; i++) {
      const x = vector[i]
      v[i] = Number.isFinite(x) ? x : 0
    }
    if (this.space === 'cosine') normalizeInPlace(v)
    return v
  }

  private runQuery(q: Float32Array, k: number, opts: QueryOptions | undefined, exclude?: number): QueryResult {
    const requested = Math.min(k, this.size)
    const ef = opts?.queryEf ?? (this.queryEfDefault === -1 ? this.h.efConstruction : this.queryEfDefault)
    const found = this.h.search(q, requested + (exclude !== undefined ? 1 : 0), ef)
    const ids: number[] = []
    const distances: number[] = []
    for (const c of found) {
      if (exclude !== undefined && c.id === exclude) continue
      ids.push(c.id)
      distances.push(c.dist)
      if (ids.length >= requested) break
    }
    return { ids, distances }
  }

  /** Snapshot-only: register an already-stored element with prebuilt links. */
  private restoreElement(
    vector: Float32Array,
    layers: Int32Array[],
    entryPoint: number,
    maxLevel: number,
  ): void {
    const id = this.h.size
    this.h.restore(vector, layers, id, entryPoint, maxLevel)
  }
}

/* ------------------------- diversity re-ranking ------------------------- */

export interface MmrEntry<T> {
  item: T
  /** relevance of this candidate, higher = better (e.g. 1 - ANN distance) */
  relevance: number
  /** L2-normalized vector used for the redundancy term */
  vector: ArrayLike<number>
}

function sim(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    s += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : s / denom
}

/**
 * Maximal Marginal Relevance re-rank — the tailored addition the original
 * leaves to the caller: trade a little relevance for a mix that doesn't
 * clump around one near-duplicate. `lambda` 1 = pure relevance,
 * 0 = pure diversity.
 */
export function mmrRerank<T>(entries: MmrEntry<T>[], k: number, lambda = 0.7): T[] {
  const picked: MmrEntry<T>[] = []
  const pool = [...entries]
  while (picked.length < k && pool.length > 0) {
    let bestIdx = 0
    let bestScore = -Infinity
    for (let i = 0; i < pool.length; i++) {
      const c = pool[i]
      let maxSim = 0
      for (const p of picked) {
        const s = sim(c.vector, p.vector)
        if (s > maxSim) maxSim = s
      }
      const score = lambda * c.relevance - (1 - lambda) * maxSim
      if (score > bestScore) {
        bestScore = score
        bestIdx = i
      }
    }
    picked.push(pool.splice(bestIdx, 1)[0])
  }
  return picked.map((p) => p.item)
}

/** Convenience: distance helper exposed for callers re-scoring results. */
export function resultDistance(space: Space, a: Float32Array, b: Float32Array): number {
  return distance(space, a, b)
}

/** Derive a stable index seed from any string (e.g. a catalog id). */
export function seedFor(name: string): number {
  return seedFromString(name) || 1
}
