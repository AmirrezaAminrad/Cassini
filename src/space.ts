/**
 * space.ts — distance metrics for the Cassini index, mirroring
 * upstream voyager's `Space.Cosine` / `Space.Euclidean` / `Space.Dot`.
 * (The original library also ships HIK for histogram data; this app's
 * vectors are dense and L2-normalized, so the three general-purpose spaces
 * cover it.)
 *
 * Like hnswlib and upstream voyager, every metric reports SMALLER = CLOSER, and euclidean
 * returns the squared L2 norm (not rooted) — keep that in mind when reading
 * `distances` out of a query.
 */

export type Space = 'cosine' | 'euclidean' | 'dot'

export const SPACES: readonly Space[] = ['cosine', 'euclidean', 'dot']

export function isSpace(value: unknown): value is Space {
  return typeof value === 'string' && (SPACES as readonly string[]).includes(value)
}

/** L2-normalize in place — cosine space stores unit vectors (upstream voyager does the same). */
export function normalizeInPlace(v: Float32Array): void {
  let sum = 0
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i]
  const norm = Math.sqrt(sum)
  if (norm === 0 || !Number.isFinite(norm)) return
  for (let i = 0; i < v.length; i++) v[i] /= norm
}

/** Inner product of two equal-length vectors. */
export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

/** Distance from `a` to `b` in the given space. Smaller = closer. */
export function distance(space: Space, a: Float32Array, b: Float32Array): number {
  if (space === 'euclidean') {
    let s = 0
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i]
      s += d * d
    }
    return s
  }
  // cosine (a pre-normalized) and dot both order by the inner product
  return 1 - dot(a, b)
}
