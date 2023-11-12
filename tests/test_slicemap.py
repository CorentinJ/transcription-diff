import unittest

import numpy as np

from transcription_diff.slice_map import SliceMap


class TestSliceMap(unittest.TestCase):
    def test_constructor(self):
        # Slices cannot have negative indices and cannot index beyond the size of Y
        with self.assertRaises(AssertionError):
            SliceMap([slice(0, -1)], 10)
        with self.assertRaises(AssertionError):
            SliceMap([slice(0, 1), slice(1, 2)], 1)

        # The slice starts cannot decrease, the same goes for the slice stops. You can however have consecutive
        # overlapping slices, e.g. [slice(0, 2), slice(0, 2)]
        with self.assertRaises(AssertionError):
            SliceMap([slice(1, 1), slice(0, 2)], 2)
        with self.assertRaises(AssertionError):
            SliceMap([slice(0, 2), slice(0, 1)], 2)
        SliceMap([slice(0, 2), slice(0, 2)], 2)

        # Slices can be empty (stop <= start)
        SliceMap([slice(2, 1)], 2)

        # Map to nothing
        SliceMap([slice(0, 0), slice(0, 0)], 0)

        # Map from nothing
        SliceMap([], 2)

    def test_getitem(self):
        # X[i] mapping to Y[smap[i]]
        m = SliceMap([slice(0, 1), slice(1, 2)], 2)
        self.assertEqual(m[1], slice(1, 2))

        # X[i:j] mapping to Y[smap[i].start:smap[j - 1].stop]
        m = SliceMap([slice(0, 1), slice(0, 2), slice(3, 3)], 3)
        self.assertEqual(m[:1], slice(0, 1))
        self.assertEqual(m[:2], slice(0, 2))
        self.assertEqual(m[1:2], slice(0, 2))
        self.assertEqual(m[1:3], slice(0, 3))

        # Other tests
        m = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2), slice(3, 4)], 5)
        seq = "abcd"

        # Step != 1 is not supported
        with self.assertRaises(AssertionError):
            m[1:2:2]

        # Index with an int is equivalent to slice of size 1
        for i in range(m.source_len):
            self.assertEqual(m[i], m[i:i + 1])

        # Index that maps to nothing
        self.assertEqual(seq[m[1]], "")

        # Slicing includes gaps
        self.assertEqual(seq[m[2:]], "bcd")

        # Empty slices
        for i in range(m.source_len + 1):
            self.assertEqual(seq[m[i:i]], "")

        # Slices beyond the size of the map
        self.assertEqual(seq[m[10:20]], "")

    def test_project(self):
        # Length must match
        m = SliceMap([slice(0, 1), slice(1, 2), slice(2, 3)], 3)
        with self.assertRaises(AssertionError):
            m.project(list("ab"))
        with self.assertRaises(AssertionError):
            m.project(list("abcd"))

        # Empty
        m = SliceMap.empty()
        self.assertEqual(m.project([]), [])

        # Map to nothing
        m = SliceMap([slice(0, 0), slice(0, 0)], 0)
        self.assertEqual(m.project(list("ab")), [])

        # Map from nothing
        m = SliceMap([], 2)
        self.assertEqual(m.project([], "?"), list("??"))

        # Identity
        m = SliceMap([slice(0, 1), slice(1, 2), slice(2, 3)], 3)
        self.assertEqual(m.project(list("abc")), list("abc"))

        # Spanning multiple indices
        m = SliceMap([slice(0, 1), slice(1, 3), slice(3, 4)], 4)
        self.assertEqual(m.project(list("abc")), list("abbc"))

        # Gap in the target space
        m = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2)], 2)
        self.assertEqual(m.project(list("abc")), list("ac"))

        # Gap in the source space
        m = SliceMap([slice(0, 1), slice(2, 3)], 3)
        self.assertEqual(m.project(list("ab"), "?"), list("a?b"))

        # Overlap
        m = SliceMap([slice(0, 1), slice(0, 1)], 1)
        self.assertEqual(m.project(list("ab")), list("b"))

        # Overlap that spans multiple characters
        m = SliceMap([slice(0, 2), slice(0, 2)], 2)
        self.assertEqual(m.project(list("ab")), list("bb"))

        # Overlap with step
        m = SliceMap([slice(0, 2), slice(1, 3)], 3)
        self.assertEqual(m.project(list("ab")), list("abb"))

        # Composition of the above
        m = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2), slice(3, 5), slice(4, 6)], 7)
        self.assertEqual(m.project(list("abcde"), "?"), list("ac?dee?"))

    def test_inverse(self):
        # Empty
        self.assertEqual(SliceMap.empty(), SliceMap.empty().inverse())

        # Map to nothing
        m = SliceMap([slice(0, 0), slice(0, 0)], 0)
        self.assertEqual(m, m.inverse().inverse())

        # Map from nothing
        m = SliceMap([], 2)
        self.assertEqual(m, m.inverse().inverse())

        # Identity
        m = SliceMap([slice(0, 1), slice(1, 2), slice(2, 3)], 3)
        self.assertEqual(m.inverse().project(list("abc")), list("abc"))
        self.assertEqual(m, m.inverse().inverse())

        # Spanning multiple indices
        m = SliceMap([slice(0, 1), slice(1, 3), slice(3, 4)], 4)
        self.assertEqual(m.inverse().project(list("abbc")), list("abc"))
        self.assertEqual(m, m.inverse().inverse())

        # Gap in the target space
        m = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2)], 2)
        self.assertEqual(m.inverse().project(list("ac"), "?"), list("a?c"))
        self.assertEqual(m, m.inverse().inverse())

        # Gap in the source space
        m = SliceMap([slice(0, 1), slice(2, 3)], 3)
        self.assertEqual(m.inverse().project(list("a?b")), list("ab"))
        self.assertEqual(m, m.inverse().inverse())

        # Composition of the above
        m = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2), slice(3, 4)], 5)
        self.assertEqual(m.inverse().project(list("ac?d?"), "?"), list("a?cd"))
        self.assertEqual(m, m.inverse().inverse())

        # Overlap
        m = SliceMap([slice(0, 1), slice(0, 1)], 1)
        self.assertEqual(m.inverse().project(list("a")), list("aa"))
        self.assertEqual(m.inverse(), SliceMap([slice(0, 2)], 2))
        self.assertEqual(m, m.inverse().inverse())

        # Overlap that spans multiple characters
        m = SliceMap([slice(0, 2), slice(0, 2)], 2)
        self.assertEqual(m.inverse().project(list("ab")), list("bb"))
        self.assertEqual(m.inverse(), SliceMap([slice(0, 2), slice(0, 2)], 2))
        self.assertEqual(m, m.inverse().inverse())

        # Overlap with step
        m = SliceMap([slice(0, 2), slice(1, 3)], 3)
        self.assertEqual(m.inverse().project(list("abc")), list("bc"))
        self.assertEqual(m.inverse(), SliceMap([slice(0, 1), slice(0, 2), slice(1, 2)], 2))
        self.assertEqual(m, m.inverse().inverse())

        # Composition of the above
        m = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2), slice(3, 5), slice(4, 6)], 7)
        self.assertEqual(m.inverse().project(list("abcdefg"), "?"), list("a?bef"))
        self.assertEqual(m.inverse(), SliceMap(
            [slice(0, 1), slice(2, 3), slice(3, 3), slice(3, 4), slice(3, 5), slice(4, 5), slice(5, 5)], 5
        ))
        self.assertEqual(m, m.inverse().inverse())

    def test_compose(self):
        # Empty
        self.assertEqual(SliceMap.empty() * SliceMap.empty(), SliceMap.empty())

        # Map to nothing
        m1 = SliceMap([slice(0, 1), slice(1, 2)], 2)
        m2 = SliceMap([slice(0, 0), slice(0, 0)], 0)
        self.assertEqual(m1 * m2, m2)

        # Map from nothing
        m1 = SliceMap([], 2)
        m2 = SliceMap([slice(0, 1), slice(1, 2)], 2)
        self.assertEqual(m1 * m2, m1)

        # Identity
        m = SliceMap([slice(0, 1), slice(1, 2), slice(2, 3)], 3)
        self.assertEqual(m * m, m)

        # Gap in the source space, gap in the source space
        m1 = SliceMap([slice(0, 1), slice(2, 3)], 3)
        m2 = SliceMap([slice(1, 2), slice(2, 3), slice(3, 4)], 4)
        self.assertEqual((m1 * m2).project(list("ab"), "?"), list("?a?b"))

        # Gap in the source space, gap in the target space
        m1 = SliceMap([slice(0, 1), slice(2, 3)], 3)
        m2 = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2)], 2)
        self.assertEqual((m1 * m2).project(list("ab"), "?"), list("ab"))

        # Gap in the target space, gap in the target space
        m1 = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2)], 2)
        m2 = SliceMap([slice(0, 0), slice(0, 1)], 1)
        self.assertEqual((m1 * m2).project(list("abc"), "?"), list("c"))

        # Gap in the target space, gap in the source space
        m1 = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2)], 2)
        m2 = SliceMap([slice(0, 1), slice(2, 3)], 3)
        self.assertEqual((m1 * m2).project(list("abc"), "?"), list("a?c"))

        # Gap in the source space, overlap
        m1 = SliceMap([slice(0, 1), slice(2, 3)], 3)
        m2 = SliceMap([slice(0, 2), slice(0, 2), slice(0, 2)], 2)
        self.assertEqual((m1 * m2).project(list("ac"), "?"), list("cc"))

        # Gap in the target space, overlap
        m1 = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2)], 2)
        m2 = SliceMap([slice(0, 2), slice(0, 2)], 2)
        self.assertEqual((m1 * m2).project(list("abc"), "?"), list("cc"))

        # Gap at the start and end of the source space
        m = SliceMap([slice(0, 1), slice(1, 1)], 1)
        self.assertEqual(m * SliceMap.lerp(1, 1), m)
        m = SliceMap([slice(0, 0), slice(0, 1)], 1)
        self.assertEqual(m * SliceMap.lerp(1, 1), m)

        # Overlap, gap in the source space
        m1 = SliceMap([slice(0, 2), slice(0, 2), slice(0, 2)], 2)
        m2 = SliceMap([slice(0, 1), slice(2, 3)], 3)
        self.assertEqual((m1 * m2).project(list("abc"), "?"), list("ccc"))

        # Overlap, gap in the target space
        m1 = SliceMap([slice(0, 2), slice(0, 2), slice(0, 2)], 3)
        m2 = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2)], 2)
        self.assertEqual((m1 * m2).project(list("abc"), "?"), list("c?"))

        # Overlap, overlap
        m1 = SliceMap([slice(0, 2), slice(0, 2)], 2)
        m2 = SliceMap([slice(1, 3), slice(1, 3)], 3)
        self.assertEqual(m1 * m2, m2)

    def test_concat(self):
        # Empty
        self.assertEqual(SliceMap.empty() + SliceMap.empty(), SliceMap.empty())

        # Map to nothing
        m1 = SliceMap([slice(0, 1), slice(1, 2)], 2)
        m2 = SliceMap([slice(0, 0), slice(0, 0)], 0)
        self.assertEqual(m1 + m2, SliceMap([slice(0, 1), slice(1, 2), slice(2, 2), slice(2, 2)], 2))
        self.assertEqual(m2 + m1, SliceMap([slice(0, 0), slice(0, 0), slice(0, 1), slice(1, 2)], 2))

        # Map from nothing
        m1 = SliceMap([], 2)
        m2 = SliceMap([slice(0, 1), slice(1, 2)], 2)
        self.assertEqual(m1 + m2, SliceMap([slice(2, 3), slice(3, 4)], 4))
        self.assertEqual(m2 + m1, SliceMap([slice(0, 1), slice(1, 2)], 4))

        # Identity
        m1 = SliceMap([slice(0, 1), slice(1, 2)], 2)
        m2 = SliceMap([slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 4)], 4)
        self.assertEqual(m1 + m1, m2)

    def test_lerp(self):
        # Empty
        self.assertEqual(SliceMap.lerp(0, 0), SliceMap.empty())

        # Map to nothing
        self.assertEqual(SliceMap.lerp(1, 0), SliceMap([slice(0, 0)], 0))

        # Map from nothing
        self.assertEqual(SliceMap.lerp(0, 1), SliceMap([], 1))

        # Identity
        m = SliceMap([slice(0, 1), slice(1, 1), slice(1, 2), slice(3, 5), slice(4, 6)], 7)
        self.assertEqual(m * SliceMap.lerp(7, 7), m)

        # Ensure the spread is even
        for i in range(1, 20):
            for j in range(1, 20):
                m = SliceMap.lerp(i, j)
                idx = np.arange(i)
                counts = np.zeros(j, dtype=int)
                for k in range(i):
                    for l in idx[m[k]]:
                        counts[l] += 1
                self.assertGreaterEqual(counts.min() + 1, counts.max())
