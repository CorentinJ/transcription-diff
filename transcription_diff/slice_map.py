from typing import Union, List, Iterable, overload

import numpy as np


class SliceMap:
    def __init__(self, smap: Union[List[slice], np.ndarray], target_len: int):
        """
        A slice map smap is a list of slices that maps from X to Y with
            - X[i] mapping to Y[smap[i]]
            - X[i:j] mapping to Y[smap[i].start:smap[j - 1].stop]

        Informally, an item in X can correspond to 0 or more consecutive items in Y. A slice of one or more items in X
        will map to the slice spanning from the leftmost corresponding item in Y to the rightmost corresponding item.

        :param smap: the list of slices. The following must hold:
            - len(smap) must be equal to the size of the X
            - slices cannot have negative indices and cannot index beyond the size of Y
            - the slice starts cannot decrease, the same goes for the slice stops. You can however have consecutive
              overlapping slices, e.g. [slice(0, 2), slice(0, 2)].
        Note that slices can be empty (stop <= start).
        The slices can also be passed as an (X, 2) shaped integer array. The second dimension holds slice starts and
        ends, respectively.
        :param target_len: the size of Y
        """
        self.source_len = len(smap)
        self.target_len = target_len

        # Convert slices to an array
        if not isinstance(smap, np.ndarray):
            self._map = np.empty((self.source_len, 2), dtype=np.int64)
            for i, sli in enumerate(smap):
                self._map[i] = [sli.start, sli.stop]
        else:
            self._map = smap.astype(np.int64, copy=True)

        assert np.all((0 <= self._map) & (self._map <= target_len)), "Slice starts/stops out of bounds"
        assert np.all(self._map[1:] >= self._map[:-1]), "Slice starts/stops must be increasing"

    def __getitem__(self, item: Union[int, slice]) -> slice:
        """
        Indexes the position in X with either an integer or a slice (step != 1 is not supported). Returns the
        corresponding slice in Y.
        """
        if np.issubdtype(type(item), np.integer):
            item = slice(item, item + 1)
        else:
            assert item.step in [None, 1], "Only steps of 1 are supported"

        view = self._map[item]
        if len(view):
            # We return a slice that spans from the lowest to the highest target indices
            return slice(view[0][0], view[-1][1])
        else:
            # We return an empty slice, it is computed so as to stay consistent with our axioms.
            pos = np.clip(0, item.start, self.source_len)
            start = self._map[pos][0] if pos < self.source_len else self.target_len
            stop = self._map[pos - 1][1] if pos > 0 else 0
            stop = max(start, stop)
            return slice(start, stop)

    def __len__(self):
        """
        Returns the size of X
        """
        return self.source_len

    def __bool__(self):
        """
        To ensure we still get a True value when the mapping is empty
        """
        return True

    def __iter__(self):
        """
        Iterates over slices, returning pairs (start, stop)
        """
        yield from ((int(start), int(end)) for start, end in self._map)

    def project(self, data: Union[np.ndarray, List], default=None) -> Union[np.ndarray, List]:
        """
        Projects data in the source space to the target space.
        A default value will be returned in place of gaps in the target space.
        In case of overlaps, the rightmost item will take priority.

        :param data: a list of arbitrary objects or a numpy array. It must be that len(data) == source_len
        :param default: the value to give to entries that nothing maps to. This value must be specified in the case
        of numpy arrays
        :return: the projected data in Y as a list or numpy array
        """
        assert len(data) == self.source_len, "The data to project must have the same length as the mapping."
        is_numpy = isinstance(data, np.ndarray)
        assert not (is_numpy and default is None), "The default value must be specified for numpy arrays."

        if is_numpy:
            projected = np.full_like(data, default, shape=self.target_len)
        else:
            projected = [default] * self.target_len

        for source_idx, (target_start, target_end) in enumerate(self._map):
            if is_numpy:
                projected[target_start:target_end] = data[source_idx]
            else:
                projected[target_start:target_end] = [data[source_idx]] * (target_end - target_start)

        return projected

    def inverse(self) -> 'SliceMap':
        """
        With self mapping from X to Y, returns the inverse Y to X mapping.
        This operation is bijective, including in the presence of gaps or overlaps.
        """
        # Find the Points Of Interest: the indices where the mapping's starts or stops increase
        bounded_map = np.concatenate((self._map, [[self.target_len, self.target_len]]))
        changes = np.diff(bounded_map, axis=0, prepend=0)
        (start_pois,), (stop_pois,) = changes[:, 1].nonzero(), changes[:, 0].nonzero()

        n_repeats = np.diff(bounded_map[start_pois, 1], prepend=0)
        inv_map_starts = np.repeat(start_pois, n_repeats)

        n_repeats = np.diff(bounded_map[stop_pois, 0], prepend=0)
        inv_map_stops = np.repeat(stop_pois, n_repeats)

        inv_map = np.stack([inv_map_starts, inv_map_stops], axis=1)

        return SliceMap(inv_map, self.source_len)

    def compose(self, other: 'SliceMap') -> 'SliceMap':
        """
        With self mapping from X to Y and other mapping from Y to Z, returns the composed X to Z mapping.
        """
        assert self.target_len == other.source_len, \
            f"Cannot compose {self.source_len}x{self.target_len} map with {other.source_len}x{other.target_len} map."

        smap = np.empty((self.source_len, 2), dtype=np.int64)
        for i in range(len(self)):
            sli = other[self[i]]
            smap[i] = [sli.start, sli.stop]

        return SliceMap(smap, other.target_len)

    def __mul__(self, other):
        """
        Multiplication is shorthand for compose
        """
        return self.compose(other)

    def concat(self, other: 'SliceMap') -> 'SliceMap':
        """
        With self mapping from Xi to Yi and other mapping from Xj to Yj, returns the concatenated mapping
        from cat(Xi, Xj) to cat(Yi, Tj).
        """
        new_map = np.concatenate((self._map, other._map + self.target_len))
        return SliceMap(new_map, self.target_len + other.target_len)

    def __add__(self, other):
        """
        Addition is shorthand for concatenation
        """
        return self.concat(other)

    def __eq__(self, other: 'SliceMap'):
        if other is None:
            return False
        return \
            self.source_len == other.source_len and \
            self.target_len == other.target_len and \
            np.array_equal(self._map, other._map)

    @staticmethod
    def from_1to1_map(oto_map: Iterable[int], target_len: int):
        """
        Creates a slicemap where each index i corresponds to the slice oto_map[i]:oto_map[i] + 1
        """
        return SliceMap([slice(p, p + 1) for p in oto_map], target_len)

    @staticmethod
    def from_ranges(ranges: Iterable[int]):
        """
        This is the non-cumulative version of a monotonic mapping:
            - SliceMap.from_ranges(r) is equivalent to SliceMap.from_monotonic_map(np.cumsum(r))
        """
        smap = []
        target_pos = 0
        for r in ranges:
            smap.append(slice(target_pos, target_pos + r))
            target_pos += r
        return SliceMap(smap, target_pos)

    @staticmethod
    def lerp(source_len: int, target_len: int):
        """
        Creates a map that linearly interpolates from X to Y, e.g. for source_len=6 and target_len=12, the slice
        2:3 in X maps to 4:6 in Y.
        """
        low = min(source_len, target_len)
        high = max(source_len, target_len)
        idx = np.linspace(0, low, high, endpoint=False, dtype=np.int64)
        smap = np.stack([idx, np.minimum(idx + 1, low)], axis=1)
        smap = SliceMap(smap, low)

        return smap if target_len == low else smap.inverse()

    @staticmethod
    def full(source_len: int, target_len: int):
        """
        Creates a map where each element in the source space maps to the entirety of the target space.
        """
        smap = np.zeros((source_len, 2), dtype=np.int64)
        smap[:, 1] = target_len
        return SliceMap(smap, target_len)

    @staticmethod
    def empty() -> 'SliceMap':
        return SliceMap([], 0)

    @staticmethod
    def identity(length: int) -> 'SliceMap':
        return SliceMap.slice(0, length, length)

    @overload
    def slice(start: int, end: int, target_len: int) -> 'SliceMap': ...
    @overload
    def slice(sli: slice, target_len: int) -> 'SliceMap': ...
    @staticmethod
    def slice(*args) -> 'SliceMap':
        """
        Convenience method. Creates a map where all elements map to a slice of a target space.
        - <start> is where the slice begins in the target space
        - <end> is where the slice ends in the target space
        - <target_len> is the size of the target space
        This method is the inverse of eye()
        """
        if len(args) == 2:
            start, end, target_len = args[0].start, args[0].stop, args[1]
        else:
            start, end, target_len = args
        assert 0 <= start <= end <= target_len, f"Invalid slice: {start}:{end} in {target_len}"
        return SliceMap(
            np.stack([np.arange(start, end), np.arange(start, end) + 1], axis=1),
            target_len
        )

    @overload
    def eye(start: int, end: int, length: int) -> 'SliceMap': ...
    @overload
    def eye(sli: slice, length: int) -> 'SliceMap': ...
    @staticmethod
    def eye(*args) -> 'SliceMap':
        """
        Convenience method. Creates a map where
        - the <start> first element map to nothing
        - the elements between <start> and <end> map to the identity
        - the elements after <end> (up to <length>) map to nothing
        This method is the inverse of slice()
        """
        if len(args) == 2:
            start, end, length = args[0].start, args[0].stop, args[1]
        else:
            start, end, length = args
        return SliceMap.full(start, 0) + SliceMap.identity(end - start) + SliceMap.full(length - end, 0)

    @staticmethod
    def compose_by_name(mapping_name: str, **mappings: 'SliceMap'):
        """
        Composes mappings together based on their names. Each SliceMap passed as <mappings> argument must have
        the name structure <source2target>.

        For example, calling the function as:
            SliceMap.compose_by_name('a2c', a2b=a2b, b2c=b2c)
        will return the composition a2c = a2b * b2c.

        An AssertionError will be raised if <source_name> or <target_name> are not found in the names of the
        passed mappings.

        Mappings that are not used in the composition may be passed. They will be ignored.
        """
        assert all(k.count("2") == 1 for k in list(mappings) + [mapping_name]), \
            f"All mappings must conform to the name convention <source2target>, got {list(mappings)}"
        source_name, target_name = mapping_name.split("2")
        source_names, target_names = zip(*[map_name.split("2") for map_name in mappings])
        assert source_name in source_names, f"Source name {source_name} not found in {source_names}"
        assert target_name in target_names, f"Target name {target_name} not found in {target_names}"

        dim_name = source_name
        composed_map = None
        seen_idx = set()
        while dim_name != target_name:
            map_idx = source_names.index(dim_name)
            assert map_idx not in seen_idx, f"Cycle detected: {list(mappings)}"
            seen_idx.add(map_idx)
            map_to_compose = mappings[f"{dim_name}2{target_names[map_idx]}"]
            composed_map = composed_map * map_to_compose if composed_map else map_to_compose
            dim_name = target_names[map_idx]

        return composed_map

    def __repr__(self):
        return f"<{self.source_len}x{self.target_len} map: {[tuple(sli) for sli in self._map]}>"
