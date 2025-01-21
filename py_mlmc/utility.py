from typing import Self
from time import perf_counter

import numpy as np


class NormalGenerator:
    """
    A memory-mapped buffer of standard normal randoms.
    Generates randoms once (or in chunks), storing in a memmap file.
    Each call slices out the needed portion. If we run out, we generate more
    at the end of the file or overwrite from the beginning (depending on your use-case).

    Because this is disk-backed, large arrays can exceed RAM if needed.
    Also, if you want to preserve random draws across runs, you can keep the memmap file around.
    """

    def __init__(
        self,
        buffer_size: int = int(1e6),
        rng: np.random.Generator | None = None,
        dtype: np.dtype = np.float64,
        filename: str = "gaussian_buffer.dat",
        create_file: bool = True,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        buffer_size : int
            Number of normal randoms in one chunk / block.
        rng : np.random.Generator
            Optional random generator. If None, a default is created.
        dtype : np.dtype
            Optional data type, default to `np.float64`.
        filename : str
            Optional file name for the memory map, default to "gaussian_buffer.dat".
        create_file : bool
            If True create a new file and generate a new chunk.
        verbose : bool
            If True, prints a message on generating a new chunk of randoms.
        """
        self.buffer_size = buffer_size
        self.rng = rng or np.random.default_rng()
        self.dtype = dtype
        self.filename = filename
        self.verbose = verbose

        self._index = 0
        if create_file:
            self._mm = np.memmap(filename=self.filename, dtype=self.dtype, mode="w+", shape=(self.buffer_size,))
            self._generate_new_chunk()
        else:
            self._mm = np.memmap(filename=self.filename, dtype=self.dtype, mode="r+", shape=(self.buffer_size,))

    def get(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: int | tuple[int, ...] = 1,
    ) -> np.ndarray:
        """
        Draws `size` samples from the buffer, with optional loc/scale shift.
        If `size` is larger than what's left in the buffer, a new buffer is created.

        Parameters
        ----------
        loc : float
            Mean of the normal distribution to apply (shift).
        scale : float
            Std dev of the normal distribution to apply (scale).
        size : int | tuple[int, ...]
            Shape of the output array.  E.g. `size=100` or `size=(100, 3)`.

        Returns
        -------
        np.ndarray
            Array of shape `size` with draws from Normal(loc, scale).
        """
        # Total number of samples needed
        count = np.prod(size)

        # Check if requested size is within buffer size
        if count > self.buffer_size:
            raise ValueError(f"Requested size {count:,} exceeds buffer size {self.buffer_size:,}.")

        # If requested more than leftover, generate a fresh buffer
        if self._index + count > self.buffer_size:
            self._generate_new_chunk()

        out = self._mm[self._index : self._index + count]
        self._index += count

        out = loc + scale * out
        return out.reshape(size)

    def reset(self) -> Self:
        """
        Reset the index so that the next call replays from the
        beginning of the current buffer. This is handy if you
        want identical draws in a subsequent simulation run.
        """
        self._index = 0
        return self

    def _generate_new_chunk(self) -> None:
        """
        Overwrites the entire memmap array with new randoms from `self.rng`.
        If your use-case needs partial or incremental expansions,
        you can adapt this method to append or to a separate offset, etc.
        """
        if self.verbose:
            print(f"[NormalGenerator] Generating {self.buffer_size:,} normals...")
            start = perf_counter()

        tmp = self.rng.normal(size=self.buffer_size)
        self._mm[:] = tmp[:]
        self._mm.flush()  # ensure changes are written to disk
        self._index = 0

        if self.verbose:
            end = perf_counter()
            print(f"[NormalGenerator] Took {end - start}s.")
