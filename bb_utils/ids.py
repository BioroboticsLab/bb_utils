import numpy as np

from bb_utils.visualization import TagArtist


class BeesbookID:
    def __init__(self, bee_id):
        if type(bee_id) is not np.array:
            bee_id = np.array(bee_id)
        bee_id = np.round(bee_id).astype(np.int)

        self.binary_12 = bee_id

    @staticmethod
    def _dec_to_bin(bee_id):
        assert(bee_id < 4096)
        bee_id = (np.array([2 ** i for i in range(11, -1, -1)]) & bee_id)
        return bee_id.astype(np.bool).astype(np.int)

    @classmethod
    def from_bin_9(cls, bee_id):
        """Initialize ID using bin_9 representation.

        Note:
            Bits are in most significant bit first order starting at the 9 o'clock position on the
            tag in clockwise orientation.

        Arguments:
            bee_id: ID in bin_9 representation
        """

        return cls(np.roll(bee_id, -3, axis=0))

    @classmethod
    def from_bin_12(cls, bee_id):
        """Initialize ID using bin_12 representation.

        Note:
            Bits are in most significant bit first order starting at the 12 o'clock position on the
            tag in clockwise orientation.

            This is the same as the bb_binary representation.

            See https://arxiv.org/pdf/1611.01331.pdf Figure 1(a) for a graphical representation.

        Arguments:
            bee_id: ID in bin_12 representation
        """

        return cls(bee_id)

    @classmethod
    def from_dec_9(cls, bee_id):
        """Initialize ID using decimal representation called 'dec_9' by Maria Sparenberg.

        Note:
            In this representation, lower decimal IDs roughly correspond to earlier hatch dates.

            This is the same as the 'ferwar' representation.

        Arguments:
            bee_id: ID in dec_9 representation
        """

        return cls.from_ferwar(bee_id)

    @classmethod
    def from_dec_12(cls, bee_id):
        """Initialize ID using decimal representation called 'dec_12' by Maria Sparenberg.

        Arguments:
            bee_id: ID in dec_12 representation
        """

        return cls(cls._dec_to_bin(bee_id))

    @classmethod
    def from_ferwar(cls, bee_id):
        """Initialize ID using decimal representation originally used by Fernando Wario.

        Note:
            In this representation, lower decimal IDs roughly correspond to earlier hatch dates.

            This is the same as the 'dec_9' representation.

        Arguments:
            bee_id: ID in ferwar representation
        """

        needs_inverse_parity = bee_id >= 2048
        bee_id = bee_id - 2048 if needs_inverse_parity else bee_id
        bin_9 = cls._dec_to_bin(bee_id)
        bin_9 = np.roll(bin_9, -1, axis=0)
        parity = np.sum(bin_9[:-1]) % 2
        if needs_inverse_parity:
            bin_9[-1] = 1 - parity
        else:
            bin_9[-1] = parity

        return cls.from_bin_9(bin_9)

    @classmethod
    def from_bb_binary(cls, bee_id):
        """Initialize ID using bb_binary representation.

        Note:
            Bits are in most significant bit first order starting at the 12 o'clock position on the
            tag in clockwise orientation.

            See https://arxiv.org/pdf/1611.01331.pdf Figure 1(a) for a graphical representation.

        Arguments:
            bee_id: ID in bb_binary representation
        """

        return cls(bee_id)

    def as_dec_9(self):
        """Return ID decimal representation called 'dec_9' by Maria Sparenberg.

        Note:
            In this representation, lower decimal IDs roughly correspond to earlier hatch dates.

            This is the same as the ferwar representation.

        Returns:
            :obj:`int`: ID in 'dec_9' decimal representation
        """
        return self.as_ferwar()

    def as_dec_12(self):
        """Return ID decimal representation called 'dec_12' by Maria Sparenberg.

        Returns:
            :obj:`int`: ID in 'dec_12' decimal representation
        """
        return np.sum(2 ** np.array(range(12))[::-1] * self.binary_12)

    def as_bin_9(self):
        """Return ID in 'bin_9' binary representation.

        Note:
            Bits are in most significant bit first order starting at the 9 o'clock position on the
            tag in clockwise orientation.

        Returns:
            :obj:`np.array`: ID in bb_binary representation
        """
        return np.roll(self.binary_12, 3, axis=0)

    def as_ferwar(self):
        """Return ID decimal representation originally used by Fernando Wario.

        Note:
            In this representation, lower decimal IDs roughly correspond to earlier hatch dates.

            This is the same as the 'dec_9' representation.

        Returns:
            :obj:`int`: ID in ferwar decimal representation
        """
        # convert to decimal id using 11 least significant bits
        bin_9 = self.as_bin_9()
        decimal_id = int(''.join([str(c) for c in bin_9[:11]]), 2)
        parity_bit = bin_9[-1]

        # determine what kind of parity bit was used and add 2^11 to decimal id
        # if uneven parity bit was used
        if (np.sum(bin_9[:11]) % 2) != parity_bit:
            decimal_id += 2048

        return decimal_id

    def as_bb_binary(self):
        """Return ID in decoder binary representation.

        Note:
            Bits are in most significant bit first order starting at the 12 o'clock position on the
            tag in clockwise orientation.

            See https://arxiv.org/pdf/1611.01331.pdf Figure 1(a) for a graphical representation.

        Returns:
            :obj:`np.array`: ID in bb_binary representation
        """

        return self.binary_12

    @staticmethod
    def batch_bb_binary_to_ferwar(ids):
        """Vectorized conversion of bb_binary IDs to ferwar decimal IDs.

        Note:
            This function can be used when a large batch of IDs has to be converted.
            It should be significantly faster than constructing a BeesbookID object
            and calling as_ferwar() for each ID.

        Arguments:
            :obj:`np.array`: array of bb_binary IDs with shape [batch_size, 12]

        Returns:
            :obj:`np.array`: array of decimal IDs in ferwar representation
        """
        binary_ids = np.round(ids).astype(np.int)

        adjusted_ids = np.roll(binary_ids, 3, axis=1)

        # convert to decimal id using 11 least significant bits
        decimal_ids = [int(''.join([str(c) for c in id[:11]]), 2) for id in adjusted_ids]

        # determine what kind of parity bit was used and add 2^11 to decimal id
        # uneven parity bit was used
        decimal_ids = np.array(decimal_ids)
        decimal_ids[(np.sum(adjusted_ids, axis=1) % 2) == 1] += 2048

        return decimal_ids

    def __repr__(self):
        return 'BeesbookID(bb_binary|bin_12: {}, ferwar|dec_9 decimal: {})'.format(
            ''.join(self.binary_12.astype(str)), self.as_ferwar())

    def _repr_png_(self):
        print(self.__repr__())

        return TagArtist().draw(self.binary_12)
