import numpy as np
import pkg_resources
import pandas as pd

from bb_utils.ids import BeesbookID


class BeeMetaInfo:
    def __init__(self):
        hatchdates_path = pkg_resources.resource_filename('bb_utils', 'data/hatchdates2016.csv')
        self.hatchdates = pd.read_csv(hatchdates_path)
        self.hatchdates.hatchdate = pd.to_datetime(self.hatchdates.hatchdate, format='%d.%m.%Y')

    def _check_date(self, timestamp):
        if timestamp.year != 2016:
            raise ValueError('Meta information only available for season 2016')

    def get_hatchdate(self, bee_id):
        """Get hatchdate of the bee with the given ID.

        Arguments:
            bee_id (:class:`.BeesbookID`): :class:`.BeesbookID` with ID

        Returns:
            :class:`datetime.dateime`: hatchdate of the bee
        """
        assert(type(bee_id) is BeesbookID)
        indices = np.where(self.hatchdates.dec12 == bee_id.as_dec_12())[0]
        if len(indices) == 0:
            raise ValueError('Unknown ID {}'.format(bee_id))
        return self.hatchdates.iloc[indices[0]].hatchdate

    def has_hatched(self, bee_id, timestamp):
        """Check whether a bee has already hatched given a timestamp and ID.

        Arguments:
            bee_id (:class:`.BeesbookID`): :class:`.BeesbookID` with ID
            timestamp (:class:`datetime.datetime`): :class:`datatime.datetime` with timestamp

        Returns:
            :bool: True if bee has hatched before the given timestamp
        """
        self._check_date(timestamp)

        bee_hatchdate = self.get_hatchdate(bee_id)
        return timestamp >= bee_hatchdate

    def get_age(self, bee_id, timestamp):
        """Check the age of a bee given a timestamp and ID.

        Arguments:
            bee_id (:class:`.BeesbookID`): :class:`.BeesbookID` with ID
            timestamp (:class:`datetime.datetime`): :class:`datatime.datetime` with timestamp

        Returns:
            :class:`datetime.timedelta` Age of the bee at the given timestamp
        """
        self._check_date(timestamp)

        bee_hatchdate = self.get_hatchdate(bee_id)
        return timestamp - bee_hatchdate
