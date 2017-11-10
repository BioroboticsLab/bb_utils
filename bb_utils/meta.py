import numpy as np
import pkg_resources
import pandas as pd

from bb_utils.ids import BeesbookID


class BeeMetaInfo:
    def __init__(self):
        hatchdates_path = pkg_resources.resource_filename('bb_utils', 'data/hatchdates2016.csv')
        self.hatchdates = pd.read_csv(hatchdates_path)
        self.hatchdates.hatchdate = pd.to_datetime(self.hatchdates.hatchdate, format='%d.%m.%Y')

        foragers_path = pkg_resources.resource_filename('bb_utils', 'data/foragergroups2016.csv')
        self.foragers = pd.read_csv(foragers_path)
        self.foragers.date = pd.to_datetime(self.foragers.date, format='%d.%m.%Y')
        self.foragers.dec12 = self.foragers.dec12.apply(lambda ids: list(map(int, ids.split(' '))))

        beenames_path = pkg_resources.resource_filename('bb_utils', 'data/beenames.csv')
        self.beenames = pd.read_csv(beenames_path, sep=' ')

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

    def get_group_memberships(self, bee_id):
        """Get forager groups of the bee with the given ID.

        Arguments:
            bee_id (:class:`.BeesbookID`): :class:`.BeesbookID` with ID

        Returns:
            :[int]: list of forager group ids
        """
        assert(type(bee_id) is BeesbookID)
        groups = []
        for row in self.foragers.iterrows():
            if bee_id.as_dec_12() in row[1].dec12:
                groups.append(row[1].group_id)
        return groups

    def get_foragergroup(self, group_id):
        """Get metainformation for a specific forager group.

        Arguments:
            group_id (:int:): Group ID

        Returns:
            :class:`pd.Series`: forager group metainformation
        """
        indices = np.where(self.foragers.group_id == group_id)[0]
        if len(indices) == 0:
            raise ValueError('Unknown ID {}'.format(group_id))
        return self.foragers.iloc[indices[0]]

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

    def get_beename(self, bee_id):
        """Return the Beename-Char-RNN generated name for the given ID.

        Arguments:
            bee_id (:class:`.BeesbookID`): :class:`.BeesbookID` with ID
        Returns:
            :class:`str` Name of the bee with the given ID
        """
        return self.beenames[self.beenames.bee_id == bee_id.as_ferwar()].name.values[0]
