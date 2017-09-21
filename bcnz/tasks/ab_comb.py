
import ipdb
import numpy as np
import pandas as pd

class ab_comb:
    """Combine the input for different extinction laws."""

    version = 1.05
    config = {}

    def combine_input(self):
        """Megre dataframes and change their format."""

        df = pd.DataFrame()

        # This task used to have more logic when the xab had a weird
        # format...
        for key, job in self.job.depend.items():
            cat = job.result.unstack()
            df = df.append(cat, ignore_index=True)


        # These are already available from EBV = 0.
        df = df[df.ext != 'none']

        df['z'] = df.z.astype(np.float)
        df['flux'] = df.flux.astype(np.float)
        df['EBV'] = df.EBV.astype(np.float)

        return df

    def run(self):
        print('starting combine_input')

        self.job.result = self.combine_input()
