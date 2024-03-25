import numpy as np

regions = ['africa', 'india', 'arabia']

for region in regions:
    for run in ['speedy', 'hybrid']:
        data = np.load(f'/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/from_dan/new_design/annual/{region}_{run.capitalize()}_lifted_index_annual.npy') #there are 28 points in the arabia region
        print(data.shape)

        months = {
            1: 31,
            2: 28,
            3: 31,
            4: 30,
            5: 31,
            6: 30,
            7: 31,
            8: 31,
            9: 30,
            10: 31,
            11: 30,
            12: 31
        }


        # Now find the rows for DJF and JJA
        DJF_cols = np.array([], dtype=int)
        JJA_cols = np.array([], dtype=int)
        counter = 0
        for year in range(1982, 1992):
            for month, days in months.items():
                if month in [1, 2, 12]:
                    # Jan, Feb, Dec
                    DJF_cols = np.concatenate((DJF_cols, np.arange(counter, counter+days*4)))
                    counter += days*4
                    if month == 2 and (year == 1984 or year == 1988):
                        # Leap year
                        counter += 4
                        DJF_cols = np.concatenate((DJF_cols, np.arange(counter, counter+4)))
                elif month in [6, 7, 8]:
                    # Jun, Jul, Aug
                    JJA_cols = np.concatenate((JJA_cols, np.arange(counter, counter+days*4)))
                    counter += days*4
                else:
                    counter += days*4

        print("DJF", DJF_cols.shape)
        print(DJF_cols[:10])
        print(DJF_cols[-10:])

        print("JJA", JJA_cols.shape)
        print(JJA_cols[:10])
        print(JJA_cols[-10:])

        DJF_data = data[:, DJF_cols]
        JJA_data = data[:, JJA_cols]

        print(DJF_data.shape)
        print(JJA_data.shape)

        np.save(f'/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/from_dan/new_design/{run}_seasonal/{region}_lifted_index_DJF.npy', DJF_data)
        np.save(f'/Users/jamesbriant/Documents/Projects/ml_climate_fusion/data/from_dan/new_design/{run}_seasonal/{region}_lifted_index_JJA.npy', JJA_data)