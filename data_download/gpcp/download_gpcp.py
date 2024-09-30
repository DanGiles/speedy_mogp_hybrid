import cdsapi

c = cdsapi.Client()

c.retrieve(
    'satellite-precipitation',
    {
        'variable': 'all',
        'format': 'zip',
        'time_aggregation': 'monthly_mean',
        'year': [
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
    },
    'download.zip')