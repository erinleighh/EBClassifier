import glob
import os
import re
import pandas as pd
import numpy as np
from scipy import stats
import exoplanet as xo
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import BoxLeastSquares
import time as timer

start = timer.time()

def classification(blsMP, z, factor):
    resClassification = 'und'

    # Line from z 3, BLS 1000 to z 15, BLS 100
    minblsMP = startMP - dropMP * factor

    # Identify significant eclipses
    if blsMP >= minblsMP or z >= 10 - 2 * factor:
        resClassification = 'EB'
        print(objName + ' IS CLASSIFIED AS AN ECLIPSING BINARY****')

    return resClassification


def findZ(relFlux, time):
    # Calculate z-score of all points, find outliers below the flux midpoint.
    z = stats.zscore(relFlux)
    potentialEclipses = z[np.where(relFlux < 1)[0]]
    peTimes = time.iloc[np.where(relFlux < 1)[0]]

    zScoreRangeIndex = range(max(np.argmin(potentialEclipses) - 2, 0),
                            min(np.argmin(potentialEclipses) + 3, potentialEclipses.size))
    zRange = np.ceil(potentialEclipses[zScoreRangeIndex]).astype(int)

    avgMaximumZ = np.average(zRange).round(0).astype(int) * -1
    # To ensure that data points near the minimum Z score data point are indeed significant

    return avgMaximumZ, potentialEclipses[zScoreRangeIndex] * -1, peTimes.iloc[zScoreRangeIndex], \
           relFlux.iloc[zScoreRangeIndex]


def autocorrelationfn(time, relFlux, relFluxErr):
    acf = xo.autocorr_estimator(time.values, relFlux.values, yerr=relFluxErr.values,
                                min_period=0.05, max_period=27, max_peaks=10)

    period = acf['autocorr'][0]
    power = acf['autocorr'][1]

    acfPowerPd = pd.DataFrame(power)
    acfLocalMaxima = acfPowerPd[(acfPowerPd.shift(1) < acfPowerPd) & (acfPowerPd.shift(-1) < acfPowerPd)]
    maxPower = np.max(acfLocalMaxima).values

    bestPeriod = period[np.where(power == maxPower)[0]][0]
    peaks = acf['peaks'][0]['period']

    if len(acf['peaks']) > 0:
        window = int(peaks / np.abs(np.nanmedian(np.diff(time))) / k)
    else:
        window = 128

    return period, power, bestPeriod, maxPower, window


def boxleastsquares(time, relFlux, relFluxErr, acfBP):
    model = BoxLeastSquares(time.values, relFlux.values, dy=relFluxErr.values)
    duration = [20 / 1440, 40 / 1440, 80 / 1440, .1]
    periodogram = model.power(period=[.5 * acfBP, acfBP, 2 * acfBP], duration=duration,
                              objective='snr')
    period = periodogram.period
    power = periodogram.power
    maxPower = np.max(periodogram.power)
    bestPeriod = periodogram.period[np.argmax(periodogram.power)]

    return period, power, bestPeriod, maxPower


def addtotable(table, oID, sec, smooth, blsMP, blsP, acfMP, acfP, amZ, rZ, flg, c, fn):
    table = table.append(
        {'Obj ID': oID, 'Sector': sec, 'Times Smoothed': smooth, 'BLS Max Pow': blsMP, 'BLS Per': blsP,
         'ACF Max Pow': acfMP, 'ACF Per': acfP, 'AvgMax Z_5': amZ, 'AvgMax Z Range': rZ, 'Flag': flg,
         'Classification': c, 'Filename': fn}, ignore_index=True)
    return table


def makegraph(xaxis, yaxis, xlabels, ylabels, lbl, color, marker=None, size=None, style=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if style is None:
        ax.scatter(xaxis, yaxis, color=color, marker=marker, s=size)
    else:
        ax.plot(xaxis, yaxis, color=color)

    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.title(lbl)
    return ax

k = 5
startMP = 1500
dropMP = 650

# Classification by Sector - First Round

lightCurves = []  # Initialize the array holding light curves
EBs = []  # Store the objects classified as eclipsing binaries

data = pd.read_csv('/data/epyc/users/jrad/TESS_CVZ/001_026_1S.csv')
files = data['file']

lcTable = pd.DataFrame(
    columns=['Obj ID', 'Sector','Times Smoothed', 'BLS Max Pow', 'BLS Per', 'ACF Max Pow', 'ACF Per',
             'AvgMax Z_5', 'AvgMax Z Range',  'Flag', 'Classification', 'Filename'])

for file in files:
    fitsTable = fits.open(file, memmap=True)
    objName = fitsTable[0].header['OBJECT']
    sector = re.search(r"sector\d+", file).group().replace('sector', '')
    print("\nReading in " + objName + " Filename: " + file)
    try:
        curveTable = Table(fitsTable[1].data).to_pandas()
    except:
        print('*************** ERROR ***************')
        f = open('errors.txt', 'a')
        f.write(file + '\n')
        f.close()
    else:
        curveData = curveTable.loc[curveTable['QUALITY'] == 0].dropna(subset=['TIME']).dropna(
            subset=['PDCSAP_FLUX']).copy()
        curveData = curveData.filter(['TIME', 'PDCSAP_FLUX', 'PDCSAP_FLUX_ERR'])

        # Find time gaps greater than 1 day
        idx = np.where((curveData['TIME'][1:]-curveData['TIME'][:-1]).isnull())[0]
        idxL = idx[np.where(idx[1:]-idx[:-1] > 1)]
        idxR = idx[np.where(idx[1:]-idx[:-1] > 1)[0]+1]

        for badDataPoint in idxL:
            # Set data points to the right to null
            r = range(badDataPoint + 1, badDataPoint + 1001)

            try:
                curveData.loc[r, 'PDCSAP_FLUX'] = np.nan
                curveData.loc[r, 'TIME'] = np.nan
            except:
                pass

        for badDataPoint in idxR:
            # Set data points to the left to null
            l = range(badDataPoint - 1000, badDataPoint)

            try:
                curveData.loc[l, 'PDCSAP_FLUX'] = np.nan
                curveData.loc[l, 'TIME'] = np.nan
            except:
                pass

        curveData = curveData.dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
        fluxMed = np.nanmedian(curveData['PDCSAP_FLUX'])
        curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)
        curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)

        classif = 'und'
        flag = ''

        # Classify based on outliers
        i = 0
        while classif == 'und' and i < 3:  # Potential to be an EB.

            zScore5, rangeZ, fluxRange, timeRange = findZ(curveData['REL_FLUX'], curveData['TIME'])

            if zScore5 < 3 or min(rangeZ) < 1:
                # Pre-classification
                # Avg Max Z Score of 0 through 2 highly unlikely to be eclipse.
                # If one of the data points is less than one, likely to be error.
                if i == 0 and zScore5 < 3:
                    classif = 'nonEB'
                i += 1
                break
            else:
                # Run ACF and BLS functions for classification

                try:
                    # Autocorrelation Function
                    print("Generating ACF periodogram.")
                    acfPeriod, acfPower, acfBestPeriod, acfMaxPower, s_window = autocorrelationfn(curveData['TIME'], curveData['REL_FLUX'], curveData['REL_FLUX_ERR'])

                    # Box Least Squares
                    print("Generating BLS periodogram.")
                    BLSperiod, BLSpower, BLSbestPeriod, BLSmaxPower = boxleastsquares(curveData['TIME'], curveData['REL_FLUX'], curveData['REL_FLUX_ERR'], acfBestPeriod)
                except:
                    classif = 'nonEB'
                    i += 1
                    break

                # Additional pre-classification
                if (i == 0 and BLSmaxPower < 100 and zScore5 < 7) or \
                        (acfMaxPower < 0.05 and zScore5 < 4) or BLSmaxPower < 60:
                    # No need to smooth attempt further, very unlikely to be obvious EBs.
                    classif = 'nonEB'
                    i += 1
                    break

                # Run classification
                classif = classification(BLSmaxPower, zScore5, i)

                if classif == 'und':
                    # Perform Smoothing
                    print("Performing smoothing on " + objName)
                    smoothedFlux = curveData['REL_FLUX'].rolling(s_window, center=True).median()

                    SOK = np.isfinite(smoothedFlux)

                    newFlux = curveData['REL_FLUX'][SOK] - smoothedFlux[SOK]

                    curveData['REL_FLUX'] = newFlux.copy()

                    curveData = curveData.dropna(subset=['TIME']).dropna(subset=['REL_FLUX']).dropna(
                        subset=['REL_FLUX_ERR']).copy()

                    fluxMed = np.nanmedian(curveData['REL_FLUX'])
                else:
                    EBs.append(objName)  # Add to printout of EBs
                i += 1

        # Add to table
        print("Adding to table.")
        
        try:
            lcTable = addtotable(lcTable, objName, sector, i, BLSmaxPower, BLSbestPeriod,
                                 acfMaxPower[0], acfBestPeriod, zScore5, rangeZ, flag, classif, file)
        except:
            lcTable = addtotable(lcTable, objName, sector, i, 'N/A', 'N/A', 'N/A', 'N/A',
                                 zScore5, rangeZ, flag, classif, file)

    print(objName + " complete.")

print('\nClassification complete.\n')

# Print table to file
print("Print curve table to file.\n")
lcTable.to_csv('curvesTable.csv', index=False)

# Print results table to file
print("Print results table to file.\n")
grouped = lcTable.groupby('Filename', as_index=False).tail(1).sort_values(['Obj ID', 'Filename'])
grouped.to_csv('resultsTable.csv', index=False)

# Print objs with EBs to file
print("Print objs w/EBs to file.\n")
EBtable = grouped[np.isin(grouped['Classification'], ['EB'])]
EBtable = EBtable['Obj ID'].drop_duplicates().values.tolist()
EBresultsTable = grouped[grouped['Obj ID'].isin(EBtable)]
EBresultsTable.to_csv('EBresults.csv', index=False)

# Print num of EBs found
EBs = list(dict.fromkeys(EBs))
print('EBs found: ' + str(len(EBs)))

print("\nProcess complete.\n")

end = timer.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
