import glob
import os
import re
import math
import pandas as pd
import numpy as np
from scipy import stats
import exoplanet as xo
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import BoxLeastSquares
import time as timer

start = timer.time()

k = 5

### Filepaths for plots ###
ebPath = 'plots/EB'
nonEBpath = 'plots/nonEB'
undPath = 'plots/und'
chartsPath = 'plots/charts'

filepaths = [ebPath, nonEBpath, undPath, chartsPath]

for filepath in filepaths:
    # Check whether the specified path exists or not
    isExist = os.path.exists(filepath)

    if not isExist:
      os.makedirs(filepath)

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

# data = pd.read_csv('EBresults.csv')
data = pd.read_csv('justInverted.csv')
objects = data['Obj ID'].drop_duplicates()
periods = pd.DataFrame(columns=['TIC', 'RA', 'DEC', 'BLS Max Power', 'BLS Best Period', 
                                'ACF Max Power', 'ACF Best Period'])

for objName in objects:
    print("\n########## Reading in " + objName + " ##########")
    observations = data['Classification'].loc[data['Obj ID'] == objName]
    objManualFlag = data['Obj  Manual Flag'].loc[data['Obj ID'] == objName].iloc[0]
    objTable = data.loc[data['Obj ID'] == objName]
    files = objTable['Filename'].copy()
    classif = objTable['Classification'].copy()
    secManualFlag = objTable['Sector Manual Flag'].copy()
    fullCurveData = pd.DataFrame(columns =['TIME', 'REL_FLUX', 'REL_FLUX_ERR'])
    i = 0

    for file in files:
        fitsTable = fits.open(file, memmap=True)
        curveTable = Table(fitsTable[1].data).to_pandas()
        curveData = curveTable.loc[curveTable['QUALITY'] == 0].dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
        curveData = curveData.filter(['TIME', 'PDCSAP_FLUX', 'PDCSAP_FLUX_ERR'])
        fitsTable.close()

        sector = re.search(r"sector\d+", file).group().replace('sector', '')
        title = 'Light Curve for ' + objName + '\n' 

        figName = objName + "_" + sector + ".png"

        print("\n##### Beginning Sector " + str(sector) + " #####")

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
        print(fluxMed)
        curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)
        curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)

        if classif.iloc[i] == 'EB':
            makegraph(curveData['TIME'], curveData['REL_FLUX'], 'BJD - 2457000 (days)', 'Relative Flux',
                      title, 'tab:purple', '.', .2)
            plt.savefig(os.path.join(ebPath, secManualFlag.iloc[i] + "_" + figName), orientation='landscape')
        elif classif.iloc[i] == 'nonEB':
            makegraph(curveData['TIME'], curveData['REL_FLUX'], 'BJD - 2457000 (days)', 'Relative Flux',
                      title, 'tab:gray', '.', .2)
            plt.savefig(os.path.join(nonEBpath, secManualFlag.iloc[i] + "_" + figName), orientation='landscape')
        else:
            makegraph(curveData['TIME'], curveData['REL_FLUX'], 'BJD - 2457000 (days)', 'Relative Flux',
                      title, 'tab:pink', '.', .2)
            plt.savefig(os.path.join(undPath, secManualFlag.iloc[i] + "_" + figName), orientation='landscape')

        plt.close()

        # Stitch sectors together for all obs LC
        if i==0: 
            fullCurveData['TIME'] = curveData['TIME'].copy()
            fullCurveData['REL_FLUX'] = curveData['REL_FLUX'].copy()
            fullCurveData['REL_FLUX_ERR'] = curveData['REL_FLUX_ERR'].copy()
        else:
            fullCurveData = pd.concat([fullCurveData, curveData[['TIME', 'REL_FLUX', 'REL_FLUX_ERR']].copy()], ignore_index=True)

        i += 1

    ### Generating LC for all observations + phase folding ###
    time = fullCurveData['TIME']
    relFlux = fullCurveData['REL_FLUX']
    relFluxErr = fullCurveData['REL_FLUX_ERR']

    if objManualFlag == "EB":
        color = "tab:purple"
    elif objManualFlag == "nonEB":
        color = "tab:gray"
    else:
        color = "tab:pink"
        
    # Graphing
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(3,1,1) # Light Curve
    ax2 = plt.subplot(3,2,3) # ACF fold (phase space)
    ax3 = plt.subplot(3,2,4) # ACF fold (time space)
    ax4 = plt.subplot(3,2,5) # BLS fold (phase space)
    ax5 = plt.subplot(3,2,6) # BLS fold (time space)

    ## Light Curve
    ax1.set_title('Light Curve for ' + objName)
    ax1.set_xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
    ax1.set_ylabel('Relative Flux')
    ax1.xaxis.set_major_locator(MaxNLocator(12))
    ax1.scatter(fullCurveData['TIME'], fullCurveData['REL_FLUX'], s=.2, c=color)
        
    # Period Finding & Phase Folding
    try:
        _, _, ACFbestPeriod, ACFmaxPow, _ = autocorrelationfn(time, relFlux, relFluxErr)
        _, _, BLSbestPeriod, BLSmaxPow = boxleastsquares(time, relFlux, relFluxErr, ACFbestPeriod)

        periods = periods.append({'TIC': objName, 'RA': fitsTable[0].header['RA_OBJ'], 
                                  'DEC': fitsTable[0].header['DEC_OBJ'], 
                                  'BLS Max Power': BLSmaxPow, 'BLS Best Period': BLSbestPeriod, 
                                  'ACF Max Power': ACFbestPeriod, 'ACF Best Period': ACFbestPeriod}, ignore_index=True)

        ## ACF fold in phase and time space
        ax2.set_title('ACF - Fold (Phase Space)')
        ax2.set_xlabel('Period (Phase)')  
        ax2.set_ylabel('Relative Flux')
        ax2.plot((time % ACFbestPeriod)/ACFbestPeriod, relFlux, 'b,')

        ax3.set_title('ACF - Fold (Time Space)')
        ax3.set_xlabel('Period (days)') 
        ax3.set_ylabel('Relative Flux')
        ax3.plot((time % ACFbestPeriod), relFlux, 'g,')

        ## BLS fold in phase and time space
        ax4.set_title('BLS - Fold (Phase Space)')
        ax4.set_xlabel('Period (Phase)')  
        ax4.set_ylabel('Relative Flux')
        ax4.plot((time % BLSbestPeriod)/BLSbestPeriod, relFlux, 'b,')

        ax5.set_title('BLS - Fold (Time Space)')
        ax5.set_xlabel('Period (days)') 
        ax5.set_ylabel('Relative Flux')
        ax5.plot((time % BLSbestPeriod), relFlux, 'g,')

    except:

        ## ACF fold in phase and time space
        ax2.set_title('ACF - Fold (Phase Space)')
        ax2.set_xlabel('Period (Phase)')  
        ax2.set_ylabel('Relative Flux')

        ax3.set_title('ACF - Fold (Time Space)')
        ax3.set_xlabel('Period (days)') 
        ax3.set_ylabel('Relative Flux')

        ## BLS fold in phase and time space
        ax4.set_title('BLS - Fold (Phase Space)')
        ax4.set_xlabel('Period (Phase)')  
        ax4.set_ylabel('Relative Flux')

        ax5.set_title('BLS - Fold (Time Space)')
        ax5.set_xlabel('Period (days)') 
        ax5.set_ylabel('Relative Flux')
        
        print('*************** ERROR ***************')
        f = open('EBerrors.txt', 'a')
        f.write(file + '\n')
        f.close()
    
    ## Saving
    figName = objName + '.png'
    plt.tight_layout()
    plt.savefig(os.path.join(chartsPath, objManualFlag + "_" + figName), orientation='landscape')
    plt.close()
        
    print("\n" + objName + " complete.")

print('\nPlotting complete.\n')

periods.to_csv('periods.csv', index=False)

end = timer.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
