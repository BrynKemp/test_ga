import base64
import random
import string
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from numpy import arange
from scipy.interpolate import make_interp_spline

# matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# def lowess(x, y, f=1. / 3.):
#     """
#     Basic LOWESS smoother with uncertainty.
#     Note:
#         - Not robust (so no iteration) and
#              only normally distributed errors.
#         - No higher order polynomials d=1
#             so linear smoother.
#     """
#     # get some paras
#     xwidth = f * (x.max() - x.min())  # effective width after reduction factor
#     N = len(x)  # number of obs
#     # Don't assume the data is sorted
#     order = np.argsort(x)
#     # storage
#     y_sm = np.zeros_like(y)
#     y_stderr = np.zeros_like(y)
#     # define the weigthing function -- clipping too!
#     tricube = lambda d: np.clip((1 - np.abs(d) ** 3) ** 3, 0, 1)
#     # run the regression for each observation i
#     for i in range(N):
#         dist = np.abs((x[order][i] - x[order])) / xwidth
#         w = tricube(dist)
#         # form linear system with the weights
#         A = np.stack([w, x[order] * w]).T
#         b = w * y[order]
#         ATA = A.T.dot(A)
#         ATb = A.T.dot(b)
#         # solve the syste
#         sol = np.linalg.solve(ATA, ATb)
#         # predict for the observation only
#         yest = A[i].dot(sol)  # equiv of A.dot(yest) just for k
#         place = order[i]
#         y_sm[place] = yest
#         sigma2 = (np.sum((A.dot(sol) - y[order]) ** 2) / N)
#         # Calculate the standard error
#         y_stderr[place] = np.sqrt(sigma2 *
#                                   A[i].dot(np.linalg.inv(ATA)
#                                            ).dot(A[i]))
#     return y_sm, y_stderr


# def getSFHData(gasfh, sfh, sfhdata):
#     gadd = gasfh * 7
#     gaw = gadd // 7
#     gad = gadd - gaw * 7
#
#     if gad == 0:
#         gastr = '%s+0 weeks' % int(gaw)
#         gadw = '%s+0' % int(gaw)
#     else:
#         gastr = '%s+%s weeks' % (int(gaw), int(gad))
#         gadw = '%s+%s' % (int(gaw), int(gad))
#
#     validate = 0
#     centile_str = ''
#
#     try:
#         result_index = sfhdata['GA'].sub(gasfh).abs().idxmin()
#         sfhrow = sfhdata.loc[result_index]
#         sfhzscore = (sfh - sfhrow.iloc[4]) / sfhrow.iloc[5]
#         sfhcentile = round(st.norm.cdf(sfhzscore) * 100)
#         centile_str = 'SFH %scm = %sth centile at %s gestation' % (sfh, sfhcentile, gastr)
#
#         if str(sfhcentile).endswith('1'):
#             centile_str = 'SFH %scm = %sst centile at %s\' gestation' % (sfh, sfhcentile, gastr)
#         elif str(sfhcentile).endswith('2'):
#             centile_str = 'SFH %scm = %snd centile at %s\' gestation' % (sfh, sfhcentile, gastr)
#         elif str(sfhcentile).endswith('3'):
#             centile_str = 'SFH %scm = %srd centile at %s\' gestation' % (sfh, sfhcentile, gastr)
#         validate = 1
#     except:
#         validate = 2
#
#     return centile_str, gadw, sfhcentile


# def get_graph():
#     buffer = BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     image_png = buffer.getvalue()
#     graph = base64.b64encode(image_png)
#     graph = graph.decode('utf-8')
#     buffer.close()
#     return graph


# def get_plot(centile_str, gaset, sfhdata, gadw1, gadw2, sfd, lfd):
#
#     if sfd == 1:
#         refstring = "Referral indicated:  SFH<10th centile"
#     else:
#         refstring = "Action - continue routine care"
#
#     if lfd == 1:
#         refstring = "Referral indicated:  SFH>90th centile x2"
#     else:
#         refstring = "Action - continue routine care"
#
#     plt.figure(figsize=(10, 6), dpi=200)
#     plt.plot(sfhdata.loc[:, 'GA'], sfhdata.loc[:, 'c95'], '-.', linewidth=2, color='#3860B2', label='95th centile')
#     plt.plot(sfhdata.loc[:, 'GA'], sfhdata.loc[:, 'c50'], '-', linewidth=2, color='#3860B2', label='50th centile')
#     plt.plot(sfhdata.loc[:, 'GA'], sfhdata.loc[:, 'c10'], '-.', linewidth=2, color='#3860B2', label='10th centile')
#
#     if sfd > 0 or lfd > 0:
#         plt.plot(gaset.loc[:, 'GA'], gaset.loc[:, 'SFH'], 'o', color='red', ms=10)
#     else:
#         plt.plot(gaset.loc[:, 'GA'], gaset.loc[:, 'SFH'], 'o', color='lime', ms=10)
#
#     spline = make_interp_spline(gaset.loc[:, 'GA'], gaset.loc[:, 'SFH'])
#
#     idxtemp1a = ''
#     idxtemp2a = ''
#     idx1temp = 0
#     idx2temp = 0
#     plotidx = ['0', '3', '5']
#
#     idxtemp1a = [idx for idx, item in enumerate(sfhdata['GAstr']) if gadw1 in item]
#     if any(x in plotidx for x in gadw1[len(gadw1) - 1]) is True:
#         idxtemp1a = [idx for idx, item in enumerate(sfhdata['GAstr']) if gadw1 in item]
#         for c in range(len(idxtemp1a)):
#             if sfhdata.loc[idxtemp1a[c], 'GAdw'] == sfhdata.loc[idxtemp1a[c], 'GAstr']:
#                 idx1temp = idxtemp1a[c]
#     else:
#         idx1temp = idxtemp1a[0]
#
#     idxtemp2a = [idx for idx, item in enumerate(sfhdata['GAstr']) if gadw2 in item]
#     if any(x in plotidx for x in gadw1[len(gadw1) - 1]) is True:
#         idxtemp2a = [idx for idx, item in enumerate(sfhdata['GAstr']) if gadw2 in item]
#         for c in range(len(idxtemp2a)):
#             if sfhdata.loc[idxtemp2a[c], 'GAdw'] == sfhdata.loc[idxtemp2a[c], 'GAstr']:
#                 idx1temp = idxtemp2a[c]
#     else:
#         idx2temp = idxtemp2a[len(idxtemp2a) - 1]
#
#     setlength = idx2temp - idx1temp + 2
#
#     spline_x = np.linspace(sfhdata.loc[idx1temp, 'GA'], sfhdata.loc[idx2temp, 'GA'], setlength)
#     spline_y = spline(spline_x)
#
#     sfhreturn = pd.DataFrame(index=range(len(sfhdata)), columns=range(3))
#     sfhreturn.columns = ['idx', 'GAdw', 'SFHplot']
#     sfhreturn.loc[:, 'idx'] = sfhdata.loc[:, 'idx']
#     sfhreturn.loc[:, 'GAdw'] = sfhdata.loc[:, 'GAdw']
#     splineidx = 0
#
#     for a in arange(idx1temp, idx2temp, 1):
#         sfhreturn.loc[a, 'SFHplot'] = spline_y[splineidx]
#         splineidx = splineidx + 1
#
#     if sfd > 0 or lfd > 0:
#         plt.plot(spline_x, spline_y, '-', linewidth=3, color='red')
#     else:
#         plt.plot(spline_x, spline_y, '-', linewidth=3, color='lime')
#
#     plt.grid(alpha=.4, linestyle='--')
#     plt.axis([16, 44, 16, 44])
#     plt.xticks(np.arange(16, 44, step=2))
#     plt.yticks(np.arange(14, 44, step=2))
#
#     graph = get_graph()
#     graph_str = 'data:image/png;base64,' + graph
#
#     plt.title(
#         refstring,
#         fontsize=14,
#         fontweight="normal",
#         pad=10,
#         loc="left"
#     )
#
#     plt.suptitle(
#         centile_str,
#         fontsize=16,
#         fontweight="bold",
#         x=0.126,
#         y=0.98,
#         ha="left",
#     )
#
#     plt.xlabel('Gestational age [weeks]')
#     plt.ylabel('SFH [cm]')
#
#     graph2 = get_graph()
#     graph2_str = 'data:image/png;base64,' + graph2
#
#     uid_graph = id_generator()
#
#     return graph_str, graph2_str, uid_graph


# with open('C:/Users/Brynk/SynologyWhole/SynologyDrive/2019/test.png', 'wb') as fh:
#     fh.write(base64.b64decode(graph))


def getCentile(gainput, bw, gender):
    gadd = gainput
    gaww = gainput // 7
    gaw = gainput / 7
    gad = int(gadd - (gaww * 7))

    if gad == 0:
        gastr = '%s weeks' % gaww
    else:
        gastr = '%s+%s weeks' % (gaww, gad)

    centile = None
    centile_str = None

    filename_data_uk90M = str(Path.cwd() / 'static/uk90M.csv')
    filename_data_uk90F = str(Path.cwd() / 'static/uk90F.csv')
    filename_data_ukwhoM = str(Path.cwd() / 'static/ukwhoM.csv')
    filename_data_ukwhoF = str(Path.cwd() / 'static/ukwhoF.csv')

    validate = 0
    try:
        uk90m = pd.read_csv(filename_data_uk90M)
        uk90f = pd.read_csv(filename_data_uk90F)
        whoptm = pd.read_csv(filename_data_ukwhoM)
        whoptf = pd.read_csv(filename_data_ukwhoF)

    except:
        validate = 2

    var_males = ['ML', 'MM', 'MS', 'm1', 'm2', 'm3', 'm10', 'm25', 'm50', 'm75', 'm90', 'm95', 'm97', 'm98', 'm99']
    var_females = ['FL', 'FM', 'FS', 'f1', 'f2', 'f3', 'f10', 'f25', 'f50', 'f75', 'f90', 'f95', 'f97', 'f98', 'f99']
    var_dates = ['ga']

    linedata = []
    scatterdata = []
    ga = []
    cent_ref = ''

    if gender == 1 and gaw < 32:
        linedata = uk90f[var_females]
        ga = uk90f[var_dates]
        # scatterdata = uk90_centfemale_matrix
        cent_ref = "UK 1990 Growth Centiles"

    elif gender == 1 and gaw >= 32:
        linedata = whoptf[var_females]
        ga = whoptf[var_dates]
        # scatterdata = whoptm_centfemale_matrix
        cent_ref = "WHO Preterm Centiles"

    elif gender == 2 and gaw < 32:
        linedata = uk90m[var_males]
        ga = uk90m[var_dates]
        # scatterdata = uk90_centmale_matrix
        cent_ref = "UK 1990 Growth Centiles"

    elif gender == 2 and gaw >= 32:
        linedata = whoptm[var_males]
        ga = whoptm[var_dates]
        # scatterdata = whoptm_centmale_matrix
        cent_ref = "WHO Preterm Centiles"

    row_lms = np.argmin(np.abs(np.array(ga) - gaw))

    if gender == 1:
        var_num = ((bw / (np.array(linedata.loc[row_lms, 'FM']) * 1000)) ** np.array(linedata.loc[row_lms, 'FL'])) - 1
        var_denom = np.array(linedata.loc[row_lms, 'FL']) * np.array(linedata.loc[row_lms, 'FS'])
        varz = var_num / var_denom
        centile = int(round((st.norm.cdf(varz) * 100), 0))
        centile_str = 'Female:  Birthweight %s g at %s = %s centile' % (bw, gastr, centile)

    elif gender == 2:
        var_num = ((bw / (np.array(linedata.loc[row_lms, 'MM']) * 1000)) ** np.array(linedata.loc[row_lms, 'ML'])) - 1
        var_denom = np.array(linedata.loc[row_lms, 'ML']) * np.array(linedata.loc[row_lms, 'MS'])
        varz = var_num / var_denom
        centile = int(round((st.norm.cdf(varz) * 100), 0))
        centile_str = 'Male:  Birthweight %s g at %s = %s centile' % (bw, gastr, centile)

    return bw, gastr, centile, cent_ref
    # return render(request, "chart2.html", {'active_page': 'chart2.html', 'div_figure': html_fig})


# def testPlot():
#     gaset = pd.DataFrame(np.zeros([6, 2]))
#     gaset.columns = ['GA', 'SFH']
#     gaset['GA'] = [23.454, 27.878, 30.212, 34.434, 36.878, 39.89]
#     gaset['SFH'] = [24, 26, 29, 32, 36, 29]
#
#     plt.figure(figsize=(8, 6), dpi=200)
#     plt.plot(gaset.loc[:, 'GA'], gaset.loc[:, 'SFH'], '-', color='green', ms=3)
#
#     plt.plot(gaset['GA'], gaset['SFH'], '-', linewidth=2, color='green')
#     plt.title('banana', loc='left')
#     plt.xlabel('Gestational age [weeks]')
#     plt.ylabel('SFH [cm]')
#     plt.grid(alpha=.4, linestyle='--')
#     plt.axis([22, 40, 22, 40])
#     plt.xticks(np.arange(22, 40, step=2))
#     plt.yticks(np.arange(22, 40, step=2))
#     plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
#     plt.tight_layout()
#     graph = get_graph()
#
#     return graph
