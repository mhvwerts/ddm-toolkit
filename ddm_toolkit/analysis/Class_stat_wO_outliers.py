#%%
import sys
sys.path.insert(1, '/Users/lancelotbarthe/GitHub/ddm-toolkit-beta-III')

#
import pickle
import numpy as np
from scipy.optimize import curve_fit

#
from ddm_toolkit.analysis.IQR_outliers_removal import IQR_outliers_removal
from ddm_toolkit.analysis.D_map_generator import D_map_generator
from ddm_toolkit.analysis.A__B_averaged_q import A__B_averaged_q
from ddm_toolkit.analysis.a_priori_noise_detector_model import a_priori_noise_detector_model
from ddm_toolkit.analysis.SNR_computation import SNR_computation


#
class stat_wO_outliers:

    def __init__(self,file,vid_size, roi_size):
        '''

        '''

        # Initialize filename to be processed
        self.file = file

        # Initialize video dimension
        self.vid_size = vid_size
        self.roi_size = roi_size
        self.map_size = int(self.vid_size/self.roi_size)


        # Load results data
        with open(self.file, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            res_list = pickle.load(f)

        # Test if ROI option was used
        if roi_size != vid_size :
            D, D_no_outliers, A_q, A_q_no_outliers, B_q, B_q_no_outliers,i_outliers, q = IQR_outliers_removal(res_list)
        else :
            D, A_q, B_q, q = extract_analysis_from_res_list(res_list)
            D_no_outliers, A_q_no_outliers, B_q_no_outliers, q  = D, A_q, B_q, q

        # Store data
        self.D_no_outliers = D_no_outliers
        self.D = D

        self.A_q_no_outliers = A_q_no_outliers
        self.A_q = A_q

        self.B_q_no_outliers = B_q_no_outliers
        self.B_q = B_q

        self.q = q

    def D_map(self):
        # Compute diffusion coefficient map
        self.D_map =  D_map_generator(self.D, self.map_size)


    def compute_avg(self):
        '''

        '''
        # Compute the average value of every-single A_q and B_q
        self.avg_Bq,self.avg_Aq, self.i_true = A__B_averaged_q(self.A_q_no_outliers, self.B_q_no_outliers)

    def fit_apriori_Noise_model(self):
        # Fit the power density of noise
        self.gamma_0, self.pcov = curve_fit(a_priori_noise_detector_model, self.q[self.i_true[0][0]:self.i_true[0][-1]],self.avg_Bq[self.i_true[0][0]:self.i_true[0][-1]])


    def SNR_computation(self):
        '''

        '''

        try:
            # Compute the signal to Noise Ratio
            self.SNR = SNR_computation(self.q, self.avg_Aq, self.avg_Bq)
        except:
            self.compute_avg()
            self.SNR = SNR_computation(self.q, self.avg_Aq, self.avg_Bq)
