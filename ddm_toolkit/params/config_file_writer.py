#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 08:47:41 2021

@author: lancelotbarthe
"""


#%%

'''
  THIS FUNCTION IS TAKING A LOAD OF ARGUMENTS :OOOO  

     SHOULD I WRITE params FILE USING THE CLASS params ?!
'''

def config_file_writer(T, path_name, file_name, file_ext, frame_start, frame_end, seg_bol_str, roi_x, roi_y, roi_size, frame_Npreview, disp_overdrive, \
                      disp_roi_contrast, px_size, frm_period, ISE_type, ISE_Nbuf, ISF_outfpn, ISF_radialaverage, ISF_display_overdrive, \
                      D_guess, config_file_name, apodization, img_overdrive):
    GENERAL_HEADER = str("[general] \n") \
                    + str("# real_video     []    Define the nature of the video to be processed (Expermental is True, else Simulated) \n") \
                    + str("# temp_deg_C     []    Sample's temperature in °C \n") \
                    + str("# \n")

    REAL_VIDEO = 'real_video = True \n'
    TEMPERATURE = 'temp_deg_C = ' + str(T) + '\n'

    VIDEO_FILE_HEADER = str("[videofile] \n") \
                        + str("# pathname                 path/name of file to be processed \n") \
                        + str("# frm_start         []     start frame number (-1: process full file) \n") \
                        + str("# frm_end           []     end frame number (-1: process full file) \n") \
                        + str("# segmentation_flag []     if true activate the segmentation of the video else process the video as a whole \n") \
                        + str("# ROI_x             []     x coordinate of first ROI pixel (pixels) \n") \
                        + str("# ROI_y             []     y coordinate of first ROI pixel (pixels) \n") \
                        + str("# frm_Npreview      []     number of frames to be previewed by previewer \n") \
                        + str("# display_overdrive []     parameter to enhance contrast of video display \n") \
                        + str("# display_ROIcontrast   [] ??????? \n") \
                        + str("# apodization   []         parameter to select apodization type   \n") \
                        + str("# image overdrive  []      Boost image brightness \n") \
                        + str("# \n") \

    PATH_NAME = 'pathname = ' + str(path_name) + str(file_name) + str(file_ext) + '\n'
    FRAME_START = 'frm_start = ' + str(frame_start) + '\n'
    FRAME_END = 'frm_end = ' + str(frame_end) + '\n'
    SEGMENTATION_VIDEO = 'segmentation_flag = ' + str(seg_bol_str) + '\n'
    ''' TO BE WRITTEN PROPERLY '''
    ROI_X = 'ROI_x = ' + str(roi_x) + '\n'
    ROI_Y = 'ROI_y = ' + str(roi_y) + '\n'
    ROI_SIZE = 'ROI_size = ' + str(roi_size) + '\n'
    FRAME_PREVIEW = 'frm_Npreview = ' + str(frame_Npreview) + '\n'
    DISPLAY_OVERDRIVE =  'display_overdrive = ' + str(disp_overdrive) + '\n'
    DISPLAY_ROI_CONTRAST = 'display_ROIcontrast = ' + str(disp_roi_contrast) + '\n'
    APODIZATION = 'apodization =' + str(apodization) + '\n'
    IMG_OVERDRIVE = 'img_overdrive = ' + str(img_overdrive) + '\n'

    REAL_WORLD_HEADER = str("[realworld] \n") \
                        + str("# px_size         [µm]    real-world height/width of a pixel (pixel spacing) \n") \
                        + str("# frm_period      [s]     real-world frame period (1/fps) (frame time spacing) \n") \
                        + str("# \n")

    PX_SIZE = 'px_size = ' + str(px_size) + '\n'
    FRM_PERIOD = 'frm_period = ' +  str(frm_period)  + '\n'

    ISE_ENGINE_HEADER = str("[ISEngine] \n")  \
                        + str("# ISE_type       select type of ImageStructureEngine (0 is basic reference engine) \n") \
                        + str("# ISE_Nbuf []    buffer size of image structure engine \n") \
                        + str("# ISF_radialaverage []    if True : store radially averaged ISF \n if False or not defined: store entire 2D ISF \n") \
                        + str("# ISF_display_overdrive parameter to enhance contrast of video display \n") \
                        + str("# \n")

    ISE_TYPE = 'ISE_type = ' + str(ISE_type) + '\n'
    ISE_NBUF = 'ISE_Nbuf = ' + str(ISE_Nbuf) + '\n'

    ISF_OUTFPN = 'ISF_outfpn = ' + str(ISF_outfpn) + '\n'
    ISF_RADIALAVERAGE = 'ISF_radialaverage = ' + str(ISF_radialaverage) + '\n'
    ISF_DISPLAY_OVERDRIVE = 'ISF_display_overdrive = ' + str(ISF_display_overdrive) + '\n'

    ANALYSIS_BROWNIAN_HEADER = str("[analysis_brownian] \n") \
                                + str("# D_guess    [µm2 s-1]   Initial guess of diffusion coefficient \n") \
                                + str("# \n")

    D_GUESS = 'D_guess = ' + str(D_guess) + '\n'

    FILE = GENERAL_HEADER + REAL_VIDEO + TEMPERATURE + '\n' + VIDEO_FILE_HEADER + PATH_NAME + FRAME_START + FRAME_END + \
            SEGMENTATION_VIDEO + ROI_X + ROI_Y + ROI_SIZE + FRAME_PREVIEW + DISPLAY_OVERDRIVE + DISPLAY_ROI_CONTRAST + APODIZATION+ IMG_OVERDRIVE + '\n' + \
            REAL_WORLD_HEADER + PX_SIZE + FRM_PERIOD + '\n' + ISE_ENGINE_HEADER + ISE_TYPE + ISE_NBUF + ISF_OUTFPN + \
            ISF_RADIALAVERAGE + ISF_DISPLAY_OVERDRIVE + '\n' + ANALYSIS_BROWNIAN_HEADER + D_GUESS

    #print(FILE)

    with open(config_file_name, 'w') as f:
        f.write(FILE)
    f.close()
