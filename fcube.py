# get ginsberg codes
import sys,glob
k="/users/rindebet/local/github/keflavich"
for p in glob.glob(k+"/*"):
    if not p in sys.path:
        sys.path.append(p)

from uvcombine.uvcombine import feather_kernel, fftmerge, feather_compare, feather_plot
from astropy.io import fits

imroot=".ch275"
imroot=""

ichan=275 # what channel of the cube to use?

intname="NRidge_mosaic_12CO_7m.pbcor"+imroot
sdname="SD.regrid"+imroot
sd_hdu=fits.open(sdname+".fits")[0]
int_hdu=fits.open(intname+".fits")[0]


# units are not looked at in feather_compare or fftmerge, so need jy/pix 
import numpy as np
import astropy.wcs
int_WCS=astropy.wcs.WCS(int_hdu)
int_pix=astropy.wcs.utils.proj_plane_pixel_scales(int_WCS)[0] # assume square pixel 
int_bm= np.sqrt(int_hdu.header['BMAJ']*int_hdu.header['BMIN'])
sd_WCS=astropy.wcs.WCS(sd_hdu)
sd_pix=astropy.wcs.utils.proj_plane_pixel_scales(sd_WCS)[0] # assume square pi
sd_bm= np.sqrt(sd_hdu.header['BMAJ']*sd_hdu.header['BMIN'])
int_ppb=(int_bm/int_pix)**2
sd_ppb =( sd_bm/sd_pix )**2


print("INT pix per bm=%f"%int_ppb)
print(" SD pix per bm=%f"%sd_ppb)

int_hdu.data=int_hdu.data/int_ppb # to Jy/pix
sd_hdu.data =sd_hdu.data/sd_ppb # to Jy/pix
sd_hdu.header['BUNIT']='Jy/pix'       
int_hdu.header['BUNIT']='Jy/pix'       

pixscale = int_pix*3600 # 8. #arcsec
lowresfwhm = sd_bm*3600 # 28.3 #arcsec

im_hi = int_hdu.data.real
im_low = sd_hdu.data.real
lowresscalefactor=1
replace_hires = False
lowpassfilterSD = False
deconvSD = False
highresscalefactor=1

if len(sd_hdu.data.shape)==2:
    nax2,nax1 = sd_hdu.data.shape
    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale,)
    fftsum, combim = fftmerge(kfft, ikfft, im_hi*highresscalefactor,
                             im_low*lowresscalefactor,
                             replace_hires=replace_hires,
                             lowpassfilterSD=lowpassfilterSD,
                             deconvSD=deconvSD,)

    int_hdu.data=combim.real
    int_hdu.writeto("combo"+imroot+".fits",overwrite=True)

elif len(sd_hdu.data.shape)==3:
    # we have a cube.  assume nchan,ny,nx
    nchan,nax2,nax1 = sd_hdu.data.shape
    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale,)

    combim=np.zeros(sd_hdu.data.shape)
    for chan in range(nchan):
        fftsum, combimi = fftmerge(kfft, ikfft, im_hi[chan]*highresscalefactor,
                                   im_low[chan]*lowresscalefactor,
                                   replace_hires=replace_hires,
                                   lowpassfilterSD=lowpassfilterSD,
                                   deconvSD=deconvSD,)
        combim[chan]=combimi.real

    combim=combim*int_ppb # to Jy/bm
    int_hdu.header['BUNIT']='Jy/beam'       
    int_hdu.data=combim.real
    int_hdu.writeto("combo"+imroot+".fits",overwrite=True)

    # get 2d version to do the plots:
    sd_hdu.data=sd_hdu.data[ichan]
    sd_hdu.writeto(sdname+".1chan.fits",overwrite=True)
    int_hdu.data=int_hdu.data[ichan]
    int_hdu.writeto(intname+".1chan.fits",overwrite=True)
    int_hdu=fits.open(intname+".1chan.fits")[0]
    sd_hdu=fits.open(sdname+".1chan.fits")[0]
else:
    print("ERROR: sd cube shape=",sd_hdu.data.shape)
    exit

import astropy.units as u
largest_scale = 10*lowresfwhm

import matplotlib.pyplot as pl                       
pl.ion()
pl.figure(0)
stats = feather_compare(int_hdu,
                        sd_hdu,
                        SAS=lowresfwhm*u.arcsec,
                        LAS=largest_scale*u.arcsec,
                        lowresfwhm=lowresfwhm*u.arcsec,
                        beam_divide_lores=False,
                       )
pl.savefig("fcube"+imroot+".feather_compare.png")



combo = feather_plot(int_hdu,sd_hdu,lowresfwhm=lowresfwhm*u.arcsec,match_units=False)
pl.legend()
pl.subplot(221)
pl.imshow(sd_hdu.data,origin="bottom",vmax=0.3)

fftsum, combim = fftmerge(kfft, ikfft, int_hdu.data*highresscalefactor,
                          sd_hdu.data*lowresscalefactor,
                          replace_hires=replace_hires,
                          lowpassfilterSD=lowpassfilterSD,
                          deconvSD=deconvSD,)
pl.subplot(222)
pl.imshow(combim.real,origin="bottom",vmax=1)

pl.savefig("fcube"+imroot+".feather_result.png")
