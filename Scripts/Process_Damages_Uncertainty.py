# Processing data on global damages attributable to individual countries and their uncertainty
#### Christopher Callahan
#### Christopher.W.Callahan.GR@dartmouth.edu

# Dependencies
import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
from rasterio import features
from affine import Affine
import geopandas as gp
import descartes
from scipy import stats
import psutil
import warnings
import time

# Data locations
loc_shp = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/ProcessedCountryShapefile/"
loc_gmst_fair = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/FAIR/GMST/"
loc_panel = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/Panel/"
loc_pop = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/Population/"
loc_damages_hist = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/Damages/Historical/"
loc_damages_country = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/Damages/Country-Attributed/"
loc_out = "/dartfs-hpc/rc/lab/C/CMIG/ccallahan/National_Attribution/Data/Damages/Processed/"

# get command line info
accounting = sys.argv[1] # consumption vs. territorial accounting
funcname = sys.argv[2] # damage function (e.g., BHMSR for BHM short run)
y1_shares = int(sys.argv[3]) # year 1 of emissions shares
y2_shares = int(sys.argv[4]) # year 2 of emissions shares
y1_damages = int(sys.argv[5]) # year 1 of damages calculation
y2_damages = int(sys.argv[6]) # year 2 of damages calculation

# Shapefile
shp = gp.read_file(loc_shp)
iso_shp = shp.ISO3.values

# Number of FAIR runs
n_fair_runs = 250

# read in FAIR GMST to get iso_attr
y1_fair = 1850
y2_fair = 2014
fair_gmst = xr.open_dataset(loc_gmst_fair+"FAIR_GMST_PPE"+str(n_fair_runs)+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y1_fair)+"-"+str(y2_fair)+".nc")
iso_attr = fair_gmst.coords["iso_attr"].values

# type of significance test
sig_test = "ks" #t

# now get global attributable damages and calculate uncertainties

if sig_test=="ks":
    fname_str = funcname+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y1_damages)+"-"+str(y2_damages)
elif sig_test=="t":
    fname_str = funcname+"_"+accounting+"_shares"+str(y1_shares)+"-"+str(y2_shares)+"_"+str(y1_damages)+"-"+str(y2_damages)+"_ttest"

start_time = time.time()

for i in iso_attr:
    print(i)
    #files_list = [x for x in os.listdir(loc_damages_country+i+"/") if ("global" in x)&(fname_str in x)&("fulldist" not in x)]
    file_in = i+"-attributed_global_gdp_damages_"+fname_str+".nc"
    damages_i = xr.open_dataset(loc_damages_country+i+"/"+file_in)
    attr_damages = damages_i.attributable_losses.sum(dim="time")
    attr_benefits = damages_i.attributable_benefits.sum(dim="time")
    # ensemble x member x fair_run x boot

    # calculate the different uncertainties
    damages_sd_total = attr_damages.std(dim=["ensemble","member","fair_run","boot"])
    damages_sd_iv = attr_damages.sel(ensemble="CESM1-SFLE").mean(dim=["fair_run","boot"]).std(dim="member")
    damages_sd_mdl = attr_damages.sel(ensemble="CMIP6").mean(dim=["fair_run","boot"]).std(dim="member")
    damages_sd_boot = attr_damages.mean(dim=["ensemble","member","fair_run"]).std(dim="boot")
    damages_sd_fair = attr_damages.mean(dim=["ensemble","member","boot"]).std(dim="fair_run")

    benefits_sd_total = attr_benefits.std(dim=["ensemble","member","fair_run","boot"])
    benefits_sd_iv = attr_benefits.sel(ensemble="CESM1-SFLE").mean(dim=["fair_run","boot"]).std(dim="member")
    benefits_sd_mdl = attr_benefits.sel(ensemble="CMIP6").mean(dim=["fair_run","boot"]).std(dim="member")
    benefits_sd_boot = attr_benefits.mean(dim=["ensemble","member","fair_run"]).std(dim="boot")
    benefits_sd_fair = attr_benefits.mean(dim=["ensemble","member","boot"]).std(dim="fair_run")

    if i == iso_attr[0]:
        global_damages_sd_total = damages_sd_total.expand_dims("iso_attr")
        global_damages_sd_iv = damages_sd_iv.expand_dims("iso_attr")
        global_damages_sd_mdl = damages_sd_mdl.expand_dims("iso_attr")
        global_damages_sd_boot = damages_sd_boot.expand_dims("iso_attr")
        global_damages_sd_fair = damages_sd_fair.expand_dims("iso_attr")

        global_benefits_sd_total = benefits_sd_total.expand_dims("iso_attr")
        global_benefits_sd_iv = benefits_sd_iv.expand_dims("iso_attr")
        global_benefits_sd_mdl = benefits_sd_mdl.expand_dims("iso_attr")
        global_benefits_sd_boot = benefits_sd_boot.expand_dims("iso_attr")
        global_benefits_sd_fair = benefits_sd_fair.expand_dims("iso_attr")
    else:
        global_damages_sd_total = xr.concat([global_damages_sd_total,damages_sd_total],dim="iso_attr")
        global_damages_sd_iv = xr.concat([global_damages_sd_iv,damages_sd_iv],dim="iso_attr")
        global_damages_sd_mdl = xr.concat([global_damages_sd_mdl,damages_sd_mdl],dim="iso_attr")
        global_damages_sd_boot = xr.concat([global_damages_sd_boot,damages_sd_boot],dim="iso_attr")
        global_damages_sd_fair = xr.concat([global_damages_sd_fair,damages_sd_fair],dim="iso_attr")

        global_benefits_sd_total = xr.concat([global_benefits_sd_total,benefits_sd_total],dim="iso_attr")
        global_benefits_sd_iv = xr.concat([global_benefits_sd_iv,benefits_sd_iv],dim="iso_attr")
        global_benefits_sd_mdl = xr.concat([global_benefits_sd_mdl,benefits_sd_mdl],dim="iso_attr")
        global_benefits_sd_boot = xr.concat([global_benefits_sd_boot,benefits_sd_boot],dim="iso_attr")
        global_benefits_sd_fair = xr.concat([global_benefits_sd_fair,benefits_sd_fair],dim="iso_attr")

#global_attr_damages.coords["iso_attr"] = iso_attr
#global_attr_benefits.coords["iso_attr"] = iso_attr


uncertainty_ds_out = xr.Dataset({"damages_sd_total":(["iso_attr"],global_damages_sd_total),
                                "damages_sd_iv":(["iso_attr"],global_damages_sd_iv),
                                "damages_sd_mdl":(["iso_attr"],global_damages_sd_mdl),
                                "damages_sd_boot":(["iso_attr"],global_damages_sd_boot),
                                "damages_sd_fair":(["iso_attr"],global_damages_sd_fair),
                                "benefits_sd_total":(["iso_attr"],global_benefits_sd_total),
                                "benefits_sd_iv":(["iso_attr"],global_benefits_sd_iv),
                                "benefits_sd_mdl":(["iso_attr"],global_benefits_sd_mdl),
                                "benefits_sd_boot":(["iso_attr"],global_benefits_sd_boot),
                                "benefits_sd_fair":(["iso_attr"],global_benefits_sd_fair)},
                                coords={"iso_attr":("iso_attr",iso_attr)})

uncertainty_ds_out.attrs["creation_date"] = str(datetime.datetime.now())
uncertainty_ds_out.attrs["created_by"] = "Christopher Callahan, Christopher.W.Callahan.GR@dartmouth.edu"
uncertainty_ds_out.attrs["variable_description"] = "uncertainty in global attributable damages and benefits for each country"
uncertainty_ds_out.attrs["created_from"] = os.getcwd()+"/Process_Damages_Uncertainty.py"
uncertainty_ds_out.attrs["ensemble_details"] = "Perturbed-parameter ensemble for carbon cycle and TCR/ECS"
uncertainty_ds_out.attrs["emissions_subtraction"] = "Leave-one-out country attribution using shares of CEDS emissions"

fname_out = loc_out+"global_attributable_benefits_losses_uncertainty_"+fname_str+".nc"
uncertainty_ds_out.to_netcdf(fname_out,mode="w")
print(fname_out,flush=True)

print("--- %s minutes ---" % ((time.time() - start_time)/60.))
